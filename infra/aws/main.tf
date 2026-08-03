terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
  }
}

provider "aws" {
  region = var.region
}

data "aws_caller_identity" "current" {}

# --- Networking: use the default VPC + subnets for a simple public Fargate service ---
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# --- ECR: container registry (mirrors Azure ACR) ---
resource "aws_ecr_repository" "backend" {
  name                 = "${var.prefix}-backend"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

# --- CloudWatch log group (mirrors Log Analytics workspace) ---
resource "aws_cloudwatch_log_group" "backend" {
  name              = "/ecs/${var.prefix}-backend"
  retention_in_days = 30
}

# --- Secrets in SSM Parameter Store (mirrors Container App secrets — not baked into image) ---
resource "aws_ssm_parameter" "azure_openai_key" {
  name  = "/${var.prefix}/azure-openai-key"
  type  = "SecureString"
  value = var.azure_openai_key != "" ? var.azure_openai_key : "placeholder"
}

resource "aws_ssm_parameter" "gemini_api_key" {
  name  = "/${var.prefix}/gemini-api-key"
  type  = "SecureString"
  value = var.gemini_api_key != "" ? var.gemini_api_key : "placeholder"
}

# --- IAM: task execution role (pull image, write logs, read secrets) ---
resource "aws_iam_role" "task_execution" {
  name = "${var.prefix}-task-exec"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "task_execution" {
  role       = aws_iam_role.task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Allow the execution role to read the two SSM secrets.
resource "aws_iam_role_policy" "read_secrets" {
  name = "${var.prefix}-read-secrets"
  role = aws_iam_role.task_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["ssm:GetParameters"]
      Resource = [
        aws_ssm_parameter.azure_openai_key.arn,
        aws_ssm_parameter.gemini_api_key.arn,
      ]
    }]
  })
}

# --- Security group: allow inbound on the app port ---
resource "aws_security_group" "backend" {
  name        = "${var.prefix}-backend-sg"
  description = "Allow inbound to the backend container"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# --- ECS cluster + Fargate service (mirrors the Container App) ---
resource "aws_ecs_cluster" "main" {
  name = "${var.prefix}-cluster"
}

resource "aws_ecs_task_definition" "backend" {
  family                   = "${var.prefix}-backend"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "1024" # 1 vCPU  (mirrors Azure cpu = 1.0)
  memory                   = "2048" # 2 GB    (mirrors Azure memory = 2Gi)
  execution_role_arn       = aws_iam_role.task_execution.arn

  container_definitions = jsonencode([{
    name      = "backend"
    image     = "${aws_ecr_repository.backend.repository_url}:${var.image_tag}"
    essential = true

    portMappings = [{
      containerPort = 8000
      protocol      = "tcp"
    }]

    environment = [
      { name = "PROVIDER", value = var.provider_name },
      { name = "PORT", value = "8000" },
      { name = "AZURE_OPENAI_ENDPOINT", value = var.azure_openai_endpoint },
      { name = "AZURE_DEPLOYMENT_NAME", value = var.azure_deployment_name },
      { name = "AZURE_OPENAI_API_VERSION", value = var.azure_openai_api_version },
      { name = "QDRANT_URL", value = var.qdrant_url },
    ]

    secrets = [
      { name = "AZURE_OPENAI_KEY", valueFrom = aws_ssm_parameter.azure_openai_key.arn },
      { name = "GEMINI_API_KEY", valueFrom = aws_ssm_parameter.gemini_api_key.arn },
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.backend.name
        "awslogs-region"        = var.region
        "awslogs-stream-prefix" = "backend"
      }
    }
  }])
}

resource "aws_ecs_service" "backend" {
  name            = "${var.prefix}-backend"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.backend.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.backend.id]
    assign_public_ip = true
  }
}

# --- Application Auto Scaling: approximate Azure's scale-to-zero (min 0) ---
# NOTE: unlike Azure Container Apps, Fargate has no request-driven scale-to-zero.
# This scales on CPU; min_capacity can be 0 but there's no built-in wake-on-request,
# so cold starts differ. See infra/aws/README.md.
resource "aws_appautoscaling_target" "backend" {
  max_capacity       = var.max_count
  min_capacity       = var.min_count
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.backend.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "cpu" {
  name               = "${var.prefix}-cpu-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.backend.resource_id
  scalable_dimension = aws_appautoscaling_target.backend.scalable_dimension
  service_namespace  = aws_appautoscaling_target.backend.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value = 70
  }
}