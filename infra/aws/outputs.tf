output "ecr_repository_url" {
  description = "ECR repository URL — docker push target."
  value       = aws_ecr_repository.backend.repository_url
}

output "ecs_cluster_name" {
  description = "ECS cluster name."
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "ECS Fargate service name."
  value       = aws_ecs_service.backend.name
}

output "log_group" {
  description = "CloudWatch log group for backend logs."
  value       = aws_cloudwatch_log_group.backend.name
}