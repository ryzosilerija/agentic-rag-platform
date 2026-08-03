# AWS deployment (Terraform)

A second infrastructure-as-code target for the platform: the same backend
container deployed to **AWS ECS Fargate**, parallel to the Azure Container
Apps stack in `../`. Demonstrates provider-agnostic, multi-cloud IaC.

## What it provisions

- **ECR** — container registry (mirrors Azure ACR)
- **CloudWatch log group** — logs (mirrors Log Analytics)
- **ECS cluster + Fargate service + task definition** — the backend
  (mirrors the Azure Container App); 1 vCPU / 2 GB
- **SSM Parameter Store** — LLM keys as SecureString secrets, injected at
  runtime, never baked into the image (mirrors Container App secrets)
- **IAM task-execution role** — least-privilege: pull image, write logs,
  read only the two secret parameters
- **Security group** — inbound on :8000
- **Application Auto Scaling** — CPU-target scaling

## Scale-to-zero difference (honest note)

Azure Container Apps scales to zero on no traffic and wakes on request.
Fargate has **no request-driven scale-to-zero**: `min_count = 0` is allowed
and autoscaling reacts to CPU, but there is no built-in wake-on-request, so
a request arriving at zero tasks won't cold-start the service the way Azure
does. For a live demo, keep `min_count = 1`. This is a genuine platform
difference between the two clouds, not a config gap.

## Usage

```bash
cd infra/aws
terraform init
terraform validate
terraform plan \
  -var="gemini_api_key=$GEMINI_API_KEY" \
  -var="provider_name=gemini"
# terraform apply  # requires a funded AWS account
```

Validated with `terraform init` + `terraform validate` (no apply, $0),
same approach as the Azure target.