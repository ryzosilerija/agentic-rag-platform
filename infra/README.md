# Infrastructure — Azure Container Apps

Terraform to deploy the Agentic RAG Platform backend to Azure Container Apps.

## What it provisions

- **Resource group** — container for all resources
- **Azure Container Registry (ACR)** — holds the backend image
- **Log Analytics workspace** — required by Container Apps
- **Container Apps environment** — the managed runtime
- **Container App** — the backend, scale-to-zero, HTTPS ingress on :8000

LLM keys are injected as Container App **secrets**, never baked into the image.

## Deploy

```bash
# 1. Build & push the image
az acr login --name <prefix>acr
docker build -t <acr-login-server>/agentic-rag-backend:latest .
docker push <acr-login-server>/agentic-rag-backend:latest

# 2. Provision
cd infra
terraform init
terraform validate
cp terraform.tfvars.example terraform.tfvars   # fill in secrets
terraform plan
terraform apply

# 3. Get the public URL
terraform output backend_url
```

## Notes

- `min_replicas = 0` scales to zero when idle, so the app costs nothing between requests.
- Qdrant is expected at `qdrant_url`; a full production deploy would run managed Qdrant or a second Container App. This config wires the variable through.
- `terraform.tfvars` (real secrets) is gitignored; only `terraform.tfvars.example` is committed.