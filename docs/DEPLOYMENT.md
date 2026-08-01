# Deployment Notes

## Kubernetes (in progress)
Local cluster via kind (Docker Desktop, k8s v1.36.1). Planned: backend Deployment + Service,
Qdrant/Neo4j as in-cluster services, Helm chart, HPA autoscaling, Prometheus + Grafana over
the existing /metrics endpoint. Paused pending local disk cleanup.

## Current deployment paths
- Local: docker compose up -d (Qdrant, Neo4j, Jaeger) + uvicorn backend
- Cloud: Terraform (Azure Container Apps) in infra/