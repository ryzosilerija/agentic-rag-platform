output "resource_group" {
  description = "Resource group name."
  value       = azurerm_resource_group.rg.name
}

output "acr_login_server" {
  description = "Container registry login server — docker push target."
  value       = azurerm_container_registry.acr.login_server
}

output "backend_url" {
  description = "Public HTTPS URL of the deployed backend."
  value       = "https://${azurerm_container_app.backend.ingress[0].fqdn}"
}