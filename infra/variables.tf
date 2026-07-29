variable "prefix" {
  description = "Name prefix for all resources (lowercase alphanumeric; ACR name must be globally unique)."
  type        = string
  default     = "agenticrag"
}

variable "location" {
  description = "Azure region."
  type        = string
  default     = "eastus"
}

variable "image_name" {
  description = "Container image repository name in ACR."
  type        = string
  default     = "agentic-rag-backend"
}

variable "image_tag" {
  description = "Container image tag."
  type        = string
  default     = "latest"
}

variable "provider_name" {
  description = "LLM provider the backend uses (azure | gemini)."
  type        = string
  default     = "azure"
}

variable "azure_openai_key" {
  description = "Azure OpenAI API key (passed as a Container App secret)."
  type        = string
  sensitive   = true
  default     = ""
}

variable "azure_openai_endpoint" {
  description = "Azure OpenAI endpoint URL."
  type        = string
  default     = ""
}

variable "azure_deployment_name" {
  description = "Azure OpenAI deployment name."
  type        = string
  default     = "gpt-5-mini"
}

variable "azure_openai_api_version" {
  description = "Azure OpenAI API version."
  type        = string
  default     = "2024-12-01-preview"
}

variable "gemini_api_key" {
  description = "Google Gemini API key (passed as a Container App secret)."
  type        = string
  sensitive   = true
  default     = ""
}

variable "qdrant_url" {
  description = "Qdrant URL reachable from the container (e.g. a managed Qdrant or a sidecar)."
  type        = string
  default     = "http://localhost:6333"
}