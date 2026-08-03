variable "prefix" {
  description = "Name prefix for all resources."
  type        = string
  default     = "agenticrag"
}

variable "region" {
  description = "AWS region."
  type        = string
  default     = "us-east-1"
}

variable "image_tag" {
  description = "Container image tag in ECR."
  type        = string
  default     = "latest"
}

variable "provider_name" {
  description = "LLM provider the backend uses (azure | gemini)."
  type        = string
  default     = "gemini"
}

variable "desired_count" {
  description = "Number of Fargate tasks to run."
  type        = number
  default     = 1
}

variable "min_count" {
  description = "Autoscaling minimum task count (0 approximates scale-to-zero; see README)."
  type        = number
  default     = 0
}

variable "max_count" {
  description = "Autoscaling maximum task count."
  type        = number
  default     = 2
}

variable "azure_openai_key" {
  description = "Azure OpenAI API key (stored in SSM Parameter Store)."
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
  description = "Google Gemini API key (stored in SSM Parameter Store)."
  type        = string
  sensitive   = true
  default     = ""
}

variable "qdrant_url" {
  description = "Qdrant URL reachable from the container."
  type        = string
  default     = "http://localhost:6333"
}