variable "region" {
  description = "AWS region for all resources."
  type        = string
  default     = "us-east-1"
}

variable "project" {
  description = "Project name used as a prefix for resource names."
  type        = string
  default     = "loom"
}

variable "environment" {
  description = "Deployment environment tag (dev, staging, prod)."
  type        = string
  default     = "dev"
}

variable "db_instance_class" {
  description = "RDS instance class for the pgvector Postgres."
  type        = string
  default     = "db.t4g.micro"
}

variable "db_allocated_storage" {
  description = "RDS storage in GB."
  type        = number
  default     = 20
}

variable "db_username" {
  description = "Master username for Postgres."
  type        = string
  default     = "loom"
}

variable "lambda_memory_mb" {
  description = "Memory for the executor Lambda."
  type        = number
  default     = 2048
}

variable "lambda_timeout_s" {
  description = "Timeout for the executor Lambda."
  type        = number
  default     = 60
}

variable "lambda_package_path" {
  description = "Path to the zipped Lambda deployment package built from infra/lambda and src/loom/sandbox."
  type        = string
  default     = "../lambda/package.zip"
}

variable "anthropic_api_key" {
  description = "Anthropic API key stored in Secrets Manager. Pass via TF_VAR_anthropic_api_key, never commit."
  type        = string
  sensitive   = true
  default     = ""
}
