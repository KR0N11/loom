output "data_bucket" {
  description = "S3 bucket holding the DuckDB file and memos."
  value       = aws_s3_bucket.data.bucket
}

output "duckdb_s3_uri" {
  description = "Where the ingestion job should upload the DuckDB file."
  value       = "s3://${aws_s3_bucket.data.bucket}/duckdb/loom.duckdb"
}

output "lambda_function_name" {
  description = "Set LOOM_LAMBDA_FUNCTION_NAME to this value."
  value       = aws_lambda_function.executor.function_name
}

output "db_endpoint" {
  description = "Postgres endpoint for LOOM_PG_DSN."
  value       = aws_db_instance.db.address
}

output "db_password_secret_arn" {
  description = "Secrets Manager ARN for the database password."
  value       = aws_secretsmanager_secret.db_password.arn
}

output "anthropic_api_key_secret_arn" {
  description = "Secrets Manager ARN for the Anthropic API key."
  value       = aws_secretsmanager_secret.anthropic_api_key.arn
}
