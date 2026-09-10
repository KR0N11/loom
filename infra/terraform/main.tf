# Loom AWS deployment path: S3 for the DuckDB file and memos, a Lambda executor
# that runs the sandbox runner, RDS Postgres 16 for pgvector, and secrets.
# This module has not been applied against a real account; treat it as a
# reviewed starting point, not a tested deployment.

locals {
  name = "${var.project}-${var.environment}"
  tags = {
    Project     = var.project
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# ---------- Networking: default VPC for simplicity ----------

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# ---------- Storage: DuckDB file and memos ----------

resource "aws_s3_bucket" "data" {
  bucket_prefix = "${local.name}-data-"
  tags          = local.tags
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# ---------- Secrets ----------

resource "random_password" "db" {
  length  = 32
  special = false
}

resource "aws_secretsmanager_secret" "db_password" {
  name_prefix = "${local.name}/db-password-"
  tags        = local.tags
}

resource "aws_secretsmanager_secret_version" "db_password" {
  secret_id     = aws_secretsmanager_secret.db_password.id
  secret_string = random_password.db.result
}

resource "aws_secretsmanager_secret" "anthropic_api_key" {
  name_prefix = "${local.name}/anthropic-api-key-"
  tags        = local.tags
}

resource "aws_secretsmanager_secret_version" "anthropic_api_key" {
  count         = var.anthropic_api_key == "" ? 0 : 1
  secret_id     = aws_secretsmanager_secret.anthropic_api_key.id
  secret_string = var.anthropic_api_key
}

# ---------- RDS Postgres 16 with pgvector ----------

resource "aws_security_group" "db" {
  name        = "${local.name}-db"
  description = "Postgres access for Loom services"
  vpc_id      = data.aws_vpc.default.id
  tags        = local.tags

  ingress {
    description     = "Postgres from the executor Lambda"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.lambda.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_db_subnet_group" "db" {
  name       = "${local.name}-db"
  subnet_ids = data.aws_subnets.default.ids
  tags       = local.tags
}

# pgvector ships with RDS Postgres 16; it is enabled per database with
# `CREATE EXTENSION vector` (see scripts/init_pgvector.sql), not via a parameter.
resource "aws_db_parameter_group" "db" {
  name   = "${local.name}-pg16"
  family = "postgres16"
  tags   = local.tags

  parameter {
    name  = "log_min_duration_statement"
    value = "1000"
  }
}

resource "aws_db_instance" "db" {
  identifier             = "${local.name}-pg"
  engine                 = "postgres"
  engine_version         = "16"
  instance_class         = var.db_instance_class
  allocated_storage      = var.db_allocated_storage
  db_name                = var.project
  username               = var.db_username
  password               = random_password.db.result
  db_subnet_group_name   = aws_db_subnet_group.db.name
  vpc_security_group_ids = [aws_security_group.db.id]
  parameter_group_name   = aws_db_parameter_group.db.name
  storage_encrypted      = true
  publicly_accessible    = false
  skip_final_snapshot    = true
  deletion_protection    = false
  tags                   = local.tags
}

# ---------- Lambda executor ----------

resource "aws_security_group" "lambda" {
  name        = "${local.name}-lambda"
  description = "Executor Lambda"
  vpc_id      = data.aws_vpc.default.id
  tags        = local.tags

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda" {
  name               = "${local.name}-executor"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
  tags               = local.tags
}

data "aws_iam_policy_document" "lambda" {
  statement {
    sid       = "ReadDuckDB"
    actions   = ["s3:GetObject", "s3:ListBucket"]
    resources = [aws_s3_bucket.data.arn, "${aws_s3_bucket.data.arn}/*"]
  }

  statement {
    sid       = "Logs"
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["arn:aws:logs:*:*:*"]
  }

  statement {
    sid       = "Vpc"
    actions   = ["ec2:CreateNetworkInterface", "ec2:DescribeNetworkInterfaces", "ec2:DeleteNetworkInterface"]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "lambda" {
  name   = "${local.name}-executor"
  role   = aws_iam_role.lambda.id
  policy = data.aws_iam_policy_document.lambda.json
}

resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/${local.name}-executor"
  retention_in_days = 14
  tags              = local.tags
}

# Package: zip built from infra/lambda/handler.py plus src/loom/sandbox with
# duckdb, pandas and pyarrow vendored (see infra/terraform/README.md).
resource "aws_lambda_function" "executor" {
  function_name = "${local.name}-executor"
  role          = aws_iam_role.lambda.arn
  runtime       = "python3.12"
  handler       = "handler.lambda_handler"
  filename      = var.lambda_package_path
  memory_size   = var.lambda_memory_mb
  timeout       = var.lambda_timeout_s
  tags          = local.tags

  ephemeral_storage {
    size = 2048
  }

  vpc_config {
    subnet_ids         = data.aws_subnets.default.ids
    security_group_ids = [aws_security_group.lambda.id]
  }

  environment {
    variables = {
      LOOM_DUCKDB_S3_URI = "s3://${aws_s3_bucket.data.bucket}/duckdb/loom.duckdb"
      LOOM_SANDBOX_ROW_LIMIT = "10000"
    }
  }

  depends_on = [aws_cloudwatch_log_group.lambda]
}
