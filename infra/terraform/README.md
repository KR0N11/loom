# Terraform: Loom AWS deployment path

Status: written and reviewed, **not applied against a real AWS account**.
Terraform was not available on the build machine, so `terraform plan` has
not been run. Expect small fixes on first apply.

## What it creates

| Resource | Purpose |
|---|---|
| S3 bucket (versioned, encrypted, private) | DuckDB file at `duckdb/loom.duckdb`, memos and eval outputs |
| Lambda `loom-<env>-executor` (python3.12, 2048 MB, 60 s) | Runs `loom.sandbox.runner` on a `SandboxRequest`; downloads the DuckDB file from S3 to `/tmp` |
| IAM role | S3 read on the data bucket, CloudWatch logs, VPC ENIs |
| RDS Postgres 16 (`db.t4g.micro` by default) | Semantic layer with the `vector` extension |
| Secrets Manager | Database password (generated) and the Anthropic API key (optional input) |
| Security groups | Lambda → Postgres on 5432 only |

Networking uses the account's default VPC through data sources to keep the
module short.

## Usage

```bash
# Build the Lambda package: handler + loom.sandbox + vendored deps
cd infra/lambda
pip install duckdb pandas pyarrow pydantic -t build/
cp handler.py build/ && cp -r ../../src/loom build/loom
(cd build && zip -qr ../package.zip .)

cd ../terraform
terraform init
terraform plan -var region=us-east-1 -var environment=dev
TF_VAR_anthropic_api_key=sk-ant-... terraform apply
```

After apply:

1. Upload the DuckDB file: `aws s3 cp data/duckdb/loom.duckdb $(terraform output -raw duckdb_s3_uri)`.
2. Enable pgvector once: `psql "$LOOM_PG_DSN" -f ../../scripts/init_pgvector.sql`.
3. Set `LOOM_SANDBOX_BACKEND=lambda`, `LOOM_LAMBDA_FUNCTION_NAME=$(terraform output -raw lambda_function_name)`,
   `LOOM_VECTOR_BACKEND=pgvector` and `LOOM_PG_DSN` from the outputs and the secret.

## Notes

- The Lambda runs inside the VPC so it can reach RDS; it therefore needs a NAT
  or VPC endpoints for S3 access. The simplest fix is an S3 gateway endpoint
  on the default VPC, not included here.
- `skip_final_snapshot` and `deletion_protection = false` are development
  settings; change both for production.
