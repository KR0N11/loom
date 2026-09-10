"""Central settings for Loom.

Approach: one pydantic-settings object read from environment / .env so every
module (LLM provider, vector repository, sandbox) gets its configuration from
one place and swapping a backend is a config change, not a code change.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProviderName(StrEnum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    CLAUDE_CODE = "claude_code"


class VectorBackend(StrEnum):
    PGVECTOR = "pgvector"
    DUCKDB = "duckdb"


class SandboxBackend(StrEnum):
    DOCKER = "docker"
    SUBPROCESS = "subprocess"
    LAMBDA = "lambda"


class EmbeddingBackend(StrEnum):
    HF = "hf"
    HASHING = "hashing"


class Settings(BaseSettings):
    """All runtime knobs. Prefix LOOM_ unless the variable is a vendor standard."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="LOOM_", extra="ignore")

    # LLM
    llm_provider: LLMProviderName = LLMProviderName.ANTHROPIC
    llm_model: str = "claude-sonnet-5"
    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    aws_region: str = Field(default="us-east-1", validation_alias="AWS_REGION")
    bedrock_model_id: str = "anthropic.claude-sonnet-5-v1:0"
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.0

    # Storage
    duckdb_path: Path = Path("data/duckdb/loom.duckdb")
    vector_backend: VectorBackend = VectorBackend.DUCKDB
    pg_dsn: str = "postgresql://loom:loom@localhost:5432/loom"
    vector_store_path: Path = Path("data/duckdb/semantic.duckdb")
    checkpoint_path: Path = Path("data/checkpoints/loom.sqlite")

    # Sandbox
    sandbox_backend: SandboxBackend = SandboxBackend.SUBPROCESS
    sandbox_timeout_s: int = 30
    sandbox_memory_mb: int = 1024
    sandbox_row_limit: int = 10_000
    sandbox_docker_image: str = "loom-sandbox:latest"
    lambda_function_name: str = "loom-executor"

    # Agents
    executor_max_retries: int = 3
    retrieval_top_k: int = 8

    # Models
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    embedding_backend: EmbeddingBackend = EmbeddingBackend.HF
    zero_shot_model: str = "facebook/bart-large-mnli"
    small_sql_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    tapas_model: str = "google/tapas-base-finetuned-wtq"

    # Tracing
    langfuse_public_key: str | None = Field(default=None, validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, validation_alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com", validation_alias="LANGFUSE_HOST"
    )

    # Paths
    benchmark_dir: Path = Path("benchmarks")
    semantic_dir: Path = Path("data/semantic")
    raw_dir: Path = Path("data/raw")
    output_dir: Path = Path("outputs")


def get_settings() -> Settings:
    """Build a fresh Settings; cheap enough to call per request and easy to override in tests."""
    return Settings()
