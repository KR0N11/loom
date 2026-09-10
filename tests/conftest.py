"""Shared fixtures: an in-memory DuckDB with tiny tables, a scripted LLM, offline settings."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from loom.config import EmbeddingBackend, SandboxBackend, Settings, VectorBackend
from loom.llm.fake_provider import FakeProvider


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        duckdb_path=tmp_path / "loom.duckdb",
        vector_store_path=tmp_path / "semantic.duckdb",
        semantic_dir=tmp_path / "semantic",
        checkpoint_path=tmp_path / "ckpt.sqlite",
        vector_backend=VectorBackend.DUCKDB,
        sandbox_backend=SandboxBackend.SUBPROCESS,
        embedding_backend=EmbeddingBackend.HASHING,
        output_dir=tmp_path / "outputs",
        benchmark_dir=tmp_path / "benchmarks",
        anthropic_api_key="test",
        langfuse_public_key=None,
        langfuse_secret_key=None,
    )


@pytest.fixture
def tiny_db(settings: Settings) -> Path:
    """A DuckDB file with a small `demo_sales` table for executor/verifier tests."""
    settings.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(settings.duckdb_path))
    conn.execute(
        """
        CREATE TABLE demo_sales AS
        SELECT * FROM (VALUES
            (1, 'ON', DATE '2024-01-05', 100.0),
            (2, 'ON', DATE '2024-02-10', 150.0),
            (3, 'QC', DATE '2024-01-20', 80.0),
            (4, 'QC', DATE '2024-02-25', 120.0),
            (5, 'BC', DATE '2024-03-01', 60.0)
        ) AS t(sale_id, province, sale_date, amount)
        """
    )
    conn.close()
    return settings.duckdb_path


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()
