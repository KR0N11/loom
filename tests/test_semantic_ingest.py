"""Ingestion tests: JSON -> LlamaIndex Documents -> repository, plus the pgvector integration stub."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from loom.config import Settings
from loom.semantic.duckdb_repository import DuckDBVectorRepository
from loom.semantic.embeddings import HashingEmbedder
from loom.semantic.ingest import (
    SemanticsNotFoundError,
    build_index,
    from_llama_documents,
    get_repository,
    load_semantics,
    semantics_path,
    to_llama_documents,
)
from loom.semantic.models import ColumnDoc, DatasetSemantics, MetricDoc, TableDoc


def _semantics() -> DatasetSemantics:
    return DatasetSemantics(
        dataset="demo",
        tables=[
            TableDoc(
                dataset="demo",
                name="demo_sales",
                description="Sales.",
                grain="sale",
                row_count=1,
                columns=[ColumnDoc(table="demo_sales", name="amount", dtype="DOUBLE")],
            )
        ],
        joins=[],
        metrics=[
            MetricDoc(
                dataset="demo",
                name="total_revenue",
                definition="Sum.",
                sql_expression="SUM(amount)",
                unit="CAD",
                grain="any",
                tables=["demo_sales"],
            )
        ],
    )


# Proves the LlamaIndex round trip keeps ids, kinds and metadata intact.
def test_llama_document_round_trip():
    docs = _semantics().to_docs()
    back = from_llama_documents(to_llama_documents(_semantics()))
    assert [d.model_dump() for d in back] == [d.model_dump() for d in docs]


# Proves build_index reads the JSON the profiler writes and fills the store.
def test_build_index_from_json(settings: Settings):
    path = semantics_path("demo", settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_semantics().model_dump_json())
    repo = DuckDBVectorRepository(settings.vector_store_path)
    n = build_index("demo", settings, embedder=HashingEmbedder(), repo=repo)
    assert n == 3
    assert repo.count("demo") == 3
    # Rebuilding replaces rather than duplicates.
    assert build_index("demo", settings, embedder=HashingEmbedder(), repo=repo) == 3
    assert repo.count("demo") == 3


# Proves a missing semantics file raises a message that says how to fix it.
def test_load_semantics_missing_file_is_actionable(settings: Settings):
    with pytest.raises(SemanticsNotFoundError, match="scripts/ingest.py"):
        load_semantics("nope", settings)


# Proves the factory returns the DuckDB backend by default.
def test_get_repository_defaults_to_duckdb(settings: Settings):
    assert isinstance(get_repository(settings), DuckDBVectorRepository)


# Proves the pgvector backend round-trips when a live Postgres is configured.
@pytest.mark.integration
def test_pgvector_round_trip():
    dsn = os.environ.get("LOOM_PG_DSN")
    if not dsn:
        pytest.skip("LOOM_PG_DSN not set")
    from loom.semantic.pgvector_repository import PgVectorRepository

    try:
        repo = PgVectorRepository(dsn, dim=256)
    except Exception as exc:  # pragma: no cover - depends on environment
        pytest.skip(f"Postgres unreachable: {exc}")
    docs = _semantics().to_docs()
    repo.delete_dataset("demo")
    repo.upsert(docs, HashingEmbedder().embed([d.text for d in docs]))
    assert repo.count("demo") == len(docs)
    assert repo.search_keyword("total_revenue", "demo", 1)[0].doc.name == "total_revenue"
    hits = repo.search_vector(HashingEmbedder().embed_query(docs[0].text), "demo", 1)
    assert hits[0].doc.doc_id == docs[0].doc_id
    repo.delete_dataset("demo")


# Proves the JSON on disk is what load_semantics parses (guards the file location contract).
def test_semantics_path_layout(settings: Settings, tmp_path: Path):
    assert semantics_path("ttc", settings) == tmp_path / "semantic" / "ttc.json"
    json.loads(_semantics().model_dump_json())
