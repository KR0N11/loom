"""Semantic-layer ingestion: DatasetSemantics JSON -> LlamaIndex Documents -> vector store.

Approach: the profiler (data package) writes `data/semantic/<dataset>.json`.
We flatten it into LlamaIndex `Document`s (one per table/column/metric/join,
metadata carried along), embed the text, and replace the dataset's rows in the
repository atomically enough for our needs (delete then upsert).
"""

from __future__ import annotations

from pathlib import Path

from llama_index.core import Document

from loom.config import Settings, VectorBackend, get_settings
from loom.semantic.embeddings import Embedder, get_embedder
from loom.semantic.models import DatasetSemantics, SemanticDoc
from loom.semantic.repository import VectorRepository


class SemanticsNotFoundError(FileNotFoundError):
    """Raised when the dataset has not been ingested/profiled yet."""


def semantics_path(dataset: str, settings: Settings) -> Path:
    return Path(settings.semantic_dir) / f"{dataset}.json"


def load_semantics(dataset: str, settings: Settings | None = None) -> DatasetSemantics:
    settings = settings or get_settings()
    path = semantics_path(dataset, settings)
    # A missing file means `scripts/ingest.py` was never run; say so instead of a bare FileNotFoundError.
    if not path.exists():
        raise SemanticsNotFoundError(
            f"no semantic metadata at {path}; run `python scripts/ingest.py --dataset {dataset}`"
        )
    return DatasetSemantics.model_validate_json(path.read_text())


def get_repository(settings: Settings | None = None, dim: int = 768) -> VectorRepository:
    """Factory: pgvector in production, DuckDB file locally."""
    settings = settings or get_settings()
    if settings.vector_backend == VectorBackend.PGVECTOR:
        from loom.semantic.pgvector_repository import PgVectorRepository

        return PgVectorRepository(settings.pg_dsn, dim=dim)
    from loom.semantic.duckdb_repository import DuckDBVectorRepository

    return DuckDBVectorRepository(settings.vector_store_path)


def to_llama_documents(semantics: DatasetSemantics) -> list[Document]:
    """LlamaIndex Documents with the flattened text and our metadata attached."""
    return [
        Document(
            doc_id=d.doc_id,
            text=d.text,
            metadata={"dataset": d.dataset, "kind": d.kind, "name": d.name, **d.metadata},
        )
        for d in semantics.to_docs()
    ]


def from_llama_documents(documents: list[Document]) -> list[SemanticDoc]:
    docs: list[SemanticDoc] = []
    for doc in documents:
        meta = dict(doc.metadata)
        docs.append(
            SemanticDoc(
                doc_id=doc.doc_id,
                dataset=str(meta.pop("dataset")),
                kind=str(meta.pop("kind")),
                name=str(meta.pop("name")),
                text=doc.text,
                metadata={k: str(v) for k, v in meta.items()},
            )
        )
    return docs


def build_index(
    dataset: str,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    repo: VectorRepository | None = None,
) -> int:
    """Embed and store every semantic document for `dataset`. Returns the count stored."""
    settings = settings or get_settings()
    semantics = load_semantics(dataset, settings)
    embedder = embedder or get_embedder(settings)
    docs = from_llama_documents(to_llama_documents(semantics))
    embeddings = embedder.embed([d.text for d in docs])
    repo = repo or get_repository(settings, dim=len(embeddings[0]) if embeddings else 768)
    # Replace, don't append: re-ingesting must not leave stale columns behind.
    repo.delete_dataset(dataset)
    repo.upsert(docs, embeddings)
    return len(docs)
