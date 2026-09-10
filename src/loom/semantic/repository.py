"""Repository pattern over the vector store.

Approach: agents talk to `VectorRepository` only. `PgVectorRepository` is the
production backend; `DuckDBVectorRepository` is a zero-infra local fallback with
the same behaviour so tests and laptops never need Postgres.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from loom.semantic.models import SemanticDoc


class ScoredDoc:
    __slots__ = ("doc", "score")

    def __init__(self, doc: SemanticDoc, score: float) -> None:
        self.doc = doc
        self.score = score


class VectorRepository(ABC):
    """Store and search semantic documents by embedding and by keyword."""

    @abstractmethod
    def upsert(self, docs: list[SemanticDoc], embeddings: list[list[float]]) -> None: ...

    @abstractmethod
    def delete_dataset(self, dataset: str) -> None: ...

    @abstractmethod
    def search_vector(
        self, query_embedding: list[float], dataset: str, top_k: int
    ) -> list[ScoredDoc]: ...

    @abstractmethod
    def search_keyword(self, query: str, dataset: str, top_k: int) -> list[ScoredDoc]: ...

    @abstractmethod
    def list_docs(self, dataset: str) -> list[SemanticDoc]: ...

    @abstractmethod
    def count(self, dataset: str | None = None) -> int: ...
