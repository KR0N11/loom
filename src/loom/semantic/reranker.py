"""Second-stage rerankers for retrieval.

Approach: hybrid search returns a candidate list; a cross-encoder
(BAAI/bge-reranker-base) rescores (query, doc) pairs jointly, which is more
accurate than embedding similarity alone. `NoopReranker` keeps the fused order
for tests and cheap runs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from loom.config import Settings, get_settings
from loom.semantic.repository import ScoredDoc


class Reranker(ABC):
    name: str = "base"

    @abstractmethod
    def rerank(self, query: str, docs: list[ScoredDoc], top_k: int) -> list[ScoredDoc]: ...


class NoopReranker(Reranker):
    """Keep incoming order; just truncate."""

    name = "noop"

    def rerank(self, query: str, docs: list[ScoredDoc], top_k: int) -> list[ScoredDoc]:
        return docs[:top_k]


class CrossEncoderReranker(Reranker):
    """sentence-transformers CrossEncoder on bge-reranker-base, loaded on first use."""

    name = "cross-encoder"

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        self.model_name = model_name
        self._model: Any | None = None

    def _load(self) -> Any:
        # Lazy load: importing the module must stay free of model downloads.
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, docs: list[ScoredDoc], top_k: int) -> list[ScoredDoc]:
        # Nothing to score: return early rather than calling the model on an empty batch.
        if not docs:
            return []
        model = self._load()
        scores = model.predict([(query, d.doc.text) for d in docs])
        rescored = [ScoredDoc(d.doc, float(s)) for d, s in zip(docs, scores, strict=True)]
        rescored.sort(key=lambda d: d.score, reverse=True)
        return rescored[:top_k]


def get_reranker(settings: Settings | None = None, enabled: bool = True) -> Reranker:
    """Factory: cross-encoder when enabled and using HF embeddings, otherwise no-op."""
    settings = settings or get_settings()
    # Offline/hashing mode implies no model downloads, so no cross-encoder either.
    if not enabled or settings.embedding_backend.value == "hashing":
        return NoopReranker()
    return CrossEncoderReranker(settings.reranker_model)
