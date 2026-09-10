"""Text embedders for the semantic layer.

Approach: one small interface (`Embedder`) with two implementations. `HFEmbedder`
wraps BAAI/bge-base-en-v1.5 via sentence-transformers for real runs;
`HashingEmbedder` is a deterministic, dependency-free fallback so tests and
offline machines can exercise the whole retrieval path without a model download.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import Any

from loom.config import EmbeddingBackend, Settings, get_settings

# bge models recommend prefixing search queries (not passages) with this instruction.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(ABC):
    """Turn text into fixed-size unit vectors."""

    dim: int = 0
    name: str = "base"

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed passages (documents to be stored)."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query; subclasses may add a query-side instruction."""
        return self.embed([text])[0]


class HashingEmbedder(Embedder):
    """Offline fallback: hash each token into one of `dim` buckets, then L2-normalize."""

    name = "hashing"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        # md5 gives a stable bucket per token across processes (unlike Python's hash()).
        for tok in _TOKEN.findall(text.lower()):
            bucket = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        # An all-zero vector (empty text) cannot be normalized; leave it as zeros.
        if norm == 0:
            return vec
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


class HFEmbedder(Embedder):
    """sentence-transformers wrapper around BAAI/bge-base-en-v1.5 (768-d, normalized)."""

    name = "hf"

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5") -> None:
        self.model_name = model_name
        self._model: Any | None = None

    def _load(self) -> Any:
        # Lazy load so importing this module never triggers a model download.
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self.dim = int(self._model.get_sentence_embedding_dimension() or 0)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        arr = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        return [[float(x) for x in row] for row in arr]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([BGE_QUERY_INSTRUCTION + text])[0]


def get_embedder(settings: Settings | None = None) -> Embedder:
    """Factory keyed on settings.embedding_backend."""
    settings = settings or get_settings()
    if settings.embedding_backend == EmbeddingBackend.HASHING:
        return HashingEmbedder()
    return HFEmbedder(settings.embedding_model)
