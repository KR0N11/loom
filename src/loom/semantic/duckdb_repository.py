"""DuckDB-backed vector repository (local fallback for pgvector).

Approach: documents and their FLOAT[] embeddings live in one DuckDB file.
Vector search uses DuckDB's `list_cosine_similarity`; keyword search uses the
in-process BM25 index, rebuilt lazily per dataset after every write. Same
contract as `PgVectorRepository`, zero infrastructure.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from loom.semantic.bm25 import BM25Index
from loom.semantic.models import SemanticDoc
from loom.semantic.repository import ScoredDoc, VectorRepository

_COLUMNS = "doc_id, dataset, kind, name, text, metadata"


class DuckDBVectorRepository(VectorRepository):
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        # ":memory:" is allowed for tests; a file path gets its folder created.
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self.path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_docs (
                doc_id TEXT PRIMARY KEY,
                dataset TEXT,
                kind TEXT,
                name TEXT,
                text TEXT,
                metadata TEXT,
                embedding FLOAT[]
            )
            """
        )
        self._bm25: dict[str, BM25Index] = {}
        self._docs_cache: dict[str, dict[str, SemanticDoc]] = {}

    # ---------- writes ----------

    def upsert(self, docs: list[SemanticDoc], embeddings: list[list[float]]) -> None:
        # Mismatched lengths would silently pair the wrong vector with a doc.
        if len(docs) != len(embeddings):
            raise ValueError("docs and embeddings must have the same length")
        rows = [
            (d.doc_id, d.dataset, d.kind, d.name, d.text, json.dumps(d.metadata), emb)
            for d, emb in zip(docs, embeddings, strict=True)
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO semantic_docs VALUES (?, ?, ?, ?, ?, ?, ?)", rows
        )
        self._invalidate({d.dataset for d in docs})

    def delete_dataset(self, dataset: str) -> None:
        self._conn.execute("DELETE FROM semantic_docs WHERE dataset = ?", [dataset])
        self._invalidate({dataset})

    def _invalidate(self, datasets: set[str]) -> None:
        # Any write makes the cached BM25 index and doc map stale for that dataset.
        for ds in datasets:
            self._bm25.pop(ds, None)
            self._docs_cache.pop(ds, None)

    # ---------- reads ----------

    def _row_to_doc(self, row: tuple) -> SemanticDoc:
        return SemanticDoc(
            doc_id=row[0],
            dataset=row[1],
            kind=row[2],
            name=row[3],
            text=row[4],
            metadata=json.loads(row[5] or "{}"),
        )

    def list_docs(self, dataset: str) -> list[SemanticDoc]:
        rows = self._conn.execute(
            f"SELECT {_COLUMNS} FROM semantic_docs WHERE dataset = ? ORDER BY doc_id", [dataset]
        ).fetchall()
        return [self._row_to_doc(r) for r in rows]

    def count(self, dataset: str | None = None) -> int:
        if dataset is None:
            row = self._conn.execute("SELECT count(*) FROM semantic_docs").fetchone()
        else:
            row = self._conn.execute(
                "SELECT count(*) FROM semantic_docs WHERE dataset = ?", [dataset]
            ).fetchone()
        return int(row[0]) if row else 0

    def search_vector(
        self, query_embedding: list[float], dataset: str, top_k: int
    ) -> list[ScoredDoc]:
        rows = self._conn.execute(
            f"""
            SELECT {_COLUMNS}, list_cosine_similarity(embedding, ?::FLOAT[]) AS score
            FROM semantic_docs
            WHERE dataset = ?
            ORDER BY score DESC
            LIMIT ?
            """,
            [query_embedding, dataset, top_k],
        ).fetchall()
        return [ScoredDoc(self._row_to_doc(r), float(r[6] or 0.0)) for r in rows]

    def _docs_by_id(self, dataset: str) -> dict[str, SemanticDoc]:
        if dataset not in self._docs_cache:
            self._docs_cache[dataset] = {d.doc_id: d for d in self.list_docs(dataset)}
        return self._docs_cache[dataset]

    def search_keyword(self, query: str, dataset: str, top_k: int) -> list[ScoredDoc]:
        # Build the BM25 index on first use per dataset; writes invalidate it.
        if dataset not in self._bm25:
            docs = self._docs_by_id(dataset)
            self._bm25[dataset] = BM25Index(
                list(docs.keys()), [d.name + " " + d.text for d in docs.values()]
            )
        docs = self._docs_by_id(dataset)
        return [
            ScoredDoc(docs[doc_id], score)
            for doc_id, score in self._bm25[dataset].search(query, top_k)
        ]

    def close(self) -> None:
        self._conn.close()
