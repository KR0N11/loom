"""Postgres + pgvector repository (production backend).

Approach: one table `semantic_docs` with a `vector(768)` column for embedding
search (`<=>` cosine distance) and a generated `tsvector` column for keyword
search (`plainto_tsquery` + `ts_rank`). Same `VectorRepository` contract as
the DuckDB fallback, so agents never know which store is live.
"""

from __future__ import annotations

import json
from typing import Any

from loom.semantic.models import SemanticDoc
from loom.semantic.repository import ScoredDoc, VectorRepository

_DDL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS semantic_docs (
    doc_id    TEXT PRIMARY KEY,
    dataset   TEXT NOT NULL,
    kind      TEXT NOT NULL,
    name      TEXT NOT NULL,
    text      TEXT NOT NULL,
    metadata  JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding vector({dim}),
    tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', name || ' ' || text)) STORED
);
CREATE INDEX IF NOT EXISTS semantic_docs_dataset_idx ON semantic_docs (dataset);
CREATE INDEX IF NOT EXISTS semantic_docs_tsv_idx ON semantic_docs USING GIN (tsv);
"""

_COLUMNS = "doc_id, dataset, kind, name, text, metadata"


class PgVectorRepository(VectorRepository):
    def __init__(self, dsn: str, dim: int = 768) -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        self.dim = dim
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._conn.execute(_DDL.format(dim=dim))
        # Registers the vector type adapter so Python lists round-trip as vectors.
        register_vector(self._conn)

    def _row_to_doc(self, row: tuple[Any, ...]) -> SemanticDoc:
        meta = row[5] if isinstance(row[5], dict) else json.loads(row[5] or "{}")
        return SemanticDoc(
            doc_id=row[0], dataset=row[1], kind=row[2], name=row[3], text=row[4], metadata=meta
        )

    def upsert(self, docs: list[SemanticDoc], embeddings: list[list[float]]) -> None:
        # Mismatched lengths would silently pair the wrong vector with a doc.
        if len(docs) != len(embeddings):
            raise ValueError("docs and embeddings must have the same length")
        with self._conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO semantic_docs (doc_id, dataset, kind, name, text, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
                ON CONFLICT (doc_id) DO UPDATE SET
                    dataset = EXCLUDED.dataset, kind = EXCLUDED.kind, name = EXCLUDED.name,
                    text = EXCLUDED.text, metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """,
                [
                    (d.doc_id, d.dataset, d.kind, d.name, d.text, json.dumps(d.metadata), emb)
                    for d, emb in zip(docs, embeddings, strict=True)
                ],
            )

    def delete_dataset(self, dataset: str) -> None:
        self._conn.execute("DELETE FROM semantic_docs WHERE dataset = %s", (dataset,))

    def search_vector(
        self, query_embedding: list[float], dataset: str, top_k: int
    ) -> list[ScoredDoc]:
        rows = self._conn.execute(
            f"""
            SELECT {_COLUMNS}, 1 - (embedding <=> %s::vector) AS score
            FROM semantic_docs WHERE dataset = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, dataset, query_embedding, top_k),
        ).fetchall()
        return [ScoredDoc(self._row_to_doc(r), float(r[6])) for r in rows]

    def search_keyword(self, query: str, dataset: str, top_k: int) -> list[ScoredDoc]:
        rows = self._conn.execute(
            f"""
            SELECT {_COLUMNS}, ts_rank(tsv, plainto_tsquery('english', %s)) AS score
            FROM semantic_docs
            WHERE dataset = %s AND tsv @@ plainto_tsquery('english', %s)
            ORDER BY score DESC
            LIMIT %s
            """,
            (query, dataset, query, top_k),
        ).fetchall()
        return [ScoredDoc(self._row_to_doc(r), float(r[6])) for r in rows]

    def list_docs(self, dataset: str) -> list[SemanticDoc]:
        rows = self._conn.execute(
            f"SELECT {_COLUMNS} FROM semantic_docs WHERE dataset = %s ORDER BY doc_id", (dataset,)
        ).fetchall()
        return [self._row_to_doc(r) for r in rows]

    def count(self, dataset: str | None = None) -> int:
        if dataset is None:
            row = self._conn.execute("SELECT count(*) FROM semantic_docs").fetchone()
        else:
            row = self._conn.execute(
                "SELECT count(*) FROM semantic_docs WHERE dataset = %s", (dataset,)
            ).fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        self._conn.close()
