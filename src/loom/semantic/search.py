"""Hybrid retrieval and context assembly for the executor.

Approach: run vector search and BM25 keyword search side by side, fuse them
with Reciprocal Rank Fusion (RRF, k=60), then rerank the fused candidates with
a cross-encoder. `build_context` then expands whatever was hit into a complete
picture: every column of every touched table plus every join and metric that
involves those tables, so the executor never has to guess a join key.
"""

from __future__ import annotations

from loom.agents.schemas import RetrievedDoc, SemanticContext
from loom.semantic.embeddings import Embedder
from loom.semantic.models import DatasetSemantics, TableDoc
from loom.semantic.repository import ScoredDoc, VectorRepository
from loom.semantic.reranker import NoopReranker, Reranker

RRF_K = 60


def reciprocal_rank_fusion(ranked_lists: list[list[ScoredDoc]], k: int = RRF_K) -> list[ScoredDoc]:
    """Merge ranked lists: each doc scores sum(1 / (k + rank)); ties broken by doc_id."""
    scores: dict[str, float] = {}
    docs: dict[str, ScoredDoc] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item.doc.doc_id] = scores.get(item.doc.doc_id, 0.0) + 1.0 / (k + rank)
            docs.setdefault(item.doc.doc_id, item)
    fused = [ScoredDoc(docs[i].doc, s) for i, s in scores.items()]
    fused.sort(key=lambda d: (-d.score, d.doc.doc_id))
    return fused


class HybridSearch:
    """Vector + keyword retrieval with RRF and optional reranking."""

    def __init__(
        self, repo: VectorRepository, embedder: Embedder, reranker: Reranker | None = None
    ) -> None:
        self.repo = repo
        self.embedder = embedder
        self.reranker = reranker or NoopReranker()

    def retrieve(self, query: str, dataset: str, top_k: int = 8) -> list[ScoredDoc]:
        candidates = max(top_k * 2, 1)
        vector_hits = self.repo.search_vector(self.embedder.embed_query(query), dataset, candidates)
        keyword_hits = self.repo.search_keyword(query, dataset, candidates)
        fused = reciprocal_rank_fusion([vector_hits, keyword_hits])
        return self.reranker.rerank(query, fused, top_k)


def _table_of(doc: ScoredDoc) -> str | None:
    meta = doc.doc.metadata
    if doc.doc.kind in ("table", "column"):
        return meta.get("table")
    return None


def build_context(docs: list[ScoredDoc], semantics: DatasetSemantics | None) -> SemanticContext:
    """Expand retrieved docs into a complete, prompt-ready description."""
    retrieved = [
        RetrievedDoc(
            doc_id=d.doc.doc_id, kind=d.doc.kind, name=d.doc.name, text=d.doc.text, score=d.score
        )
        for d in docs
    ]
    tables: list[str] = []
    for d in docs:
        t = _table_of(d)
        # Joins and metrics also name tables; pull those in so the expansion is complete.
        if d.doc.kind == "join":
            for key in ("left", "right"):
                if d.doc.metadata.get(key):
                    tables.append(d.doc.metadata[key])
        elif d.doc.kind == "metric" and d.doc.metadata.get("tables"):
            tables.extend(d.doc.metadata["tables"].split(","))
        elif t:
            tables.append(t)
    tables = list(dict.fromkeys(t for t in tables if t))

    # Without the curated semantics we can only echo back what was retrieved.
    if semantics is None:
        columns = [d.doc.name for d in docs if d.doc.kind == "column"]
        metrics = [d.doc.name for d in docs if d.doc.kind == "metric"]
        joins = [d.doc.name for d in docs if d.doc.kind == "join"]
        text = "\n".join(f"- {d.doc.text}" for d in docs)
        return SemanticContext(
            tables=tables,
            columns=columns,
            metrics=metrics,
            joins=joins,
            docs=retrieved,
            context_text=text,
        )

    table_docs: dict[str, TableDoc] = {t.name: t for t in semantics.tables}
    touched = [t for t in tables if t in table_docs]
    # If retrieval hit nothing usable, fall back to every table so the executor still has a schema.
    if not touched:
        touched = list(table_docs)
    touched_set = set(touched)
    join_docs = [
        j for j in semantics.joins if j.left_table in touched_set or j.right_table in touched_set
    ]
    metric_docs = [m for m in semantics.metrics if touched_set.intersection(m.tables)]
    all_columns = [f"{t}.{c.name}" for t in touched for c in table_docs[t].columns]

    lines: list[str] = []
    for t in touched:
        td = table_docs[t]
        span = (
            f" Time: {td.time_column} from {td.time_min} to {td.time_max}."
            if td.time_column
            else ""
        )
        lines.append(
            f"TABLE {td.name} ({td.row_count} rows) — {td.description} Grain: {td.grain}.{span}"
        )
        for c in td.columns:
            unit = f", unit={c.unit}" if c.unit else ""
            ex = f", e.g. {', '.join(c.sample_values[:3])}" if c.sample_values else ""
            nulls = f", {c.null_fraction:.0%} null" if c.null_fraction >= 0.05 else ""
            lines.append(f"  - {c.name} {c.dtype}{unit}{nulls}: {c.description}{ex}")
    if join_docs:
        lines.append("ALLOWED JOINS:")
        for j in join_docs:
            lines.append(
                f"  - {j.left_table} {j.kind.upper()} JOIN {j.right_table} ON {j.on}. {j.note}".rstrip()
            )
    if metric_docs:
        lines.append("METRIC DEFINITIONS (use these exact expressions):")
        for m in metric_docs:
            caveat = f" Caveat: {m.caveats}" if m.caveats else ""
            lines.append(
                f"  - {m.name} [{m.unit}, grain={m.grain}]: {m.definition} SQL: {m.sql_expression}.{caveat}"
            )
    return SemanticContext(
        tables=touched,
        columns=all_columns,
        metrics=[m.name for m in metric_docs],
        joins=[f"{j.left_table}->{j.right_table} ON {j.on}" for j in join_docs],
        docs=retrieved,
        context_text="\n".join(lines),
    )
