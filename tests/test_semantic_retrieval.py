"""Semantic layer tests: repository round trip, BM25, RRF, context expansion, agent."""

from __future__ import annotations

from pathlib import Path

import pytest

from loom.agents.schemas import AnalysisPlan, AnalysisType, PlanStep, SemanticContext
from loom.agents.semantic_agent import SemanticAgent
from loom.agents.state import LoomState
from loom.semantic.bm25 import BM25Index
from loom.semantic.duckdb_repository import DuckDBVectorRepository
from loom.semantic.embeddings import HashingEmbedder
from loom.semantic.models import (
    ColumnDoc,
    DatasetSemantics,
    JoinDoc,
    MetricDoc,
    SemanticDoc,
    TableDoc,
)
from loom.semantic.repository import ScoredDoc
from loom.semantic.reranker import NoopReranker
from loom.semantic.search import HybridSearch, build_context, reciprocal_rank_fusion


@pytest.fixture
def semantics() -> DatasetSemantics:
    """Two tables, one join, one metric — enough to exercise expansion."""
    return DatasetSemantics(
        dataset="demo",
        tables=[
            TableDoc(
                dataset="demo",
                name="demo_sales",
                description="One row per sale.",
                grain="sale",
                row_count=5,
                time_column="sale_date",
                time_min="2024-01-05",
                time_max="2024-03-01",
                columns=[
                    ColumnDoc(table="demo_sales", name="sale_id", dtype="INTEGER"),
                    ColumnDoc(
                        table="demo_sales",
                        name="province",
                        dtype="VARCHAR",
                        description="Two-letter province code",
                        sample_values=["ON", "QC"],
                    ),
                    ColumnDoc(table="demo_sales", name="sale_date", dtype="DATE"),
                    ColumnDoc(
                        table="demo_sales",
                        name="amount",
                        dtype="DOUBLE",
                        description="Sale value",
                        unit="CAD",
                    ),
                ],
            ),
            TableDoc(
                dataset="demo",
                name="demo_provinces",
                description="Province lookup.",
                grain="province",
                row_count=3,
                columns=[
                    ColumnDoc(table="demo_provinces", name="code", dtype="VARCHAR"),
                    ColumnDoc(table="demo_provinces", name="region", dtype="VARCHAR"),
                ],
            ),
        ],
        joins=[
            JoinDoc(
                dataset="demo",
                left_table="demo_sales",
                right_table="demo_provinces",
                on="demo_sales.province = demo_provinces.code",
            )
        ],
        metrics=[
            MetricDoc(
                dataset="demo",
                name="total_revenue",
                definition="Sum of sale amount.",
                sql_expression="SUM(amount)",
                unit="CAD",
                grain="any",
                tables=["demo_sales"],
                caveats="Excludes refunds.",
            )
        ],
    )


@pytest.fixture
def repo(tmp_path: Path, semantics: DatasetSemantics) -> DuckDBVectorRepository:
    repo = DuckDBVectorRepository(tmp_path / "semantic.duckdb")
    docs = semantics.to_docs()
    repo.upsert(docs, HashingEmbedder().embed([d.text for d in docs]))
    return repo


# Proves docs written to the DuckDB store come back with the same ids and metadata.
def test_upsert_and_list_round_trip(repo: DuckDBVectorRepository, semantics: DatasetSemantics):
    docs = semantics.to_docs()
    stored = repo.list_docs("demo")
    assert {d.doc_id for d in stored} == {d.doc_id for d in docs}
    assert repo.count("demo") == len(docs)
    col = next(d for d in stored if d.doc_id == "demo.column.demo_sales.amount")
    assert col.metadata == {"table": "demo_sales", "column": "amount"}


# Proves upserting the same doc_id twice replaces rather than duplicates.
def test_upsert_is_idempotent(repo: DuckDBVectorRepository, semantics: DatasetSemantics):
    before = repo.count("demo")
    docs = semantics.to_docs()
    repo.upsert(docs, HashingEmbedder().embed([d.text for d in docs]))
    assert repo.count("demo") == before


# Proves vector search ranks the doc whose text matches the query first.
def test_vector_search_ranks_matching_doc_first(repo: DuckDBVectorRepository):
    emb = HashingEmbedder()
    hits = repo.search_vector(
        emb.embed_query("Metric total_revenue: Sum of sale amount."), "demo", 3
    )
    assert hits[0].doc.doc_id == "demo.metric.total_revenue"
    assert hits[0].score >= hits[-1].score


# Proves the keyword path finds an exact column name (embeddings can miss rare tokens).
def test_keyword_search_finds_exact_column_name(repo: DuckDBVectorRepository):
    hits = repo.search_keyword("sale_id", "demo", 3)
    assert hits[0].doc.doc_id == "demo.column.demo_sales.sale_id"


# Proves BM25 scores a document containing the query term above one that does not.
def test_bm25_prefers_documents_with_the_term():
    idx = BM25Index(["a", "b", "c"], ["flood claims paid", "subway delay minutes", "bus route"])
    ranked = idx.search("subway delay", 3)
    assert [r[0] for r in ranked] == ["b"]


# Proves RRF puts a doc that appears in both lists above docs seen once.
def test_rrf_rewards_docs_in_both_lists():
    def sd(i: str, s: float) -> ScoredDoc:
        return ScoredDoc(SemanticDoc(doc_id=i, dataset="d", kind="column", name=i, text=i), s)

    fused = reciprocal_rank_fusion([[sd("x", 0.9), sd("y", 0.8)], [sd("y", 5.0), sd("z", 4.0)]])
    assert fused[0].doc.doc_id == "y"
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)


# Proves deleting one dataset leaves another dataset's docs untouched.
def test_delete_dataset_isolates_datasets(repo: DuckDBVectorRepository):
    other = SemanticDoc(doc_id="o.table.t", dataset="other", kind="table", name="t", text="table t")
    repo.upsert([other], HashingEmbedder().embed([other.text]))
    repo.delete_dataset("demo")
    assert repo.count("demo") == 0
    assert repo.count("other") == 1
    assert repo.search_keyword("table", "demo", 5) == []


# Proves a hit on one column expands to every column of that table plus its join and metric.
def test_build_context_expands_touched_tables(
    repo: DuckDBVectorRepository, semantics: DatasetSemantics
):
    search = HybridSearch(repo, HashingEmbedder(), NoopReranker())
    docs = search.retrieve("amount by province", "demo", top_k=2)
    ctx = build_context(docs, semantics)
    assert "demo_sales" in ctx.tables
    assert {
        "demo_sales.sale_id",
        "demo_sales.province",
        "demo_sales.sale_date",
        "demo_sales.amount",
    } <= set(ctx.columns)
    assert ctx.joins == ["demo_sales->demo_provinces ON demo_sales.province = demo_provinces.code"]
    assert ctx.metrics == ["total_revenue"]
    assert "SUM(amount)" in ctx.context_text
    assert "Excludes refunds." in ctx.context_text
    assert "unit=CAD" in ctx.context_text


# Proves that with no retrieval hits we still hand the executor every table's schema.
def test_build_context_falls_back_to_all_tables(semantics: DatasetSemantics):
    ctx = build_context([], semantics)
    assert ctx.tables == ["demo_sales", "demo_provinces"]


# Proves the agent returns a SemanticContext keyed for the graph state and uses plan text.
def test_semantic_agent_returns_context(repo: DuckDBVectorRepository, semantics: DatasetSemantics):
    search = HybridSearch(repo, HashingEmbedder(), NoopReranker())
    agent = SemanticAgent(search, lambda ds: semantics, top_k=3)
    plan = AnalysisPlan(
        question="q",
        analysis_type=AnalysisType.LOOKUP,
        dataset="demo",
        steps=[PlanStep(step_id=1, description="sum revenue per province")],
        metrics_needed=["total_revenue"],
    )
    state = LoomState(question="How much revenue per province?", dataset="demo", plan=plan)
    assert "total_revenue" in agent.build_query(state)
    out = agent.run(state)
    assert isinstance(out["semantic"], SemanticContext)
    assert out["semantic"].docs
    assert "demo_sales" in out["semantic"].tables


# Proves the hashing embedder is deterministic and unit length, which cosine search relies on.
def test_hashing_embedder_is_deterministic_and_normalized():
    emb = HashingEmbedder(dim=64)
    a, b = emb.embed(["subway delay", "subway delay"])
    assert a == b
    assert sum(x * x for x in a) == pytest.approx(1.0)
    assert emb.embed([""])[0] == [0.0] * 64
