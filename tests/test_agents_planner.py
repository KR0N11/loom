"""Planner agent: classifies, decomposes, records usage, and uses the strategy guidance."""

from __future__ import annotations

from loom.agents.planner import PlannerAgent, table_summary
from loom.agents.schemas import AnalysisType
from loom.llm.fake_provider import FakeProvider
from loom.semantic.models import ColumnDoc, DatasetSemantics, MetricDoc, TableDoc
from tests.agent_helpers import demo_state, j


def _semantics() -> DatasetSemantics:
    return DatasetSemantics(
        dataset="demo",
        tables=[
            TableDoc(
                dataset="demo",
                name="demo_sales",
                description="sales",
                grain="one row per sale",
                row_count=5,
                columns=[ColumnDoc(table="demo_sales", name="amount", dtype="DOUBLE")],
            )
        ],
        joins=[],
        metrics=[
            MetricDoc(
                dataset="demo",
                name="total_sales",
                definition="d",
                sql_expression="sum(amount)",
                unit="usd",
                grain="all",
                tables=["demo_sales"],
            )
        ],
    )


# The planner turns the LLM reply into a typed AnalysisPlan and adds the call's usage.
def test_planner_builds_plan(settings):
    reply = j(
        {
            "analysis_type": "trend",
            "steps": [{"step_id": 1, "description": "monthly sales"}],
            "metrics_needed": ["total_sales"],
            "time_range_hint": "2024",
        }
    )
    provider = FakeProvider([reply])
    agent = PlannerAgent(provider, settings=settings, semantics_loader=lambda d, s: _semantics())
    out = agent.run(demo_state())
    plan = out["plan"]
    assert plan.analysis_type is AnalysisType.TREND
    assert plan.steps[0].description == "monthly sales"
    assert plan.dataset == "demo" and plan.question == "total sales"
    assert out["usage"].input_tokens == 10
    system, _ = provider.calls[0]
    assert "demo_sales" in system and "total_sales" in system and "Trend:" in system


# A malformed first reply is corrected on the second attempt.
def test_planner_retries_bad_json(settings):
    good = j({"analysis_type": "lookup", "steps": [{"step_id": 1, "description": "x"}]})
    provider = FakeProvider(["not json", good])
    agent = PlannerAgent(provider, settings=settings, semantics_loader=lambda d, s: None)
    out = agent.run(demo_state())
    assert out["plan"].analysis_type is AnalysisType.LOOKUP
    assert len(provider.calls) == 2


# Without metadata the planner still runs with a placeholder table list.
def test_table_summary_none():
    assert "no table metadata" in table_summary(None)
