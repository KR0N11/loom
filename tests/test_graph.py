"""Supervisor graph end to end with scripted LLM replies and a real LangGraph + SQLite checkpoint."""

from __future__ import annotations

from loom.agents.schemas import CheckStatus
from loom.agents.supervisor import run_question
from loom.llm.fake_provider import FakeProvider
from tests.agent_helpers import DirectDuckDBExecutor, demo_semantic, j

PLAN = j(
    {
        "analysis_type": "lookup",
        "steps": [{"step_id": 1, "description": "sum sales", "needs_tables": ["demo_sales"]}],
    }
)
EXEC = j({"language": "sql", "code": "SELECT sum(amount) AS total_sales FROM demo_sales"})
KEYS = j(
    {
        "key_numbers": [
            {"name": "total_sales", "value": 510.0, "unit": "usd", "derivation": "sum"}
        ],
        "summary": "510",
    }
)
ALT = j(
    {
        "alternate_sql": "SELECT sum(s) AS total_sales FROM (SELECT province, sum(amount) s FROM demo_sales GROUP BY 1)"
    }
)
NARR = j({"title": "t", "headline": "h", "findings": ["f"], "recommendation": "r"})


class StubSemanticAgent:
    def run(self, state):
        return {"semantic": demo_semantic()}


# Full path: plan -> semantic -> execute -> verify -> narrate, with a checkpoint written.
def test_graph_end_to_end(settings, tiny_db):
    provider = FakeProvider([PLAN, EXEC, KEYS, ALT, NARR])
    final = run_question(
        "total sales",
        "demo",
        settings=settings,
        provider=provider,
        thread_id="t1",
        semantic_agent=StubSemanticAgent(),
        sandbox=DirectDuckDBExecutor(),
    )
    assert final.plan is not None and final.semantic is not None
    assert final.executions[0].rows == [[510.0]]
    assert final.verification is not None and final.verification.overall is CheckStatus.PASS
    assert final.memo is not None and final.memo.headline == "h"
    assert final.usage.input_tokens == 50
    assert settings.checkpoint_path.exists()
    assert provider.replies == []


# Injection flows through the graph: verifier fails and the narrator marks the memo UNVERIFIED.
def test_graph_with_injection(settings, tiny_db):
    provider = FakeProvider([PLAN, EXEC, KEYS, ALT, NARR])
    final = run_question(
        "total sales",
        "demo",
        settings=settings,
        provider=provider,
        injection="scale",
        thread_id="t2",
        semantic_agent=StubSemanticAgent(),
        sandbox=DirectDuckDBExecutor(),
    )
    assert final.verification is not None and final.verification.overall is CheckStatus.FAIL
    assert final.memo is not None and final.memo.headline.startswith("UNVERIFIED:")


# When every attempt fails the graph skips the verifier and emits a failure memo.
def test_graph_failure_path(settings, tiny_db):
    settings.executor_max_retries = 1
    bad = j({"language": "sql", "code": "SELECT * FROM nope"})
    provider = FakeProvider([PLAN, bad, bad])
    final = run_question(
        "total sales",
        "demo",
        settings=settings,
        provider=provider,
        thread_id="t3",
        semantic_agent=StubSemanticAgent(),
        sandbox=DirectDuckDBExecutor(),
    )
    assert final.executions == [] and final.verification is None
    assert final.errors == ["step 1 failed after 1 retries"]
    assert final.memo is not None and "no query succeeded" in final.memo.headline
    assert provider.replies == []
