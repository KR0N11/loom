"""Executor agent: runs each step, retries on sandbox errors, records every attempt."""

from __future__ import annotations

from loom.agents.executor import ExecutorAgent
from loom.llm.fake_provider import FakeProvider
from tests.agent_helpers import DirectDuckDBExecutor, demo_plan, demo_semantic, demo_state, j

GOOD_SQL = "SELECT sum(amount) AS total_sales FROM demo_sales"
BAD_SQL = "SELECT sum(amount) FROM no_such_table"
KEYS = j(
    {
        "key_numbers": [
            {"name": "total_sales", "value": 510.0, "unit": "usd", "derivation": "sum(amount)"}
        ],
        "summary": "Total is 510.",
    }
)


def _agent(settings, replies):
    return ExecutorAgent(FakeProvider(replies), DirectDuckDBExecutor(), settings)


# Happy path: one step, one query, key numbers extracted from the real rows.
def test_executor_happy_path(settings, tiny_db):
    agent = _agent(settings, [j({"language": "sql", "code": GOOD_SQL}), KEYS])
    out = agent.run(demo_state(plan=demo_plan(), semantic=demo_semantic()))
    ex = out["executions"][0]
    assert ex.rows == [[510.0]] and ex.columns == ["total_sales"]
    assert ex.key_numbers[0].value == 510.0
    assert len(ex.queries) == 1 and ex.queries[0].ok
    assert out["errors"] == []


# A failing query is repaired on retry; both attempts are kept in the audit trail.
def test_executor_retries_then_succeeds(settings, tiny_db):
    agent = _agent(
        settings,
        [j({"language": "sql", "code": BAD_SQL}), j({"language": "sql", "code": GOOD_SQL}), KEYS],
    )
    out = agent.run(demo_state(plan=demo_plan(), semantic=demo_semantic()))
    ex = out["executions"][0]
    assert [q.ok for q in ex.queries] == [False, True]
    assert "no_such_table" in (ex.queries[0].error or "")
    assert ex.queries[1].attempt == 2


# After max retries the step is abandoned, an error is recorded, and later steps are skipped.
def test_executor_gives_up(settings, tiny_db):
    settings.executor_max_retries = 2
    replies = [j({"language": "sql", "code": BAD_SQL})] * 3
    agent = _agent(settings, replies)
    out = agent.run(demo_state(plan=demo_plan(steps=2), semantic=demo_semantic()))
    assert out["executions"] == []
    assert out["errors"] == ["step 1 failed after 2 retries"]
    # 3 attempts = first + 2 retries; no key-number call was made.
    assert len(agent.provider.calls) == 3


# Step 2 sees step 1's summary so multi-step plans chain.
def test_executor_passes_prior_summary(settings, tiny_db):
    replies = [
        j({"language": "sql", "code": GOOD_SQL}),
        KEYS,
        j({"language": "sql", "code": GOOD_SQL}),
        KEYS,
    ]
    agent = _agent(settings, replies)
    out = agent.run(demo_state(plan=demo_plan(steps=2), semantic=demo_semantic()))
    assert len(out["executions"]) == 2
    _, msgs = agent.provider.calls[2]
    assert "Step 1 result: Total is 510." in msgs[0].content
