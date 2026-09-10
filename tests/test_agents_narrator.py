"""Narrator agent: enforces UNVERIFIED headline, builds the audit appendix, renders charts."""

from __future__ import annotations

from pathlib import Path

from loom.agents.narrator import NarratorAgent, memo_to_markdown
from loom.agents.schemas import (
    CheckStatus,
    ExecutionLanguage,
    QueryRecord,
    VerificationCheck,
    VerificationReport,
)
from loom.llm.fake_provider import FakeProvider
from tests.agent_helpers import demo_execution, demo_plan, demo_state, j

NARR = j(
    {
        "title": "Sales",
        "headline": "Sales total 510.",
        "findings": ["f1"],
        "recommendation": "do x",
        "caveats": [],
        "charts": [{"title": "by province", "kind": "bar", "x": "province", "y": "amount"}],
    }
)


def _report(overall: CheckStatus, flags: list[str]) -> VerificationReport:
    return VerificationReport(
        overall=overall,
        checks=[VerificationCheck(name="c", status=overall, detail="d")],
        flags=flags,
        queries=[
            QueryRecord(
                attempt=1,
                language=ExecutionLanguage.SQL,
                code="SELECT 1 AS alt",
                ok=True,
                row_count=1,
            )
        ],
    )


# A failed verification forces the UNVERIFIED prefix and copies flags into caveats.
def test_unverified_prefix_enforced(settings):
    state = demo_state(
        plan=demo_plan(),
        executions=[demo_execution()],
        verification=_report(CheckStatus.FAIL, ["reconcile: 510 vs 5100"]),
    )
    memo = NarratorAgent(FakeProvider([NARR]), settings).run(state)["memo"]
    assert memo.headline.startswith("UNVERIFIED:")
    assert "reconcile: 510 vs 5100" in memo.caveats
    assert memo.verification_summary.startswith("FAIL")


# The appendix lists executor and verifier queries, and markdown renders each as a code block.
def test_appendix_lists_all_queries(settings):
    state = demo_state(
        plan=demo_plan(), executions=[demo_execution()], verification=_report(CheckStatus.PASS, [])
    )
    memo = NarratorAgent(FakeProvider([NARR]), settings).run(state)["memo"]
    assert [q.code for q in memo.appendix_queries] == [
        "SELECT sum(amount) AS total_sales FROM demo_sales",
        "SELECT 1 AS alt",
    ]
    md = memo_to_markdown(memo)
    assert "## Audit appendix" in md and md.count("```sql") == 2
    assert not memo.headline.startswith("UNVERIFIED")


# A chart whose columns exist is rendered to disk; one whose columns do not is dropped.
def test_chart_rendered_when_columns_exist(settings):
    ex = demo_execution(rows=[["ON", 250.0], ["QC", 200.0]]).model_copy(
        update={"columns": ["province", "amount"]}
    )
    state = demo_state(
        plan=demo_plan(), executions=[ex], verification=_report(CheckStatus.PASS, [])
    )
    memo = NarratorAgent(FakeProvider([NARR]), settings).run(state)["memo"]
    assert len(memo.charts) == 1 and Path(memo.charts[0].path).exists()
    assert Path(memo.charts[0].path).with_suffix(".html").exists()
    state2 = demo_state(
        plan=demo_plan(), executions=[demo_execution()], verification=_report(CheckStatus.PASS, [])
    )
    memo2 = NarratorAgent(FakeProvider([NARR]), settings).run(state2)["memo"]
    assert memo2.charts == []


# With no executions the narrator writes a failure memo without calling the LLM.
def test_failure_memo_without_llm(settings):
    provider = FakeProvider([])
    memo = NarratorAgent(provider, settings).run(demo_state(errors=["step 1 failed"]))["memo"]
    assert memo.headline.startswith("UNVERIFIED:") and memo.caveats == ["step 1 failed"]
    assert provider.calls == []
