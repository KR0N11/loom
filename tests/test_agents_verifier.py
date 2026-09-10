"""Verifier agent: independent re-derivation plus deterministic checks; injection is caught."""

from __future__ import annotations

from loom.agents.injection import InjectionMode, inject
from loom.agents.schemas import AnalysisType, CheckStatus, KeyNumber
from loom.agents.verifier import VerifierAgent
from loom.llm.fake_provider import FakeProvider
from tests.agent_helpers import (
    DirectDuckDBExecutor,
    demo_execution,
    demo_plan,
    demo_semantic,
    demo_state,
    j,
)

ALT = j(
    {
        "alternate_sql": "SELECT sum(s) AS total_sales FROM (SELECT province, sum(amount) AS s FROM demo_sales GROUP BY 1)"
    }
)


def _run(settings, state):
    return VerifierAgent(FakeProvider([ALT]), DirectDuckDBExecutor(), settings).run(state)


# Matching numbers via a different aggregation path pass.
def test_reconcile_pass(settings, tiny_db):
    out = _run(
        settings,
        demo_state(plan=demo_plan(), semantic=demo_semantic(), executions=[demo_execution()]),
    )
    report = out["verification"]
    assert report.overall is CheckStatus.PASS
    rec = {c.name: c for c in report.checks}["reconcile:total_sales"]
    assert rec.status is CheckStatus.PASS and rec.observed == 510.0
    assert report.reconciled_numbers[0].value == 510.0
    assert len(report.queries) == 1 and "GROUP BY" in report.queries[0].code


# A scaled (x10) executor output is flagged as FAIL, and the corrupted executions are passed on.
def test_injection_scale_is_caught(settings, tiny_db):
    state = demo_state(
        plan=demo_plan(), semantic=demo_semantic(), executions=[demo_execution()], injection="scale"
    )
    out = _run(settings, state)
    report = out["verification"]
    assert report.overall is CheckStatus.FAIL
    assert out["executions"][0].key_numbers[0].value == 5100.0
    assert any("5100" in f for f in report.flags)


# Duplicate full rows in the result are a hard failure (a fanned-out join).
def test_duplicate_rows_fail(settings, tiny_db):
    ex = demo_execution(rows=[[510.0], [510.0]])
    out = _run(settings, demo_state(plan=demo_plan(), semantic=demo_semantic(), executions=[ex]))
    checks = {c.name: c.status for c in out["verification"].checks}
    assert checks["duplicate_check_1"] is CheckStatus.FAIL


# An average over few rows is a warning, not a failure.
def test_small_sample_warns(settings, tiny_db):
    ex = demo_execution()
    ex = ex.model_copy(
        update={"key_numbers": [KeyNumber(name="avg_sale", value=510.0, unit="usd")]}
    )
    alt = j({"alternate_sql": "SELECT 510.0 AS avg_sale"})
    out = VerifierAgent(FakeProvider([alt]), DirectDuckDBExecutor(), settings).run(
        demo_state(plan=demo_plan(), semantic=demo_semantic(), executions=[ex])
    )
    report = out["verification"]
    assert report.overall is CheckStatus.WARN
    assert {c.name: c.status for c in report.checks}["sample_size_1"] is CheckStatus.WARN


# An unknown unit is a warning; swap_unit injection therefore surfaces.
def test_swap_unit_warns(settings, tiny_db):
    state = demo_state(
        plan=demo_plan(),
        semantic=demo_semantic(),
        executions=[demo_execution()],
        injection="swap_unit",
    )
    out = _run(settings, state)
    statuses = {c.name: c.status for c in out["verification"].checks}
    assert statuses["unit_check:total_sales"] is CheckStatus.WARN


# A trend query without any date reference gets a time_range warning.
def test_trend_without_dates_warns(settings, tiny_db):
    state = demo_state(
        plan=demo_plan(AnalysisType.TREND), semantic=demo_semantic(), executions=[demo_execution()]
    )
    out = _run(settings, state)
    statuses = {c.name: c.status for c in out["verification"].checks}
    assert statuses["time_range_1"] is CheckStatus.WARN


# A failing alternate query cannot confirm anything: WARN, never PASS.
def test_alternate_query_failure_warns(settings, tiny_db):
    bad = j({"alternate_sql": "SELECT * FROM missing"})
    out = VerifierAgent(FakeProvider([bad]), DirectDuckDBExecutor(), settings).run(
        demo_state(plan=demo_plan(), semantic=demo_semantic(), executions=[demo_execution()])
    )
    assert out["verification"].overall is CheckStatus.WARN
    assert not out["verification"].queries[0].ok


# Injection never mutates the original executions and every mode changes something measurable.
def test_inject_modes_are_pure():
    ex = demo_execution(rows=[[510.0], ["2024-01-05"]])
    for mode in InjectionMode:
        out = inject([ex], mode)
        assert ex.key_numbers[0].value == 510.0 and ex.rows[0][0] == 510.0
        changed = out[0].key_numbers[0] != ex.key_numbers[0] or out[0].rows != ex.rows
        assert changed, mode
    assert inject([ex], "off_by_year")[0].rows[1][0] == "2025-01-05"
