"""Runner and injection runner with a monkeypatched run_question."""

from __future__ import annotations

from pathlib import Path

import pytest

from loom.agents.schemas import (
    CheckStatus,
    ExecutionLanguage,
    ExecutionResult,
    KeyNumber,
    RetrievedDoc,
    SemanticContext,
    VerificationReport,
)
from loom.agents.state import LoomState
from loom.config import Settings
from loom.eval.benchmark import Benchmark, BenchmarkQuestion, Difficulty
from loom.eval.injection_runner import catch_rates, resolve_modes, run_injection_suite
from loom.eval.runner import EvalResults, run_benchmark, score_state
from loom.llm.base import LLMUsage


def _question(i: int = 0, dataset: str = "demo") -> BenchmarkQuestion:
    return BenchmarkQuestion(
        id=f"{dataset}-q{i}",
        dataset=dataset,
        question="total in ON?",
        difficulty=Difficulty.EASY,
        analysis_type="lookup",
        gold_sql="SELECT 250.0 AS total_amount",
        gold_answer={"total_amount": 250.0},
        gold_columns=["total_amount"],
        gold_rows=[[250.0]],
        expected_tables=["demo_sales"],
        expected_metrics=["total_amount"],
    )


def _state(value: float, overall: CheckStatus, injection: str | None = None) -> LoomState:
    return LoomState(
        question="total in ON?",
        dataset="demo",
        injection=injection,
        semantic=SemanticContext(
            tables=["demo_sales"],
            columns=[],
            metrics=[],
            joins=[],
            docs=[
                RetrievedDoc(doc_id="1", kind="table", name="demo_sales", text="", score=1.0),
                RetrievedDoc(
                    doc_id="2", kind="column", name="demo_sales.amount", text="", score=0.5
                ),
                RetrievedDoc(doc_id="3", kind="table", name="demo_other", text="", score=0.4),
            ],
            context_text="",
        ),
        executions=[
            ExecutionResult(
                step_id=1,
                language=ExecutionLanguage.SQL,
                code="SELECT 1",
                columns=["total_amount"],
                rows=[[value]],
                row_count=1,
                key_numbers=[KeyNumber(name="total_amount", value=value, unit="CAD")],
            )
        ],
        verification=VerificationReport(overall=overall, checks=[]),
        usage=LLMUsage(cost_usd=0.01, latency_s=1.5, model="fake"),
    )


# Proves a correct, verified answer scores as exec-accurate, numeric match, zero correction.
def test_score_state_correct() -> None:
    row = score_state(_question(), _state(250.0, CheckStatus.PASS), None)
    assert row.exec_ok and row.exec_acc and row.numeric_match and row.zero_correction
    assert row.retrieval_p == 0.5 and row.retrieval_r == 0.5
    assert row.verifier_overall == "pass" and row.caught is None
    assert row.cost_usd == 0.01


# Proves an injected run that the verifier flagged counts as caught, and is not zero-correction.
def test_score_state_injected_caught() -> None:
    row = score_state(_question(), _state(2500.0, CheckStatus.FAIL, "scale"), "scale")
    assert row.caught is True and not row.exec_acc and not row.zero_correction


# Proves a run with no executions records the errors.
def test_score_state_failed_run() -> None:
    state = LoomState(question="x", dataset="demo", errors=["executor gave up"])
    row = score_state(_question(), state, None)
    assert not row.exec_ok and row.error == "executor gave up"


# Proves the runner scores every job, persists incrementally, survives exceptions, and resumes.
def test_run_benchmark_and_resume(settings: Settings, tmp_path: Path) -> None:
    calls: list[tuple[str, str | None]] = []

    def fake_run(question: str, dataset: str, **kw: object) -> LoomState:
        calls.append((str(kw["thread_id"]), kw["injection"]))  # type: ignore[arg-type]
        if kw["thread_id"] == "eval-demo-q2-clean":
            raise RuntimeError("boom")
        injected = kw["injection"] is not None
        return _state(
            2500.0 if injected else 250.0, CheckStatus.FAIL if injected else CheckStatus.PASS
        )

    bench = Benchmark(questions=[_question(i) for i in range(3)])
    out = tmp_path / "run"
    results = run_benchmark(
        bench,
        settings=settings,
        provider=None,
        injection_modes=("scale",),
        out_dir=out,
        run_question=fake_run,
    )
    assert len(results.rows) == 6
    assert (out / "results.json").exists()
    errored = [r for r in results.rows if r.error]
    assert len(errored) == 1 and errored[0].error.startswith("RuntimeError")
    assert results.execution_accuracy() == pytest.approx(2 / 3)
    assert results.catch_rate("scale") == 1.0
    assert results.false_alarm_rate() == 0.0
    # Resume skips everything already scored.
    calls.clear()
    resumed = run_benchmark(
        bench,
        settings=settings,
        provider=None,
        injection_modes=("scale",),
        out_dir=out,
        resume=True,
        run_question=fake_run,
    )
    assert calls == [] and len(resumed.rows) == 6
    assert EvalResults.load(out / "results.json").summary()["n"] == 3


# Proves the injection suite reports per-mode catch rates and false alarms.
def test_injection_suite(settings: Settings, tmp_path: Path) -> None:
    def fake_run(question: str, dataset: str, **kw: object) -> LoomState:
        mode = kw["injection"]
        if mode is None:
            return _state(250.0, CheckStatus.PASS)
        # The verifier catches scale but misses swap_unit in this fake.
        return _state(2500.0, CheckStatus.FAIL if mode == "scale" else CheckStatus.PASS, str(mode))

    bench = Benchmark(questions=[_question(i) for i in range(2)])
    results, report = run_injection_suite(
        bench,
        settings,
        None,
        sample_n=None,
        modes=("scale", "swap_unit"),
        out_dir=tmp_path / "inj",
        run_question=fake_run,
    )
    assert report.per_mode == {"scale": 1.0, "swap_unit": 0.0}
    assert report.overall == 0.5 and report.false_alarm_rate == 0.0 and report.n_injected == 4
    assert catch_rates(results) == report


# Proves mode parsing accepts 'all' and lists, and rejects typos.
def test_resolve_modes() -> None:
    assert len(resolve_modes("all")) == 6
    assert resolve_modes("scale, zero") == ("scale", "zero")
    assert resolve_modes(None) == ()
    with pytest.raises(ValueError):
        resolve_modes("scalee")
