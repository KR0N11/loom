"""Report rendering and CI gate rules."""

from __future__ import annotations

from pathlib import Path

from loom.eval.gate import evaluate_gate, run_gate
from loom.eval.report import write_report
from loom.eval.runner import BaselineRow, EvalResults, EvalRow


def _results(acc: list[bool], catches: list[bool] | None = None) -> EvalResults:
    rows = [
        EvalRow(
            id=f"q{i}",
            dataset="ttc" if i % 2 else "nfip",
            difficulty=["easy", "medium", "hard"][i % 3],
            analysis_type="lookup",
            exec_ok=True,
            exec_acc=ok,
            numeric_match=ok,
            verifier_overall="pass",
            zero_correction=ok,
            cost_usd=0.02,
            latency_s=float(i + 1),
            retrieval_p=0.5,
            retrieval_r=1.0,
            planner_type="lookup",
        )
        for i, ok in enumerate(acc)
    ]
    for i, caught in enumerate(catches or []):
        rows.append(
            EvalRow(
                id=f"q{i}",
                dataset="ttc",
                difficulty="easy",
                analysis_type="lookup",
                injection="scale",
                verifier_overall="fail" if caught else "pass",
                caught=caught,
            )
        )
    return EvalResults(started_at="2026-01-01T00:00:00", model="fake", rows=rows)


# Proves the report contains every section and the headline numbers.
def test_report_renders(tmp_path: Path) -> None:
    results = _results([True, True, False, True], [True, False])
    results.baseline_rows = [
        BaselineRow(baseline="qwen-coder", id="q0", difficulty="easy", dataset="ttc", exec_ok=True)
    ]
    results.routing = {"bart zero-shot": 0.75}
    text = write_report(results, tmp_path / "r.md", {"reviewed": 0, "unreviewed": 4})
    assert (tmp_path / "r.md").exists()
    for section in ("## Overall", "## By difficulty", "## By dataset", "## By analysis type"):
        assert section in text
    assert "| Execution accuracy | 75.0% |" in text
    assert "| scale | 50.0% |" in text
    assert "Small-model vs LLM comparison" in text and "qwen-coder" in text
    assert "Zero-shot router baseline" in text
    assert "Planner routing accuracy vs gold analysis type: 100.0%" in text


# Proves an accuracy drop of 6 points fails and 4 points passes.
def test_gate_accuracy_rule() -> None:
    baseline = _results([True] * 100, [True] * 5)
    candidate_bad = _results([True] * 94 + [False] * 6, [True] * 5)
    candidate_ok = _results([True] * 96 + [False] * 4, [True] * 5)
    assert not evaluate_gate(candidate_bad, baseline).passed
    assert evaluate_gate(candidate_ok, baseline).passed


# Proves any drop in verifier catch rate fails even with equal accuracy.
def test_gate_catch_rate_rule() -> None:
    baseline = _results([True] * 10, [True] * 10)
    candidate = _results([True] * 10, [True] * 9 + [False])
    result = evaluate_gate(candidate, baseline)
    assert not result.passed and "catch rate dropped" in result.messages[0]
    assert "FAIL" in result.markdown()


# Proves a missing baseline passes with a warning and the CLI wrapper reads files.
def test_gate_missing_baseline(tmp_path: Path) -> None:
    candidate = _results([True, False])
    result = evaluate_gate(candidate, None)
    assert result.passed and "WARNING" in result.messages[0]
    path = tmp_path / "cand.json"
    candidate.save(path)
    assert run_gate(path, tmp_path / "missing.json").passed
