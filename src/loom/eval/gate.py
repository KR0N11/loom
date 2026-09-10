"""CI regression gate.

Approach: compare a candidate results.json with the baseline committed from
main. Execution accuracy may not drop more than 5 points; verifier catch rate
may not drop at all. A missing baseline passes with a warning so the very first
run can seed it.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from loom.eval.runner import EvalResults

MAX_ACCURACY_DROP = 0.05


class GateResult(BaseModel):
    passed: bool
    messages: list[str]
    candidate_accuracy: float
    baseline_accuracy: float | None
    candidate_catch: float | None
    baseline_catch: float | None

    def markdown(self) -> str:
        status = "PASS" if self.passed else "FAIL"

        def fmt(v: float | None) -> str:
            return "n/a" if v is None else f"{100 * v:.1f}%"

        lines = [
            f"## Eval gate: {status}",
            "",
            "| Metric | Baseline (main) | Candidate | Rule |",
            "|---|---|---|---|",
            f"| Execution accuracy | {fmt(self.baseline_accuracy)} | {fmt(self.candidate_accuracy)} | drop ≤ 5 pts |",
            f"| Verifier catch rate | {fmt(self.baseline_catch)} | {fmt(self.candidate_catch)} | no drop |",
            "",
        ]
        lines += [f"- {m}" for m in self.messages]
        return "\n".join(lines)


def evaluate_gate(candidate: EvalResults, baseline: EvalResults | None) -> GateResult:
    cand_acc = candidate.execution_accuracy()
    cand_catch = candidate.catch_rate()
    messages: list[str] = []
    # No baseline yet: cannot regress against nothing, so pass and say so loudly.
    if baseline is None:
        messages.append("WARNING: no baseline results found; gate passes by default.")
        return GateResult(
            passed=True,
            messages=messages,
            candidate_accuracy=cand_acc,
            baseline_accuracy=None,
            candidate_catch=cand_catch,
            baseline_catch=None,
        )
    base_acc = baseline.execution_accuracy()
    base_catch = baseline.catch_rate()
    passed = True
    # Rule 1: accuracy may drop at most 5 points (tiny epsilon for float noise).
    if cand_acc < base_acc - MAX_ACCURACY_DROP - 1e-9:
        passed = False
        messages.append(
            f"execution accuracy dropped {100 * (base_acc - cand_acc):.1f} points (limit 5)"
        )
    # Rule 2: verifier catch rate may not drop at all when both runs measured it.
    if base_catch is not None and cand_catch is not None and cand_catch < base_catch - 1e-9:
        passed = False
        messages.append(
            f"verifier catch rate dropped from {100 * base_catch:.1f}% to {100 * cand_catch:.1f}%"
        )
    elif base_catch is not None and cand_catch is None:
        messages.append("WARNING: candidate run has no injection rows; catch rate not compared.")
    if passed and not messages:
        messages.append("no regression detected")
    return GateResult(
        passed=passed,
        messages=messages,
        candidate_accuracy=cand_acc,
        baseline_accuracy=base_acc,
        candidate_catch=cand_catch,
        baseline_catch=base_catch,
    )


def run_gate(candidate_path: Path, baseline_path: Path | None) -> GateResult:
    candidate = EvalResults.load(candidate_path)
    baseline = None
    if baseline_path is not None and baseline_path.exists():
        baseline = EvalResults.load(baseline_path)
    return evaluate_gate(candidate, baseline)
