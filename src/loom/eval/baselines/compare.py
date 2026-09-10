"""Small-model vs LLM comparison table."""

from __future__ import annotations

from collections import defaultdict

from loom.eval.metrics import mean
from loom.eval.runner import BaselineRow, EvalResults, EvalRow


def _pct(v: float) -> str:
    return f"{100 * v:.1f}%"


def comparison_table(results: EvalResults) -> str:
    """One row per system, restricted to the question ids every baseline attempted."""
    by_name: dict[str, list[BaselineRow]] = defaultdict(list)
    for b in results.baseline_rows:
        by_name[b.baseline].append(b)
    llm_rows: list[EvalRow] = results.main_rows()
    lines = [
        "| System | n | Exec acc | Numeric match | Mean latency | Mean cost |",
        "|---|---|---|---|---|---|",
        f"| LLM pipeline ({results.model or 'llm'}) | {len(llm_rows)} | "
        f"{_pct(mean([float(r.exec_acc) for r in llm_rows]))} | "
        f"{_pct(mean([float(r.numeric_match) for r in llm_rows]))} | "
        f"{mean([r.latency_s for r in llm_rows]):.1f}s | "
        f"${mean([r.cost_usd for r in llm_rows]):.4f} |",
    ]
    for name in sorted(by_name):
        rows = [b for b in by_name[name] if not (b.error or "").startswith("skipped")]
        lines.append(
            f"| {name} | {len(rows)} | {_pct(mean([float(r.exec_acc) for r in rows]))} | "
            f"{_pct(mean([float(r.numeric_match) for r in rows]))} | "
            f"{mean([r.latency_s for r in rows]):.1f}s | ${mean([r.cost_usd for r in rows]):.4f} |"
        )
    return "\n".join(lines)
