"""Markdown eval report.

Approach: everything is derived from `EvalResults` so the same file that CI
gates on is the file humans read. Tables are grouped by difficulty, dataset and
analysis type, with verifier catch rates per injection mode and cost/latency.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from loom.eval.injection_runner import catch_rates
from loom.eval.metrics import mean, percentile
from loom.eval.runner import EvalResults, EvalRow


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _group_table(rows: list[EvalRow], key: Callable[[EvalRow], str], label: str) -> str:
    groups: dict[str, list[EvalRow]] = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    lines = [
        f"| {label} | n | Exec acc | Numeric match | Zero-correction | Retrieval P | Retrieval R | Mean cost | Mean latency |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in sorted(groups):
        g = groups[name]
        precision = [r.retrieval_p for r in g if r.retrieval_p is not None]
        recall = [r.retrieval_r for r in g if r.retrieval_r is not None]
        lines.append(
            f"| {name} | {len(g)} | {_pct(mean([float(r.exec_acc) for r in g]))} | "
            f"{_pct(mean([float(r.numeric_match) for r in g]))} | "
            f"{_pct(mean([float(r.zero_correction) for r in g]))} | "
            f"{_pct(mean(precision) if precision else None)} | "
            f"{_pct(mean(recall) if recall else None)} | "
            f"${mean([r.cost_usd for r in g]):.4f} | {mean([r.latency_s for r in g]):.1f}s |"
        )
    return "\n".join(lines)


def render_report(results: EvalResults, benchmark_counts: dict[str, int] | None = None) -> str:
    rows = results.main_rows()
    summary = results.summary()
    catches = catch_rates(results)
    parts: list[str] = []
    parts.append("# Loom benchmark report\n")
    parts.append(f"Run started: {results.started_at}  \nModel: `{results.model or 'n/a'}`  \n")
    if benchmark_counts:
        parts.append(
            f"Questions: {benchmark_counts.get('reviewed', 0)} reviewed, "
            f"{benchmark_counts.get('unreviewed', 0)} unreviewed\n"
        )
    # Headline numbers first so the gate result is explainable at a glance.
    parts.append("## Overall\n")
    parts.append("| Metric | Value |\n|---|---|")
    parts.append(f"| Questions run | {summary['n']} |")
    parts.append(f"| Execution accuracy | {_pct(summary['execution_accuracy'])} |")
    parts.append(f"| Numerical tolerance match (1%) | {_pct(summary['numeric_match'])} |")
    parts.append(f"| Zero human correction | {_pct(summary['zero_correction'])} |")
    parts.append(f"| Semantic retrieval precision | {_pct(summary['retrieval_precision'])} |")
    parts.append(f"| Semantic retrieval recall | {_pct(summary['retrieval_recall'])} |")
    parts.append(f"| Verifier catch rate (injected) | {_pct(summary['verifier_catch_rate'])} |")
    parts.append(f"| Verifier false-alarm rate | {_pct(summary['false_alarm_rate'])} |")
    parts.append(f"| Mean cost per question | ${summary['mean_cost_usd']:.4f} |")
    latencies = [r.latency_s for r in rows]
    parts.append(
        f"| Latency mean / p95 | {mean(latencies):.1f}s / {percentile(latencies, 95):.1f}s |"
    )
    errors = [r for r in rows if r.error]
    parts.append(f"| Questions with errors | {len(errors)} |\n")
    parts.append("## By difficulty\n")
    parts.append(_group_table(rows, lambda r: r.difficulty, "Difficulty") + "\n")
    parts.append("## By dataset\n")
    parts.append(_group_table(rows, lambda r: r.dataset, "Dataset") + "\n")
    parts.append("## By analysis type\n")
    parts.append(_group_table(rows, lambda r: r.analysis_type, "Type") + "\n")
    parts.append("## Verifier catch rate by injection mode\n")
    parts.append("| Injection mode | Catch rate |\n|---|---|")
    for mode, rate in catches.per_mode.items():
        parts.append(f"| {mode} | {_pct(rate)} |")
    parts.append(f"| **overall** ({catches.n_injected} injected runs) | {_pct(catches.overall)} |")
    parts.append(f"| false alarms on correct clean runs | {_pct(catches.false_alarm_rate)} |\n")
    routed = [r for r in rows if r.planner_type]
    if routed:
        acc = mean([float(r.planner_type == r.analysis_type) for r in routed])
        parts.append(f"Planner routing accuracy vs gold analysis type: {_pct(acc)}\n")
    if results.routing:
        parts.append("## Zero-shot router baseline\n")
        parts.append("| Router | Accuracy |\n|---|---|")
        for name, acc in results.routing.items():
            parts.append(f"| {name} | {_pct(acc)} |")
        parts.append("")
    if results.baseline_rows:
        from loom.eval.baselines.compare import comparison_table

        parts.append("## Small-model vs LLM comparison\n")
        parts.append(comparison_table(results) + "\n")
    if errors:
        parts.append("## Errors\n")
        for r in errors[:20]:
            parts.append(f"- `{r.id}`: {r.error}")
        parts.append("")
    return "\n".join(parts)


def write_report(
    results: EvalResults, path: Path, benchmark_counts: dict[str, int] | None = None
) -> str:
    text = render_report(results, benchmark_counts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text
