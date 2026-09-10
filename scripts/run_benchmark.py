"""Run the benchmark and write results.json + report.md.

Usage: uv run python scripts/run_benchmark.py [--sample 30] [--seed 0] [--injection all|scale,sign]
       [--injection-sample 10] [--baselines qwen,tapas,router] [--resume] [--out outputs/eval/run1]
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

from loom.config import get_settings
from loom.eval.benchmark import Benchmark
from loom.eval.injection_runner import resolve_modes
from loom.eval.report import write_report
from loom.eval.runner import EvalResults, EvalRow, run_benchmark


def _progress(row: EvalRow) -> None:
    tag = f"[{row.injection}]" if row.injection else ""
    status = "ERR" if row.error else ("ok" if row.numeric_match else "miss")
    print(
        f"{row.id}{tag}: {status} verifier={row.verifier_overall} {row.latency_s:.1f}s", flush=True
    )


def _run_baselines(names: list[str], bench: Benchmark, results: EvalResults, settings) -> None:  # type: ignore[no-untyped-def]
    from loom.sandbox import get_executor
    from loom.semantic.ingest import load_semantics
    from loom.semantic.search import build_context

    sandbox = get_executor(settings)
    contexts = {
        d: build_context([], load_semantics(d, settings)).context_text
        for d in {q.dataset for q in bench.questions}
    }
    for name in names:
        if name == "qwen":
            from loom.eval.baselines.small_sql import QwenSQLBaseline

            base = QwenSQLBaseline(settings, sandbox)
            results.baseline_rows += [base.answer(q, contexts[q.dataset]) for q in bench.questions]
        elif name == "sqlcoder":
            from loom.eval.baselines.sqlcoder import SQLCoderBaseline

            base_sc = SQLCoderBaseline(settings, sandbox)
            results.baseline_rows += [
                base_sc.answer(q, contexts[q.dataset]) for q in bench.questions
            ]
        elif name == "tapas":
            from loom.eval.baselines.tapas import TapasLookupBaseline

            tapas = TapasLookupBaseline(settings)
            results.baseline_rows += [tapas.answer(q) for q in bench.questions]
        elif name == "router":
            from loom.eval.baselines.zero_shot_router import routing_accuracy, zero_shot_classifier

            results.routing["bart-large-mnli zero-shot"] = routing_accuracy(
                bench, zero_shot_classifier(settings.zero_shot_model)
            )
        else:
            raise SystemExit(f"unknown baseline {name!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("benchmarks/questions.yaml"))
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--injection", default=None, help="'all' or comma list of modes")
    parser.add_argument("--injection-sample", type=int, default=None)
    parser.add_argument("--baselines", default=None, help="comma list: qwen,sqlcoder,tapas,router")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    settings = get_settings()
    bench = Benchmark.load(args.benchmark)
    out_dir = args.out or (
        Path(settings.output_dir) / "eval" / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    results = run_benchmark(
        bench,
        settings=settings,
        sample_n=args.sample,
        seed=args.seed,
        injection_modes=resolve_modes(args.injection),
        injection_sample_n=args.injection_sample,
        out_dir=out_dir,
        resume=args.resume,
        on_progress=_progress,
    )
    if args.baselines:
        sampled = bench.sample(args.sample, args.seed)
        _run_baselines([b.strip() for b in args.baselines.split(",")], sampled, results, settings)
        results.save(out_dir / "results.json")
    report = write_report(results, out_dir / "report.md", bench.composition()["reviewed"])
    print(report)
    print(f"\nresults: {out_dir / 'results.json'}\nreport:  {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
