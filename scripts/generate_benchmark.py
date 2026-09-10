"""Generate benchmarks/questions.yaml from templates (+ optional LLM candidates).

Usage: uv run python scripts/generate_benchmark.py [--n-llm 30] [--seed 0] [--out benchmarks/questions.yaml]
LLM candidates need a working provider; templates alone produce the full benchmark.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

from loom.config import get_settings
from loom.data.duckdb_store import connect
from loom.eval.benchmark import Benchmark
from loom.eval.generate import LLMGenerator, TemplateGenerator
from loom.eval.templates import TEMPLATE_SETS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-llm", type=int, default=0, help="LLM candidates per dataset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/questions.yaml"))
    parser.add_argument("--datasets", default="ttc,nfip")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    conn = connect(settings, read_only=True)
    datasets = [d.strip() for d in args.datasets.split(",")]
    questions = []
    # Templates first: grounded in real values and guaranteed to execute.
    for dataset in datasets:
        gen = TemplateGenerator(conn, seed=args.seed)
        questions += gen.generate(TEMPLATE_SETS[dataset])
    # Optional LLM candidates, validated against the data before being kept.
    if args.n_llm > 0:
        from loom.llm import get_provider
        from loom.semantic.ingest import load_semantics
        from loom.semantic.search import build_context

        provider = get_provider(settings)
        for dataset in datasets:
            semantics = load_semantics(dataset, settings)
            context = build_context([], semantics).context_text
            questions += LLMGenerator(conn, provider).generate(dataset, context, args.n_llm)
    bench = Benchmark(questions=questions)
    header = (
        f"Loom benchmark, generated {dt.datetime.now():%Y-%m-%d %H:%M} seed={args.seed}\n"
        f"generators: template={bench.composition()['generator'].get('template', 0)} "
        f"llm={bench.composition()['generator'].get('llm', 0)}\n"
        "All questions start reviewed: false. Flip to true (and add review_note) after a human check.\n"
        "Add LLM-proposed candidates with: uv run python scripts/generate_benchmark.py --n-llm 30"
    )
    bench.save(args.out, header=header)
    print(f"wrote {len(bench.questions)} questions to {args.out}")
    for key, counts in bench.composition().items():
        print(f"  {key}: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
