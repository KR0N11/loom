"""Command-line entrypoint: `loom ask|ingest|build-semantic|benchmark|serve|ui`.

Approach: argparse only; every subcommand parses arguments and delegates to a
library function, so there is no business logic to test here beyond wiring
and exit codes (0 ok, 1 pipeline error, 2 verification failed).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from loom.agents.schemas import CheckStatus
from loom.config import get_settings

DATASET_CHOICES = ["ttc", "nfip", "all"]


def _dataset_list(name: str) -> list[str]:
    return ["ttc", "nfip"] if name == "all" else [name]


def cmd_ask(args: argparse.Namespace) -> int:
    from loom.agents.supervisor import run_question
    from loom.api.app import persist_state, render_memo

    settings = get_settings()
    state = run_question(args.question, args.dataset, settings=settings, injection=args.injection)
    markdown = render_memo(state)
    # Persist state and memo so `loom ask` output is auditable like the API's.
    out_dir = persist_state(state, settings).parent
    (out_dir / "memo.md").write_text(markdown)
    if args.out:
        Path(args.out).write_text(markdown)
    if args.json:
        print(state.model_dump_json(indent=2))
    else:
        print(markdown)
    status = state.verification.overall.value if state.verification else "none"
    print(
        f"\n[loom] thread={state.thread_id} verification={status} "
        f"cost=${state.usage.cost_usd:.4f} latency={state.usage.latency_s:.1f}s "
        f"memo={out_dir / 'memo.md'}",
        file=sys.stderr,
    )
    # Exit codes let scripts distinguish "ran but unverified" from "crashed".
    if state.memo is None:
        return 1
    if state.verification and state.verification.overall == CheckStatus.FAIL:
        return 2
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    from loom.data.ingest import ingest

    settings = get_settings()
    for dataset in _dataset_list(args.dataset):
        report = ingest(dataset, settings, limit=args.limit)
        print(f"[loom] ingested {dataset}: {report}")
    return 0


def cmd_build_semantic(args: argparse.Namespace) -> int:
    from loom.semantic.ingest import build_index

    settings = get_settings()
    for dataset in _dataset_list(args.dataset):
        n = build_index(dataset, settings)
        print(f"[loom] indexed {dataset}: {n} documents")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    script = Path(__file__).resolve().parents[2] / "scripts" / "run_benchmark.py"
    cmd = [sys.executable, str(script)]
    if args.sample:
        cmd += ["--sample", str(args.sample)]
    if args.injection:
        cmd += ["--injection", args.injection]
    return subprocess.call(cmd)


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("loom.api.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    ui_path = Path(__file__).resolve().parent / "ui" / "app.py"
    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(ui_path), "--server.port", str(args.port)]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="loom", description="Loom data-to-decision CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ask", help="ask a question and print the memo")
    p.add_argument("question")
    p.add_argument("--dataset", required=True, choices=["ttc", "nfip"])
    p.add_argument("--injection", default=None, help="eval only: corrupt executor output")
    p.add_argument("--json", action="store_true", help="print full state JSON instead")
    p.add_argument("--out", default=None, help="also write memo markdown here")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("ingest", help="download, clean, profile and load a dataset")
    p.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("build-semantic", help="embed and index the semantic layer")
    p.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    p.set_defaults(func=cmd_build_semantic)

    p = sub.add_parser("benchmark", help="run the eval benchmark")
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--injection", default=None, help="all or comma-separated modes")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("serve", help="run the FastAPI backend")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("ui", help="run the Streamlit front end")
    p.add_argument("--port", type=int, default=8501)
    p.set_defaults(func=cmd_ui)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
