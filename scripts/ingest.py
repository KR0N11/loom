"""CLI: download, clean, load and profile one or both datasets.

Usage: uv run python scripts/ingest.py --dataset all [--limit N]
"""

from __future__ import annotations

import argparse

from loom.data.adapters import list_adapters
from loom.data.ingest import ingest


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Loom datasets into DuckDB.")
    parser.add_argument("--dataset", default="all", help="ttc | nfip | all")
    parser.add_argument("--limit", type=int, default=None, help="cap rows per table")
    args = parser.parse_args()
    names = sorted(list_adapters()) if args.dataset == "all" else [args.dataset]
    for name in names:
        report = ingest(name, limit=args.limit)
        print(f"[{report.dataset}] {report.elapsed_s:.1f}s -> {report.semantic_path}")
        for table, rows in report.rows_per_table.items():
            print(f"  {table}: {rows:,} rows")


if __name__ == "__main__":
    main()
