"""Build (or rebuild) the semantic-layer vector index for one or all datasets."""

from __future__ import annotations

import argparse
import time

from loom.config import get_settings
from loom.semantic.ingest import build_index

DATASETS = ["ttc", "nfip"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    args = parser.parse_args()
    settings = get_settings()
    targets = DATASETS if args.dataset == "all" else [args.dataset]
    for ds in targets:
        start = time.perf_counter()
        n = build_index(ds, settings)
        print(
            f"{ds}: indexed {n} docs in {time.perf_counter() - start:.1f}s "
            f"({settings.vector_backend}, {settings.embedding_backend})"
        )


if __name__ == "__main__":
    main()
