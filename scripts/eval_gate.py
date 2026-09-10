"""CI gate: exit 1 if execution accuracy drops >5 points or verifier catch rate drops at all.

Usage: uv run python scripts/eval_gate.py <candidate results.json> [--baseline benchmarks/baseline_results.json]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loom.eval.gate import run_gate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--baseline", type=Path, default=Path("benchmarks/baseline_results.json"))
    args = parser.parse_args(argv)
    result = run_gate(args.candidate, args.baseline)
    print(result.markdown())
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
