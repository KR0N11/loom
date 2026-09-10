"""Rough per-million-token prices so every trace and eval row carries a cost.

Prices are approximate list prices; override via PRICE_TABLE if they change.
"""

from __future__ import annotations

# (input $/M tokens, output $/M tokens)
PRICE_TABLE: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (3.0, 15.0),
    "claude-opus-5": (15.0, 75.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Look up by longest matching prefix so Bedrock ids like anthropic.claude-sonnet-5-v1 match."""
    best: tuple[float, float] | None = None
    best_len = -1
    for key, prices in PRICE_TABLE.items():
        if key in model and len(key) > best_len:
            best, best_len = prices, len(key)
    if best is None:
        return 0.0
    return (input_tokens * best[0] + output_tokens * best[1]) / 1_000_000
