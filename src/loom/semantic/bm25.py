"""Tiny BM25 index used for the keyword half of hybrid search.

Approach: pure Python, no extra dependency. Documents are short (one table,
column or metric each), so building the index in memory per dataset is cheap.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens; snake_case splits into its parts so `min_delay` matches `delay`."""
    return _TOKEN.findall(text.lower())


class BM25Index:
    def __init__(self, doc_ids: list[str], texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.doc_ids = doc_ids
        self.k1 = k1
        self.b = b
        self._tfs: list[Counter[str]] = [Counter(tokenize(t)) for t in texts]
        self._lens = [sum(tf.values()) for tf in self._tfs]
        self._avg_len = (sum(self._lens) / len(self._lens)) if self._lens else 0.0
        df: Counter[str] = Counter()
        for tf in self._tfs:
            df.update(tf.keys())
        n = len(self._tfs)
        # Standard BM25 idf with the +1 smoothing so rare terms never go negative.
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        q_tokens = tokenize(query)
        scored: list[tuple[str, float]] = []
        for doc_id, tf, length in zip(self.doc_ids, self._tfs, self._lens, strict=True):
            score = 0.0
            for tok in q_tokens:
                # Tokens absent from the document contribute nothing.
                if tok not in tf:
                    continue
                f = tf[tok]
                denom = f + self.k1 * (1 - self.b + self.b * length / (self._avg_len or 1.0))
                score += self._idf.get(tok, 0.0) * f * (self.k1 + 1) / denom
            # Only documents that matched at least one term are results.
            if score > 0:
                scored.append((doc_id, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
