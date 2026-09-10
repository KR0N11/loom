"""Metric edge cases: tolerance, NaN, empty rows, normalisation."""

from __future__ import annotations

import math

from loom.eval.metrics import (
    execution_accuracy,
    numeric_match,
    percentile,
    retrieval_precision_recall,
    verifier_catch,
    within_tolerance,
)


# Proves row order and int/float/str formatting do not matter, but values do.
def test_execution_accuracy_normalises_rows() -> None:
    gold = [["ON", 250.0], ["QC", 200]]
    assert execution_accuracy([["qc", "200"], ["ON ", 250]], gold)
    assert not execution_accuracy([["ON", 250.0], ["QC", 201]], gold)


# Proves an empty prediction only matches an empty gold.
def test_execution_accuracy_empty() -> None:
    assert execution_accuracy([], [])
    assert not execution_accuracy([], [[1]])
    assert not execution_accuracy([[1]], [])


# Proves NaN cells compare as missing rather than never-equal.
def test_execution_accuracy_nan_is_none() -> None:
    assert execution_accuracy([[math.nan]], [[None]])


# Proves the 1% relative tolerance and the absolute floor at zero.
def test_within_tolerance() -> None:
    assert within_tolerance(100.5, 100.0)
    assert not within_tolerance(102.0, 100.0)
    assert within_tolerance(0.0, 0.0)
    assert not within_tolerance(math.nan, 1.0)


# Proves main-number match and coverage over all gold numbers.
def test_numeric_match_main_and_coverage() -> None:
    gold = {"total": 1000.0, "share": 0.25}
    main_ok, coverage = numeric_match([1005.0], gold)
    assert main_ok and coverage == 0.5
    main_ok, coverage = numeric_match({"a": 0.251, "b": 999.0}, gold)
    assert main_ok and coverage == 1.0
    assert numeric_match([], gold) == (False, 0.0)
    assert numeric_match([1.0], {}) == (False, 0.0)


# Proves precision/recall handle empty sets without dividing by zero.
def test_retrieval_precision_recall() -> None:
    assert retrieval_precision_recall({"a", "b", "c"}, {"a", "d"}) == (1 / 3, 0.5)
    assert retrieval_precision_recall(set(), {"a"}) == (0.0, 0.0)
    assert retrieval_precision_recall({"a"}, set()) == (None, None)


# Proves a catch is only counted for injected runs that ended in FAIL.
def test_verifier_catch() -> None:
    assert verifier_catch("fail", True) is True
    assert verifier_catch("pass", True) is False
    assert verifier_catch("fail", False) is None


# Proves nearest-rank p95 and the empty-list guard.
def test_percentile() -> None:
    assert percentile([], 95) == 0.0
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert percentile([5.0], 50) == 5.0
