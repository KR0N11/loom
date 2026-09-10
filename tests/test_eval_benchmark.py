"""Benchmark schema: YAML round trip, stratified sampling, composition."""

from __future__ import annotations

from pathlib import Path

from loom.eval.benchmark import Benchmark, BenchmarkQuestion, Difficulty


def _q(i: int, dataset: str, diff: Difficulty) -> BenchmarkQuestion:
    return BenchmarkQuestion(
        id=f"{dataset}-{diff.value}-{i}",
        dataset=dataset,
        question=f"q{i}",
        difficulty=diff,
        analysis_type="lookup",
        gold_sql="SELECT 1 AS n",
        gold_answer={"n": 1.0},
        gold_columns=["n"],
        gold_rows=[[1]],
        expected_tables=["t"],
    )


def _bench() -> Benchmark:
    qs = []
    for dataset in ("ttc", "nfip"):
        for diff in Difficulty:
            qs += [_q(i, dataset, diff) for i in range(5)]
    return Benchmark(questions=qs)


# Proves save/load preserves every field, including the comment header.
def test_yaml_round_trip(tmp_path: Path) -> None:
    bench = _bench()
    path = tmp_path / "q.yaml"
    bench.save(path, header="line one\nline two")
    text = path.read_text()
    assert text.startswith("# line one\n# line two\n")
    loaded = Benchmark.load(path)
    assert loaded == bench
    assert loaded.by_id("ttc-hard-0").difficulty == Difficulty.HARD


# Proves stratified sampling covers every (dataset, difficulty) stratum and is deterministic.
def test_stratified_sample() -> None:
    bench = _bench()
    sample = bench.sample(12, seed=1)
    assert len(sample.questions) == 12
    strata = {(q.dataset, q.difficulty) for q in sample.questions}
    assert len(strata) == 6
    assert [q.id for q in bench.sample(12, seed=1).questions] == [q.id for q in sample.questions]
    assert len(bench.sample(None).questions) == 30
    assert len(bench.sample(100).questions) == 30


# Proves composition counts what the report header needs.
def test_composition_and_main_number() -> None:
    bench = _bench()
    comp = bench.composition()
    assert comp["dataset"] == {"ttc": 15, "nfip": 15}
    assert comp["difficulty"] == {"easy": 10, "medium": 10, "hard": 10}
    assert comp["reviewed"] == {"unreviewed": 30}
    assert bench.questions[0].main_number == ("n", 1.0)
