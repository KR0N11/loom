"""Generators: templates against the tiny demo DB and LLM candidates with a scripted provider."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from loom.eval.benchmark import Difficulty
from loom.eval.generate import (
    GoldSQLError,
    LLMGenerator,
    Template,
    TemplateGenerator,
    materialise_gold,
)
from loom.eval.templates import DEMO_TEMPLATES, TEMPLATE_SETS
from loom.llm.fake_provider import FakeProvider


@pytest.fixture
def conn(tiny_db: Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(tiny_db), read_only=True)


# Proves every demo template yields grounded questions with executed gold answers.
def test_template_generator_demo(conn: duckdb.DuckDBPyConnection) -> None:
    questions = TemplateGenerator(conn, seed=0).generate(DEMO_TEMPLATES)
    assert len(questions) >= 6
    by_key = {q.id.split("-", 1)[1].rsplit("-", 1)[0] for q in questions}
    assert "total-by-province" in by_key
    ontario = next(q for q in questions if q.question.endswith("in ON?"))
    assert ontario.gold_answer == {"total_amount": 250.0}
    assert ontario.gold_rows == [[250.0]]
    assert ontario.difficulty == Difficulty.EASY
    assert not ontario.reviewed and ontario.generator == "template"
    # Sample values become SQL literals, never parameters left unfilled.
    assert "{" not in ontario.gold_sql


# Proves same-pool slots never repeat a value and _a/_b pairs are ordered.
def test_template_same_pool_distinct_and_ordered(conn: duckdb.DuckDBPyConnection) -> None:
    compare = next(t for t in DEMO_TEMPLATES if t.key == "province-compare")
    compare.max_variants = 10
    questions = TemplateGenerator(conn, seed=3).instantiate(compare)
    assert questions
    for q in questions:
        words = q.question.replace(".", "").split()
        a, b = words[words.index("between") + 1], words[-1]
        assert a != b and a < b


# Proves a template whose SQL breaks is skipped rather than producing a bogus gold answer.
def test_template_bad_sql_skipped(conn: duckdb.DuckDBPyConnection) -> None:
    bad = Template(
        key="bad",
        dataset="demo",
        difficulty=Difficulty.EASY,
        analysis_type="lookup",
        question="?",
        sql="SELECT nope FROM demo_sales",
        expected_tables=["demo_sales"],
    )
    assert TemplateGenerator(conn).instantiate(bad) == []


# Proves gold validation rules: empty, too many rows, no numeric column, and date conversion.
def test_materialise_gold_rules(conn: duckdb.DuckDBPyConnection) -> None:
    with pytest.raises(GoldSQLError):
        materialise_gold(conn, "SELECT amount FROM demo_sales WHERE amount < 0")
    with pytest.raises(GoldSQLError):
        materialise_gold(conn, "SELECT range AS r FROM range(100)")
    with pytest.raises(GoldSQLError):
        materialise_gold(conn, "SELECT province FROM demo_sales LIMIT 1")
    gold = materialise_gold(conn, "SELECT min(sale_date) AS d, count(*) AS n FROM demo_sales")
    assert gold.rows == [["2024-01-05", 5]]
    assert gold.answer == {"n": 5.0}


# Proves the LLM generator keeps validated candidates and drops ones whose SQL fails.
def test_llm_generator_validates(conn: duckdb.DuckDBPyConnection) -> None:
    reply = json.dumps(
        {
            "candidates": [
                {
                    "question": "Total sales?",
                    "gold_sql": "SELECT sum(amount) AS total FROM demo_sales",
                    "analysis_type": "lookup",
                    "difficulty": "easy",
                    "expected_tables": ["demo_sales"],
                },
                {
                    "question": "Broken",
                    "gold_sql": "SELECT missing FROM demo_sales",
                    "analysis_type": "lookup",
                    "difficulty": "easy",
                },
            ]
        }
    )
    provider = FakeProvider([reply])
    questions = LLMGenerator(conn, provider).generate("demo", "ctx", n=1)
    assert len(questions) == 1
    assert questions[0].generator == "llm"
    assert questions[0].gold_answer == {"total": 510.0}
    assert provider.calls[0][1][0].content.startswith("Dataset: demo")


# Proves the real template sets are well-formed (every slot has a sampler).
def test_real_templates_well_formed() -> None:
    import re

    for dataset in ("ttc", "nfip"):
        for t in TEMPLATE_SETS[dataset]:
            slots = set(re.findall(r"{(\w+)}", t.sql)) | set(re.findall(r"{(\w+)}", t.question))
            assert slots == set(t.params), t.key
            assert t.dataset == dataset
