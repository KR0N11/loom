"""Profiling on a tiny DuckDB table and merge with curated metadata."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from loom.data.adapters.base import DataSourceAdapter
from loom.data.profiling import profile_dataset, profile_table
from loom.semantic.models import JoinDoc, MetricDoc


class _StubAdapter(DataSourceAdapter):
    name = "demo"

    def download(self, raw_dir: Path, limit: int | None = None) -> dict[str, Path]:
        return {}

    def clean(self, raw_files: dict[str, Path]) -> dict[str, pd.DataFrame]:
        return {}

    def table_descriptions(self) -> dict[str, tuple[str, str]]:
        return {"sales": ("Demo sales.", "one row per sale")}

    def column_descriptions(self) -> dict[str, dict[str, tuple[str, str | None]]]:
        return {"sales": {"amount": ("Sale amount.", "CAD")}}

    def joins(self) -> list[JoinDoc]:
        return []

    def metrics(self) -> list[MetricDoc]:
        return [
            MetricDoc(
                dataset="demo",
                name="revenue",
                definition="Sum of amount.",
                sql_expression="SUM(amount)",
                unit="CAD",
                grain="any",
                tables=["demo_sales"],
            )
        ]

    def time_columns(self) -> dict[str, str]:
        return {"sales": "sale_date"}


def _conn() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect()
    conn.execute(
        "CREATE TABLE demo_sales AS SELECT * FROM (VALUES "
        "(1, 'ON', DATE '2024-01-05', 100.0), (2, 'QC', DATE '2024-02-10', NULL), "
        "(3, 'ON', DATE '2024-03-01', 60.0)) AS t(sale_id, province, sale_date, amount)"
    )
    return conn


# Proves: null fraction, distinct counts, min/max and time range come from the data.
def test_profile_table() -> None:
    prof = profile_table(_conn(), "demo_sales", time_column="sale_date")
    assert prof.row_count == 3
    by_name = {c.name: c for c in prof.columns}
    assert (
        by_name["amount"].null_fraction == round(1 / 3, 4)
        or abs(by_name["amount"].null_fraction - 1 / 3) < 1e-6
    )
    assert by_name["province"].distinct_count == 2
    assert by_name["amount"].min_value == "60.0" and by_name["amount"].max_value == "100.0"
    assert prof.time_min == "2024-01-05" and prof.time_max == "2024-03-01"


# Proves: an empty table profiles without a divide-by-zero.
def test_profile_empty_table() -> None:
    conn = duckdb.connect()
    conn.execute("CREATE TABLE t (x INTEGER)")
    prof = profile_table(conn, "t")
    assert prof.row_count == 0 and prof.columns[0].null_fraction == 0.0


# Proves: curated descriptions win, uncurated columns get an auto description, metrics carry over.
def test_profile_dataset_merges_curated_metadata() -> None:
    sem = profile_dataset(_conn(), _StubAdapter())
    assert sem.dataset == "demo"
    table = sem.tables[0]
    assert table.name == "demo_sales" and table.grain == "one row per sale"
    cols = {c.name: c for c in table.columns}
    assert cols["amount"].description == "Sale amount." and cols["amount"].unit == "CAD"
    assert cols["province"].description.startswith("Column province of table demo_sales")
    assert sem.metrics[0].name == "revenue"
    assert table.time_column == "sale_date"
    # to_docs() must flatten without error so the semantic index can be built from it.
    kinds = {d.kind for d in sem.to_docs()}
    assert kinds == {"table", "column", "metric"}
