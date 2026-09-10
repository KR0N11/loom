"""Profile loaded DuckDB tables and merge with hand-written metadata.

Approach: profiling is computed from the data (row counts, dtypes, null
fractions, distinct counts, ranges, sample values) and *merged* with the
adapter's curated descriptions, joins and metrics into a `DatasetSemantics`
document. Curated text wins where both exist; auto text fills the rest so
every column is at least discoverable by the retriever.
"""

from __future__ import annotations

import duckdb
from pydantic import BaseModel, Field

from loom.data.adapters.base import DataSourceAdapter
from loom.semantic.models import ColumnDoc, DatasetSemantics, TableDoc


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    null_fraction: float
    distinct_count: int
    min_value: str | None = None
    max_value: str | None = None
    sample_values: list[str] = Field(default_factory=list)


class TableProfile(BaseModel):
    table: str
    row_count: int
    columns: list[ColumnProfile]
    time_column: str | None = None
    time_min: str | None = None
    time_max: str | None = None


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _one(conn: duckdb.DuckDBPyConnection, sql: str) -> tuple:
    """fetchone() that never returns None (aggregates always yield one row)."""
    row = conn.execute(sql).fetchone()
    return row if row is not None else ()


def profile_table(
    conn: duckdb.DuckDBPyConnection, table: str, time_column: str | None = None
) -> TableProfile:
    row_count = int(_one(conn, f"SELECT COUNT(*) FROM {_q(table)}")[0])
    cols = conn.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = ? ORDER BY ordinal_position",
        [table],
    ).fetchall()
    profiles: list[ColumnProfile] = []
    for name, dtype in cols:
        nulls, distinct, mn, mx = _one(
            conn,
            f"SELECT COUNT(*) - COUNT({_q(name)}), COUNT(DISTINCT {_q(name)}), "
            f"MIN({_q(name)})::VARCHAR, MAX({_q(name)})::VARCHAR FROM {_q(table)}",
        )
        samples = conn.execute(
            f"SELECT DISTINCT {_q(name)}::VARCHAR FROM {_q(table)} "
            f"WHERE {_q(name)} IS NOT NULL LIMIT 5"
        ).fetchall()
        profiles.append(
            ColumnProfile(
                name=name,
                dtype=dtype,
                # Guard against an empty table so profiling never divides by zero.
                null_fraction=(nulls / row_count) if row_count else 0.0,
                distinct_count=int(distinct),
                min_value=mn,
                max_value=mx,
                sample_values=[s[0] for s in samples],
            )
        )
    time_min = time_max = None
    # Time range is what the planner uses to sanity-check a question's period.
    if time_column and any(p.name == time_column for p in profiles):
        time_min, time_max = _one(
            conn,
            f"SELECT MIN({_q(time_column)})::VARCHAR, MAX({_q(time_column)})::VARCHAR "
            f"FROM {_q(table)}",
        )
    return TableProfile(
        table=table,
        row_count=row_count,
        columns=profiles,
        time_column=time_column,
        time_min=time_min,
        time_max=time_max,
    )


def profile_dataset(
    conn: duckdb.DuckDBPyConnection, adapter: DataSourceAdapter
) -> DatasetSemantics:
    """Profile every table the adapter describes and merge with curated metadata."""
    descriptions = adapter.table_descriptions()
    col_desc = adapter.column_descriptions()
    time_cols = adapter.time_columns()
    tables: list[TableDoc] = []
    for short, (desc, grain) in descriptions.items():
        full = adapter.qualified(short)
        prof = profile_table(conn, full, time_cols.get(short))
        columns: list[ColumnDoc] = []
        for cp in prof.columns:
            hand = col_desc.get(short, {}).get(cp.name)
            # Curated description wins; otherwise an auto line keeps the column retrievable.
            description = hand[0] if hand else f"Column {cp.name} of table {full} ({cp.dtype})."
            unit = hand[1] if hand else None
            columns.append(
                ColumnDoc(
                    table=full,
                    name=cp.name,
                    dtype=cp.dtype,
                    description=description,
                    unit=unit,
                    null_fraction=round(cp.null_fraction, 4),
                    distinct_count=cp.distinct_count,
                    sample_values=cp.sample_values,
                    min_value=cp.min_value,
                    max_value=cp.max_value,
                )
            )
        tables.append(
            TableDoc(
                dataset=adapter.name,
                name=full,
                description=desc,
                grain=grain,
                row_count=prof.row_count,
                columns=columns,
                time_column=prof.time_column,
                time_min=prof.time_min,
                time_max=prof.time_max,
            )
        )
    return DatasetSemantics(
        dataset=adapter.name, tables=tables, joins=adapter.joins(), metrics=adapter.metrics()
    )
