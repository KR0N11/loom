"""Documents that make up the semantic layer.

Approach: tables, columns, metrics and joins are all flattened into one
`SemanticDoc` shape for the vector store; richer typed models carry the curated
metadata (grain, units, allowed joins) that the executor must respect.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ColumnDoc(BaseModel):
    table: str
    name: str
    dtype: str
    description: str = ""
    unit: str | None = None
    null_fraction: float = 0.0
    distinct_count: int | None = None
    sample_values: list[str] = Field(default_factory=list)
    min_value: str | None = None
    max_value: str | None = None


class TableDoc(BaseModel):
    dataset: str
    name: str
    description: str
    grain: str
    row_count: int
    columns: list[ColumnDoc]
    time_column: str | None = None
    time_min: str | None = None
    time_max: str | None = None


class JoinDoc(BaseModel):
    dataset: str
    left_table: str
    right_table: str
    on: str
    kind: str = "left"
    note: str = ""


class MetricDoc(BaseModel):
    dataset: str
    name: str
    definition: str
    sql_expression: str
    unit: str
    grain: str
    tables: list[str]
    caveats: str = ""


class SemanticDoc(BaseModel):
    """Flattened form stored in the vector repository."""

    doc_id: str
    dataset: str
    kind: str  # table | column | metric | join
    name: str
    text: str
    metadata: dict[str, str] = Field(default_factory=dict)


class DatasetSemantics(BaseModel):
    dataset: str
    tables: list[TableDoc]
    joins: list[JoinDoc]
    metrics: list[MetricDoc]

    def to_docs(self) -> list[SemanticDoc]:
        """Flatten into retrievable documents. Text is what gets embedded and BM25-indexed."""
        docs: list[SemanticDoc] = []
        for t in self.tables:
            cols = ", ".join(f"{c.name} ({c.dtype})" for c in t.columns)
            time_note = (
                f" Time column {t.time_column} spans {t.time_min} to {t.time_max}."
                if t.time_column
                else ""
            )
            docs.append(
                SemanticDoc(
                    doc_id=f"{self.dataset}.table.{t.name}",
                    dataset=self.dataset,
                    kind="table",
                    name=t.name,
                    text=f"Table {t.name}: {t.description} Grain: {t.grain}. "
                    f"{t.row_count} rows.{time_note} Columns: {cols}.",
                    metadata={"table": t.name},
                )
            )
            for c in t.columns:
                unit = f" Unit: {c.unit}." if c.unit else ""
                samples = f" Examples: {', '.join(c.sample_values[:5])}." if c.sample_values else ""
                rng = (
                    f" Range: {c.min_value} to {c.max_value}."
                    if c.min_value is not None and c.max_value is not None
                    else ""
                )
                docs.append(
                    SemanticDoc(
                        doc_id=f"{self.dataset}.column.{t.name}.{c.name}",
                        dataset=self.dataset,
                        kind="column",
                        name=f"{t.name}.{c.name}",
                        text=f"Column {t.name}.{c.name} ({c.dtype}): {c.description}{unit}"
                        f" Null fraction {c.null_fraction:.2f}.{rng}{samples}",
                        metadata={"table": t.name, "column": c.name},
                    )
                )
        for j in self.joins:
            docs.append(
                SemanticDoc(
                    doc_id=f"{self.dataset}.join.{j.left_table}.{j.right_table}",
                    dataset=self.dataset,
                    kind="join",
                    name=f"{j.left_table}->{j.right_table}",
                    text=f"Allowed join: {j.left_table} {j.kind.upper()} JOIN {j.right_table} "
                    f"ON {j.on}. {j.note}",
                    metadata={"left": j.left_table, "right": j.right_table},
                )
            )
        for m in self.metrics:
            docs.append(
                SemanticDoc(
                    doc_id=f"{self.dataset}.metric.{m.name}",
                    dataset=self.dataset,
                    kind="metric",
                    name=m.name,
                    text=f"Metric {m.name}: {m.definition} SQL: {m.sql_expression}. "
                    f"Unit: {m.unit}. Grain: {m.grain}. Tables: {', '.join(m.tables)}. "
                    f"{m.caveats}",
                    metadata={"tables": ",".join(m.tables)},
                )
            )
        return docs
