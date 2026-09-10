"""Adapter pattern: every data source exposes the same download/clean/load/describe steps.

Approach: the pipeline (ingest script, semantic builder, eval generator) only
knows this interface. Adding a dataset = one new subclass + metric definitions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import duckdb
import pandas as pd

from loom.semantic.models import JoinDoc, MetricDoc


class DataSourceAdapter(ABC):
    """Contract for a pluggable data source."""

    #: short id used in CLI (--dataset nfip) and as a namespace in DuckDB / vector store
    name: str = ""
    description: str = ""

    @abstractmethod
    def download(self, raw_dir: Path, limit: int | None = None) -> dict[str, Path]:
        """Fetch raw files into raw_dir. Returns {table_name: file_path}. Idempotent."""

    @abstractmethod
    def clean(self, raw_files: dict[str, Path]) -> dict[str, pd.DataFrame]:
        """Parse and clean raw files into tidy DataFrames, one per table."""

    @abstractmethod
    def table_descriptions(self) -> dict[str, tuple[str, str]]:
        """{table: (description, grain)} — hand-written context for the semantic layer."""

    @abstractmethod
    def column_descriptions(self) -> dict[str, dict[str, tuple[str, str | None]]]:
        """{table: {column: (description, unit)}} for the columns worth explaining."""

    @abstractmethod
    def joins(self) -> list[JoinDoc]: ...

    @abstractmethod
    def metrics(self) -> list[MetricDoc]: ...

    def time_columns(self) -> dict[str, str]:
        """{table: time column} used for time-range profiling. Optional."""
        return {}

    def load(self, conn: duckdb.DuckDBPyConnection, frames: dict[str, pd.DataFrame]) -> None:
        """Write each DataFrame to DuckDB as <dataset>_<table>, replacing existing tables."""
        for table, df in frames.items():
            full = self.qualified(table)
            conn.register("_loom_tmp", df)
            conn.execute(f"CREATE OR REPLACE TABLE {full} AS SELECT * FROM _loom_tmp")
            conn.unregister("_loom_tmp")

    def qualified(self, table: str) -> str:
        return f"{self.name}_{table}"
