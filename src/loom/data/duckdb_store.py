"""One place to open the analytical DuckDB database."""

from __future__ import annotations

from pathlib import Path

import duckdb

from loom.config import Settings, get_settings


def connect(settings: Settings | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open the configured DuckDB file, creating parent folders on first write."""
    settings = settings or get_settings()
    path = Path(settings.duckdb_path)
    # Read-only opens fail on a missing file, so create the folder only when writing.
    if not read_only:
        path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path), read_only=read_only)


def list_tables(conn: duckdb.DuckDBPyConnection, dataset: str | None = None) -> list[str]:
    rows = conn.execute("SELECT table_name FROM information_schema.tables ORDER BY 1").fetchall()
    names = [r[0] for r in rows]
    # Dataset tables are namespaced by prefix, e.g. nfip_claims.
    if dataset:
        names = [n for n in names if n.startswith(f"{dataset}_")]
    return names
