"""Code that runs INSIDE the sandbox.

Approach: this module is deliberately dependency-light (duckdb + pandas only)
so the same file can be shipped into a Docker image or a Lambda layer. It
reads one SandboxRequest as JSON on stdin, executes it with hard limits, and
writes one SandboxResult as JSON on stdout. Nothing else is trusted: the SQL is
checked to be a single read-only statement, pandas code runs with a stripped
builtins table, and the row limit is applied before rows leave the process.
"""

from __future__ import annotations

import datetime as dt
import decimal
import math
import re
import sys
import time
from typing import Any

import duckdb
import pandas as pd

from loom.sandbox.base import SandboxRequest, SandboxResult

# Statements that could change data, touch the filesystem or the network.
_FORBIDDEN_PREFIXES = (
    "insert",
    "update",
    "delete",
    "create",
    "drop",
    "alter",
    "copy",
    "attach",
    "detach",
    "install",
    "load",
    "pragma",
    "set",
    "reset",
    "export",
    "import",
    "call",
    "truncate",
    "vacuum",
    "checkpoint",
    "begin",
    "commit",
    "rollback",
    "use",
)

# Functions that read or write files/network even inside a SELECT.
_FORBIDDEN_FUNCTIONS = re.compile(
    r"\b(read_csv|read_csv_auto|read_parquet|read_json|read_json_auto|read_text|read_blob|"
    r"glob|copy|httpfs|sniff_csv|parquet_scan|csv_scan)\s*\(",
    re.IGNORECASE,
)

# Builtins that let pandas code escape the namespace: file, import, dynamic code.
_SAFE_BUILTINS = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in (
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "filter",
        "float",
        "int",
        "isinstance",
        "len",
        "list",
        "map",
        "max",
        "min",
        "range",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
        "True",
        "False",
        "None",
        "ValueError",
        "KeyError",
        "TypeError",
    )
}


class SandboxRejectedError(ValueError):
    """The request was refused before running anything."""


def strip_sql_comments(sql: str) -> str:
    """Remove -- and /* */ comments so a forbidden keyword cannot hide behind one."""
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"--[^\n]*", " ", sql)
    return sql.strip()


def validate_sql(sql: str) -> str:
    """Return the cleaned statement or raise. Without this an LLM could emit
    `COPY nfip_claims TO '/etc/cron.d/x'` or `DROP TABLE nfip_claims`."""
    cleaned = strip_sql_comments(sql).rstrip(";").strip()
    # Empty query is a model bug, not something to run.
    if not cleaned:
        raise SandboxRejectedError("empty SQL")
    # A second statement after a semicolon would run outside the LIMIT wrapper.
    if ";" in cleaned:
        raise SandboxRejectedError("multiple statements are not allowed")
    first = cleaned.split(None, 1)[0].lower()
    # Only SELECT / WITH (CTEs) / DESCRIBE / SHOW-style reads are allowed.
    if first in _FORBIDDEN_PREFIXES or first not in {"select", "with", "from", "describe", "show"}:
        raise SandboxRejectedError(f"statement type {first!r} is not allowed (read-only sandbox)")
    # File/network table functions bypass the read-only connection.
    if _FORBIDDEN_FUNCTIONS.search(cleaned):
        raise SandboxRejectedError("file or network table functions are not allowed")
    return cleaned


def to_json_safe(value: Any) -> Any:
    """Convert DuckDB/pandas scalars into plain JSON values."""
    # Missing values of any flavour become null.
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (dt.datetime, dt.date, dt.time, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, dt.timedelta):
        return value.total_seconds()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    # numpy scalars expose .item() to become native Python types.
    if hasattr(value, "item") and not isinstance(value, (list, dict, str)):
        try:
            return to_json_safe(value.item())
        except (ValueError, TypeError):
            return str(value)
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (str, int, bool, float)):
        return value
    return str(value)


def frame_to_result(df: pd.DataFrame, row_limit: int, elapsed: float) -> SandboxResult:
    """Apply the row limit and flatten a DataFrame into columns/rows."""
    truncated = len(df) > row_limit
    trimmed = df.head(row_limit)
    rows = [[to_json_safe(v) for v in row] for row in trimmed.itertuples(index=False, name=None)]
    return SandboxResult(
        ok=True,
        columns=[str(c) for c in trimmed.columns],
        rows=rows,
        row_count=len(rows),
        truncated=truncated,
        elapsed_s=elapsed,
    )


def run_sql(request: SandboxRequest, conn: duckdb.DuckDBPyConnection) -> SandboxResult:
    """Execute one validated read-only statement with a LIMIT wrapper."""
    cleaned = validate_sql(request.code)
    # Fetch one extra row so we can report truncation without counting everything.
    wrapped = f"SELECT * FROM ({cleaned}) AS _loom_q LIMIT {request.row_limit + 1}"
    start = time.perf_counter()
    cursor = conn.execute(wrapped)
    columns = [d[0] for d in cursor.description or []]
    raw_rows = cursor.fetchall()
    elapsed = time.perf_counter() - start
    truncated = len(raw_rows) > request.row_limit
    rows = [[to_json_safe(v) for v in row] for row in raw_rows[: request.row_limit]]
    return SandboxResult(
        ok=True,
        columns=columns,
        rows=rows,
        row_count=len(rows),
        truncated=truncated,
        elapsed_s=elapsed,
    )


def run_pandas(request: SandboxRequest, conn: duckdb.DuckDBPyConnection) -> SandboxResult:
    """Exec pandas code with pre-loaded tables and a restricted namespace."""
    import numpy as np

    namespace: dict[str, Any] = {"__builtins__": dict(_SAFE_BUILTINS), "pd": pd, "np": np}
    # Each requested table becomes a DataFrame variable with the table's name.
    for table in request.tables:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
            raise SandboxRejectedError(f"bad table name {table!r}")
        namespace[table] = conn.table(table).df()
    start = time.perf_counter()
    exec(compile(request.code, "<sandbox>", "exec"), namespace)  # noqa: S102
    elapsed = time.perf_counter() - start
    result = namespace.get("result")
    # The contract is "assign a DataFrame to `result`"; anything else is unusable downstream.
    if isinstance(result, pd.Series):
        result = result.reset_index()
    if not isinstance(result, pd.DataFrame):
        raise SandboxRejectedError(
            "pandas code must assign a DataFrame to a variable named `result`"
        )
    return frame_to_result(result, request.row_limit, elapsed)


def apply_memory_limit(memory_mb: int) -> None:
    """Cap address space on Linux. macOS ignores/rejects RLIMIT_AS for practical
    purposes (allocations fail early with mmap errors), so it is skipped there."""
    if sys.platform != "linux":
        return
    import resource

    limit = memory_mb * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ValueError, OSError):
        # A hard limit lower than requested is not fatal; the parent still enforces time.
        pass


def execute(request: SandboxRequest) -> SandboxResult:
    """Run one request and never raise: every failure becomes ok=False with a message."""
    start = time.perf_counter()
    apply_memory_limit(request.memory_mb)
    try:
        conn = duckdb.connect(request.duckdb_path, read_only=True)
    except Exception as exc:  # noqa: BLE001
        return SandboxResult(ok=False, error=f"cannot open database: {exc}", elapsed_s=0.0)
    try:
        if request.language == "sql":
            return run_sql(request, conn)
        if request.language == "pandas":
            return run_pandas(request, conn)
        return SandboxResult(ok=False, error=f"unknown language {request.language!r}")
    except SandboxRejectedError as exc:
        return SandboxResult(
            ok=False, error=f"rejected: {exc}", elapsed_s=time.perf_counter() - start
        )
    except Exception as exc:  # noqa: BLE001
        # Error text goes back to the executor agent so it can self-correct.
        return SandboxResult(
            ok=False,
            error=f"{type(exc).__name__}: {str(exc)[:2000]}",
            elapsed_s=time.perf_counter() - start,
        )
    finally:
        conn.close()


def main() -> None:
    """stdin JSON -> stdout JSON. Used by subprocess, Docker and Lambda backends."""
    raw = sys.stdin.read()
    try:
        request = SandboxRequest.model_validate_json(raw)
    except Exception as exc:  # noqa: BLE001
        sys.stdout.write(SandboxResult(ok=False, error=f"bad request: {exc}").model_dump_json())
        return
    result = execute(request)
    result.backend = "runner"
    sys.stdout.write(result.model_dump_json())


if __name__ == "__main__":
    main()
