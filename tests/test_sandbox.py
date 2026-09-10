"""Sandbox tests: read-only SQL enforcement, limits, pandas namespace, backend transports."""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path
from unittest import mock

import pytest
from botocore.stub import Stubber

from loom.config import SandboxBackend, Settings
from loom.sandbox.base import SandboxRequest, SandboxResult
from loom.sandbox.docker_executor import CONTAINER_DB_PATH, DockerExecutor
from loom.sandbox.factory import get_executor
from loom.sandbox.lambda_executor import LambdaExecutor
from loom.sandbox.runner import execute, to_json_safe, validate_sql
from loom.sandbox.subprocess_executor import SubprocessExecutor


def _req(db: Path, code: str, language: str = "sql", **kw: object) -> SandboxRequest:
    return SandboxRequest(language=language, code=code, duckdb_path=str(db), **kw)  # type: ignore[arg-type]


# SQL happy path returns columns and JSON rows through a real subprocess.
def test_sql_happy_path(tiny_db: Path) -> None:
    res = SubprocessExecutor().run(
        _req(tiny_db, "SELECT province, sum(amount) AS total FROM demo_sales GROUP BY 1 ORDER BY 1")
    )
    assert res.ok, res.error
    assert res.columns == ["province", "total"]
    assert res.rows == [["BC", 60.0], ["ON", 250.0], ["QC", 200.0]]
    assert res.backend == "subprocess"


# Row limit trims the output and sets the truncated flag.
def test_row_limit_truncation(tiny_db: Path) -> None:
    res = execute(_req(tiny_db, "SELECT * FROM demo_sales ORDER BY sale_id", row_limit=2))
    assert res.ok
    assert res.row_count == 2
    assert res.truncated is True


# Under the limit: no truncation flag.
def test_row_limit_not_truncated(tiny_db: Path) -> None:
    res = execute(_req(tiny_db, "SELECT * FROM demo_sales", row_limit=100))
    assert res.ok and res.row_count == 5 and res.truncated is False


# Writes, file exports and stacked statements are refused before execution.
@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE demo_sales",
        "COPY demo_sales TO '/tmp/x.csv'",
        "SELECT 1; DROP TABLE demo_sales",
        "DELETE FROM demo_sales",
        "-- sneaky\nCREATE TABLE t AS SELECT 1",
        "SELECT * FROM read_csv_auto('/etc/passwd')",
        "PRAGMA database_list",
        "",
    ],
)
def test_rejects_non_readonly(tiny_db: Path, sql: str) -> None:
    res = execute(_req(tiny_db, sql))
    assert res.ok is False
    assert res.error is not None and res.error.startswith("rejected:")


# Data is untouched after a rejected statement (the connection is read-only anyway).
def test_state_unchanged_after_rejection(tiny_db: Path) -> None:
    execute(_req(tiny_db, "DROP TABLE demo_sales"))
    res = execute(_req(tiny_db, "SELECT count(*) FROM demo_sales"))
    assert res.ok and res.rows == [[5]]


# CTEs are legitimate read-only SQL.
def test_with_cte_allowed() -> None:
    assert validate_sql("WITH t AS (SELECT 1 AS x) SELECT * FROM t;").startswith("WITH")


# A runaway SQL query is killed at the wall-clock limit.
def test_sql_timeout(tiny_db: Path) -> None:
    res = SubprocessExecutor().run(
        _req(
            tiny_db,
            "SELECT count(*) FROM range(10000000000) a, range(1000) b",
            timeout_s=1,
        )
    )
    assert res.ok is False
    assert res.error == "timeout after 1s"


# A runaway pandas loop is killed at the wall-clock limit.
def test_pandas_timeout(tiny_db: Path) -> None:
    res = SubprocessExecutor().run(
        _req(tiny_db, "while True:\n    pass", language="pandas", timeout_s=1)
    )
    assert res.ok is False and res.error == "timeout after 1s"


# Pandas happy path: tables are pre-loaded as DataFrames named after the table.
def test_pandas_happy_path(tiny_db: Path) -> None:
    res = SubprocessExecutor().run(
        _req(
            tiny_db,
            "result = demo_sales.groupby('province', as_index=False)['amount'].sum()",
            language="pandas",
            tables=["demo_sales"],
        )
    )
    assert res.ok, res.error
    assert res.columns == ["province", "amount"]
    assert sorted(res.rows) == [["BC", 60.0], ["ON", 250.0], ["QC", 200.0]]


# Dangerous builtins are absent from the pandas namespace.
@pytest.mark.parametrize(
    "code",
    [
        "result = open('/etc/passwd').read()",
        "import os\nresult = demo_sales",
        "result = __import__('os').listdir('/')",
        "result = eval('1+1')",
    ],
)
def test_pandas_blocked_builtins(tiny_db: Path, code: str) -> None:
    res = execute(_req(tiny_db, code, language="pandas", tables=["demo_sales"]))
    assert res.ok is False
    assert res.error is not None


# Forgetting to assign `result` is a clean, explainable failure.
def test_pandas_missing_result(tiny_db: Path) -> None:
    res = execute(_req(tiny_db, "x = demo_sales.head()", language="pandas", tables=["demo_sales"]))
    assert res.ok is False
    assert "result" in (res.error or "")


# A Series result is promoted to a DataFrame instead of failing.
def test_pandas_series_result(tiny_db: Path) -> None:
    res = execute(
        _req(
            tiny_db,
            "result = demo_sales.groupby('province')['amount'].sum()",
            language="pandas",
            tables=["demo_sales"],
        )
    )
    assert res.ok and res.columns == ["province", "amount"]


# Pandas row limit applies too.
def test_pandas_row_limit(tiny_db: Path) -> None:
    res = execute(
        _req(tiny_db, "result = demo_sales", language="pandas", tables=["demo_sales"], row_limit=3)
    )
    assert res.ok and res.row_count == 3 and res.truncated


# Dates, decimals, NaN and numpy scalars all serialise to JSON.
def test_json_safety(tiny_db: Path) -> None:
    res = execute(
        _req(
            tiny_db,
            "SELECT sale_date, CAST(amount AS DECIMAL(10,2)) AS d, NULL AS n, "
            "'nan'::DOUBLE AS f, TIMESTAMP '2024-01-01 10:00:00' AS ts FROM demo_sales LIMIT 1",
        )
    )
    assert res.ok, res.error
    row = res.rows[0]
    assert row[0] == "2024-01-05"
    assert isinstance(row[1], float)
    assert row[2] is None and row[3] is None
    assert row[4].startswith("2024-01-01T10:00:00")
    json.dumps(res.model_dump())
    import decimal

    assert to_json_safe(decimal.Decimal("1.5")) == 1.5
    assert to_json_safe(float("inf")) is None


# Bad SQL comes back as an error message the agent can use to self-correct.
def test_sql_error_message(tiny_db: Path) -> None:
    res = execute(_req(tiny_db, "SELECT nope FROM demo_sales"))
    assert res.ok is False
    assert "nope" in (res.error or "")


# Docker backend builds the isolation flags and rewrites the DB path for the container.
def test_docker_argv_and_stdin(tiny_db: Path) -> None:
    ex = DockerExecutor(image="loom-sandbox:test", docker_bin="/usr/bin/docker")
    fake = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=SandboxResult(ok=True, rows=[[1]], row_count=1).model_dump_json(),
        stderr="",
    )
    with mock.patch("loom.sandbox.docker_executor.subprocess.run", return_value=fake) as run:
        res = ex.run(_req(tiny_db, "SELECT 1"))
    argv = run.call_args.args[0]
    assert argv[:3] == ["/usr/bin/docker", "run", "--rm"]
    assert "--network" in argv and argv[argv.index("--network") + 1] == "none"
    assert "--read-only" in argv and "--pids-limit" in argv
    assert f"{tiny_db}:{CONTAINER_DB_PATH}:ro" in argv
    assert argv[-3:] == ["python", "-m", "loom.sandbox.runner"]
    sent = json.loads(run.call_args.kwargs["input"])
    assert sent["duckdb_path"] == CONTAINER_DB_PATH
    assert res.ok and res.backend == "docker"


# Docker timeout is reported the same way as the subprocess backend.
def test_docker_timeout(tiny_db: Path) -> None:
    ex = DockerExecutor(image="x", docker_bin="docker")
    with mock.patch(
        "loom.sandbox.docker_executor.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="docker", timeout=5),
    ):
        res = ex.run(_req(tiny_db, "SELECT 1", timeout_s=5))
    assert res.ok is False and res.error == "timeout after 5s"


# Lambda backend round-trips the request payload and parses the result payload.
def test_lambda_stub_roundtrip(tiny_db: Path) -> None:
    import boto3

    client = boto3.client("lambda", region_name="us-east-1")
    stub = Stubber(client)
    request = _req("s3://bucket/loom.duckdb", "SELECT 1")
    expected = SandboxResult(ok=True, columns=["x"], rows=[[1]], row_count=1)
    stub.add_response(
        "invoke",
        {"StatusCode": 200, "Payload": io.BytesIO(expected.model_dump_json().encode())},
        {
            "FunctionName": "loom-executor",
            "InvocationType": "RequestResponse",
            "Payload": request.model_dump_json().encode("utf-8"),
        },
    )
    with stub:
        res = LambdaExecutor("loom-executor", "us-east-1", client=client).run(request)
    assert res.ok and res.rows == [[1]] and res.backend == "lambda"


# A Lambda-side crash becomes a clean error result.
def test_lambda_function_error() -> None:
    import boto3

    client = boto3.client("lambda", region_name="us-east-1")
    stub = Stubber(client)
    stub.add_response(
        "invoke",
        {
            "StatusCode": 200,
            "FunctionError": "Unhandled",
            "Payload": io.BytesIO(b'{"errorMessage": "boom"}'),
        },
    )
    with stub:
        res = LambdaExecutor("f", "us-east-1", client=client).run(_req("s3://b/k", "SELECT 1"))
    assert res.ok is False and "boom" in (res.error or "")


# The Lambda handler runs the shared runner on a local path without touching S3.
def test_lambda_handler_local_path(tiny_db: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("lambda_handler", "infra/lambda/handler.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = module.handler(_req(tiny_db, "SELECT count(*) AS n FROM demo_sales").model_dump(), None)
    assert out["ok"] and out["rows"] == [[5]] and out["backend"] == "lambda"


# Factory maps each setting to its backend class.
def test_factory(settings: Settings) -> None:
    assert isinstance(get_executor(settings), SubprocessExecutor)
    settings.sandbox_backend = SandboxBackend.DOCKER
    assert isinstance(get_executor(settings), DockerExecutor)
    settings.sandbox_backend = SandboxBackend.LAMBDA
    with mock.patch("boto3.client"):
        assert isinstance(get_executor(settings), LambdaExecutor)


# Real moto Lambda invocation needs Docker; kept as an integration placeholder.
@pytest.mark.integration
@pytest.mark.skip(reason="moto Lambda invoke requires Docker")
def test_moto_lambda_invoke() -> None:
    pass
