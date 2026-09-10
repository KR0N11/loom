"""AWS Lambda backend: swap-in for the Docker sandbox in the cloud path.

Approach: the request is the Lambda payload, the response is the SandboxResult
JSON. Lambda already gives a per-invocation memory/time/network boundary, so
the executor is only a transport. `duckdb_path` may be an s3:// URI that the
handler (infra/lambda/handler.py) downloads to /tmp before running.
"""

from __future__ import annotations

import json
import time
from typing import Any

from loom.sandbox.base import SandboxExecutor, SandboxRequest, SandboxResult


class LambdaExecutor(SandboxExecutor):
    name = "lambda"

    def __init__(self, function_name: str, region: str, client: Any | None = None) -> None:
        self.function_name = function_name
        # Injectable client so tests can use botocore's Stubber.
        if client is None:
            import boto3

            client = boto3.client("lambda", region_name=region)
        self._client = client

    def run(self, request: SandboxRequest) -> SandboxResult:
        start = time.perf_counter()
        try:
            resp = self._client.invoke(
                FunctionName=self.function_name,
                InvocationType="RequestResponse",
                Payload=request.model_dump_json().encode("utf-8"),
            )
        except Exception as exc:  # noqa: BLE001
            return SandboxResult(
                ok=False,
                error=f"lambda invoke failed: {exc}",
                elapsed_s=time.perf_counter() - start,
                backend=self.name,
            )
        payload = json.loads(resp["Payload"].read())
        # A function-level error (unhandled exception, timeout) arrives as an error payload.
        if resp.get("FunctionError"):
            return SandboxResult(
                ok=False,
                error=f"lambda error: {payload.get('errorMessage', payload)}",
                elapsed_s=time.perf_counter() - start,
                backend=self.name,
            )
        result = SandboxResult.model_validate(payload)
        result.backend = self.name
        return result
