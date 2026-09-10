"""AWS Lambda entrypoint for the Loom sandbox.

Approach: the event IS a SandboxRequest. If `duckdb_path` is an s3:// URI the
file is downloaded once to /tmp (cached across warm invocations), then the
same `loom.sandbox.runner.execute` used locally runs the request. Lambda's own
memory/timeout settings are the resource limits; the function has no VPC
egress so there is no network.
"""

from __future__ import annotations

import os
from typing import Any

from loom.sandbox.base import SandboxRequest
from loom.sandbox.runner import execute

_CACHE_DIR = "/tmp/loom"


def _localize(path: str) -> str:
    """Download s3://bucket/key to /tmp if needed and return the local path."""
    if not path.startswith("s3://"):
        return path
    import boto3

    bucket, _, key = path[5:].partition("/")
    local = os.path.join(_CACHE_DIR, os.path.basename(key))
    # Warm containers keep /tmp, so skip the download when the file is already there.
    if not os.path.exists(local):
        os.makedirs(_CACHE_DIR, exist_ok=True)
        boto3.client("s3").download_file(bucket, key, local)
    return local


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    request = SandboxRequest.model_validate(event)
    request.duckdb_path = _localize(request.duckdb_path)
    result = execute(request)
    result.backend = "lambda"
    return result.model_dump()
