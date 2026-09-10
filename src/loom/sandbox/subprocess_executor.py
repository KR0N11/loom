"""Local subprocess backend.

Approach: run `loom.sandbox.runner` in a fresh interpreter with a wall-clock
timeout and a minimal environment. This gives process isolation and time
limits on any laptop; it does NOT give a real network or filesystem boundary.
On Linux `unshare -n` (if available) drops the network namespace. For true
isolation use the Docker backend; this one exists so tests and dev never need
Docker.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

from loom.sandbox.base import SandboxExecutor, SandboxRequest, SandboxResult


class SubprocessExecutor(SandboxExecutor):
    name = "subprocess"

    def __init__(self, python: str | None = None) -> None:
        self.python = python or sys.executable

    def _argv(self) -> list[str]:
        argv = [self.python, "-m", "loom.sandbox.runner"]
        # Linux only: drop the network namespace so the child has no sockets at all.
        unshare = shutil.which("unshare") if sys.platform == "linux" else None
        if unshare:
            argv = [unshare, "-n", "--", *argv]
        return argv

    def _env(self) -> dict[str, str]:
        # Minimal environment: no credentials, no proxy variables, but keep PATH/PYTHONPATH
        # so the interpreter can import loom.sandbox.runner.
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV", ""),
            "HOME": os.environ.get("HOME", "/tmp"),
            "no_proxy": "*",
            "NO_PROXY": "*",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        return {k: v for k, v in env.items() if v}

    def run(self, request: SandboxRequest) -> SandboxResult:
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                self._argv(),
                input=request.model_dump_json(),
                capture_output=True,
                text=True,
                timeout=request.timeout_s,
                env=self._env(),
                check=False,
            )
        except subprocess.TimeoutExpired:
            # subprocess.run kills the child on timeout; report it as a clean failure.
            return SandboxResult(
                ok=False,
                error=f"timeout after {request.timeout_s}s",
                elapsed_s=time.perf_counter() - start,
                backend=self.name,
            )
        # A crash before the runner wrote JSON (e.g. OOM kill) has no stdout to parse.
        if not proc.stdout.strip():
            return SandboxResult(
                ok=False,
                error=f"sandbox exited {proc.returncode} without output: {proc.stderr[-1000:]}",
                elapsed_s=time.perf_counter() - start,
                backend=self.name,
            )
        result = SandboxResult.model_validate_json(proc.stdout)
        result.backend = self.name
        return result
