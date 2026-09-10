"""Docker backend: the real isolation boundary.

Approach: same runner, but inside a container with no network, a CPU and
memory cap, a pid limit, a read-only root filesystem and the DuckDB file
mounted read-only. The request is piped over stdin so no files are written.
"""

from __future__ import annotations

import shutil
import subprocess
import time

from loom.sandbox.base import SandboxExecutor, SandboxRequest, SandboxResult

CONTAINER_DB_PATH = "/data/loom.duckdb"


class DockerExecutor(SandboxExecutor):
    name = "docker"

    def __init__(self, image: str, cpus: float = 1.0, docker_bin: str | None = None) -> None:
        self.image = image
        self.cpus = cpus
        self.docker_bin = docker_bin or shutil.which("docker") or "docker"

    def argv(self, request: SandboxRequest) -> list[str]:
        """Build the docker command; separate so tests can assert the flags."""
        return [
            self.docker_bin,
            "run",
            "--rm",
            "-i",
            "--network",
            "none",
            "--cpus",
            str(self.cpus),
            "--memory",
            f"{request.memory_mb}m",
            "--pids-limit",
            "64",
            "--read-only",
            "--tmpfs",
            "/tmp:size=64m",
            "-v",
            f"{request.duckdb_path}:{CONTAINER_DB_PATH}:ro",
            self.image,
            "python",
            "-m",
            "loom.sandbox.runner",
        ]

    def run(self, request: SandboxRequest) -> SandboxResult:
        start = time.perf_counter()
        # Inside the container the database lives at the mount point, not the host path.
        inner = request.model_copy(update={"duckdb_path": CONTAINER_DB_PATH})
        try:
            proc = subprocess.run(
                self.argv(request),
                input=inner.model_dump_json(),
                capture_output=True,
                text=True,
                # Small grace period for container start-up on top of the runner's own budget.
                timeout=request.timeout_s + 5,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                ok=False,
                error=f"timeout after {request.timeout_s}s",
                elapsed_s=time.perf_counter() - start,
                backend=self.name,
            )
        if not proc.stdout.strip():
            return SandboxResult(
                ok=False,
                error=f"container exited {proc.returncode} without output: {proc.stderr[-1000:]}",
                elapsed_s=time.perf_counter() - start,
                backend=self.name,
            )
        result = SandboxResult.model_validate_json(proc.stdout)
        result.backend = self.name
        return result
