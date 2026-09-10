"""Pick the sandbox backend from settings.

Strategy pattern: every backend implements SandboxExecutor.run(request), so the
executor agent never knows whether code ran in a subprocess, a Docker container
or an AWS Lambda. Switching is LOOM_SANDBOX_BACKEND=lambda, no code change.
"""

from __future__ import annotations

from loom.config import SandboxBackend, Settings, get_settings
from loom.sandbox.base import SandboxExecutor


def get_executor(settings: Settings | None = None) -> SandboxExecutor:
    settings = settings or get_settings()
    if settings.sandbox_backend == SandboxBackend.DOCKER:
        from loom.sandbox.docker_executor import DockerExecutor

        return DockerExecutor(settings.sandbox_docker_image)
    if settings.sandbox_backend == SandboxBackend.LAMBDA:
        from loom.sandbox.lambda_executor import LambdaExecutor

        return LambdaExecutor(settings.lambda_function_name, settings.aws_region)
    from loom.sandbox.subprocess_executor import SubprocessExecutor

    return SubprocessExecutor()
