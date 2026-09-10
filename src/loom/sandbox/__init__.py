"""Sandboxed query execution with swappable backends (subprocess, Docker, Lambda)."""

from loom.sandbox.base import SandboxExecutor, SandboxRequest, SandboxResult
from loom.sandbox.factory import get_executor

__all__ = ["SandboxExecutor", "SandboxRequest", "SandboxResult", "get_executor"]
