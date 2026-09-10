# ADR 0004: Sandboxed execution with swappable backends (Strategy)

Status: Accepted. Date: 2026-09-10.

## Context

The executor and verifier run model-written SQL and pandas. That code must not
be able to reach the network, write to the host, or run for ever. The
deployment target may be a container host or AWS Lambda.

## Decision

Define `SandboxExecutor.run(SandboxRequest) -> SandboxResult` with JSON-able
request and result models. Ship three strategies chosen by
`LOOM_SANDBOX_BACKEND`:

- `docker`: `docker run --rm --network none --cpus 1 --memory N --pids-limit
  64 --read-only` with the DuckDB file mounted read-only. This is the
  production isolation.
- `subprocess`: a child Python process with a wall-clock timeout, a minimal
  environment and (on Linux) a memory rlimit; a convenience for laptops and CI.
- `lambda`: `boto3` invoke of a function whose handler runs the same runner
  after fetching the DuckDB file from S3.

Inside every backend the same `runner` enforces a read-only SQL allow-list
(single statement, no DDL / DML / COPY / ATTACH / INSTALL / LOAD / PRAGMA /
SET), a row limit with truncation flag, and a restricted namespace for pandas.

## Consequences

- Swapping in Lambda is a config change plus the Terraform module; callers do
  not change.
- The subprocess backend is not a security boundary on macOS; documentation
  says so (see `docs/security.md`).
- The runner is dependency-light so the sandbox image stays small.
