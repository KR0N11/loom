# ADR 0001: DuckDB as the analytical engine

Status: Accepted. Date: 2026-09-10.

## Context

Loom runs LLM-generated analytical queries over two public datasets of a few
hundred thousand rows each. The engine must run embedded (no server to
operate), be fast for aggregations, and be readable from a locked-down
sandbox and from an AWS Lambda.

## Decision

Use DuckDB as the only analytical store. Each dataset adapter writes its tables
as `<dataset>_<table>` into one DuckDB file. The sandbox opens the file
read-only; the Lambda executor downloads the same file from S3 to `/tmp`.
Generated SQL targets the DuckDB dialect; pandas is an alternative execution
language for the same data.

## Consequences

- No database server to provision or secure for the analytical path.
- One file is trivially copied into a `--network none` container or a Lambda.
- Vector search is not DuckDB's job; the semantic layer uses pgvector (ADR 0003),
  with a DuckDB-backed fallback only for local development.
- Very large datasets (the full NFIP claims table) are out of scope for a
  single-file setup; adapters take a bounded slice.
