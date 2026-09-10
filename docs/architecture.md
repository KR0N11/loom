# Architecture walk-through

This document follows one question through Loom, names what each agent
receives and emits (schemas in `src/loom/agents/schemas.py`), and explains the
verification, audit, tracing and Lambda designs.

## Request lifecycle

`run_question(question, dataset, ...)` in `loom.agents.supervisor` creates a
`LoomState` and runs the LangGraph graph. The graph is
planner → semantic → executor → verifier → narrator, with one conditional
edge: if the executor produced no `ExecutionResult`, the verifier is skipped
and the narrator writes a memo saying the analysis failed.

### 1. Planner

- Receives: `question`, `dataset`, and a table summary from
  `load_semantics(dataset)`.
- Emits: `AnalysisPlan` with `analysis_type` (`lookup`, `trend`, `comparison`,
  `driver`), ordered `PlanStep`s (each with `needs_tables` and `depends_on`),
  `metrics_needed`, `time_range_hint`, `assumptions`.
- The `AnalysisType` selects a `AnalysisStrategy` (Strategy pattern) that adds
  planner guidance, executor guidance and the verifier checks that matter for
  that type.

### 2. Semantic-layer agent

- Receives: the question and plan.
- Emits: `SemanticContext` with `tables`, `columns`, `metrics`, `joins`, the
  retrieved `RetrievedDoc`s with scores, and `context_text`, a compact prompt
  block listing every touched table with grain and columns (type, unit,
  examples), the allowed joins and the metric definitions with SQL
  expressions and caveats.
- Retrieval is hybrid (vector + keyword, RRF fusion, cross-encoder rerank)
  through the `VectorRepository` (Repository pattern).

### 3. Executor

- Receives: plan, `SemanticContext.context_text`, and summaries of earlier
  steps.
- Emits: one `ExecutionResult` per step: `language` (`sql` or `pandas`), the
  final `code`, `columns`, `rows`, `row_count`, `key_numbers` (`KeyNumber`
  name / value / unit / derivation) and every attempt as a `QueryRecord`.
- Every attempt goes to the sandbox as a `SandboxRequest`; on an error the
  model is re-prompted with the error text up to `executor_max_retries`.

### 4. Verifier

- Receives: executions, plan, semantic context. In eval mode
  `LoomState.injection` corrupts the executions first (see ADR 0005).
- Emits: `VerificationReport` with `overall` (`pass` / `warn` / `fail`),
  a list of `VerificationCheck`s, `reconciled_numbers`, `flags`, and the
  alternate queries as `QueryRecord`s.

Checks:

| Check | How | Outcome |
|---|---|---|
| Re-derivation | Alternate SQL per key number by a different aggregation path or table; compared within 1% relative tolerance | FAIL on mismatch |
| Unit | Key-number unit appears in the semantic layer units | WARN if unknown |
| Sample size | `row_count > 0`; averages, rates, shares and ratios with fewer than 30 rows | WARN |
| Time range | Trend and comparison queries must carry a date filter when a time-range hint exists | WARN |
| Nulls | Alternate query result contains nulls | WARN |
| Duplicates | Duplicate full rows in the original result | FAIL |
| Truncation | Row limit hit | WARN |

Overall is FAIL if any check fails, WARN if any warns, otherwise PASS.

### 5. Narrator

- Receives: plan, executions, verification report.
- Emits: `Memo` with `title`, `headline`, `findings`, `recommendation`,
  `caveats`, `charts` (`ChartSpec` with rendered file paths),
  `appendix_queries` and `verification_summary`.
- If verification is FAIL, code (not the prompt) forces the headline to start
  with `UNVERIFIED:` and copies the flags into the caveats.
- `memo_to_markdown` renders the memo and the audit appendix.

## Audit appendix

Every `QueryRecord` from every executor attempt (including failed ones) and
every verifier alternate query is listed with attempt number, language,
success flag, error text, row count and the code itself. Nothing that touched
the data is omitted, so a reviewer can rerun any number in the memo.

## Tracing model

`loom.tracing.Tracer` wraps Langfuse. `run_question` opens one root
observation per question. Each graph node runs inside a span; each LLM call
is recorded as a generation carrying the model, the prompt version from
`loom.agents.prompts`, input and output tokens, estimated cost and latency.
Sandbox runs and retries are events inside the owning span. Without Langfuse
keys every call is a no-op.

## Data plane

Adapters (`loom.data.adapters`) download, clean and load tables into DuckDB as
`<dataset>_<table>`. Profiling produces per-column statistics; combined with
hand-written table descriptions, metric definitions and allowed joins it is
saved as `data/semantic/<dataset>.json` (`DatasetSemantics`). The semantic
ingest turns that into LlamaIndex documents, embeds them and writes them to the
vector repository.

## Sandbox and the Lambda swap-in

`SandboxRequest` and `SandboxResult` are JSON-able Pydantic models. The same
`runner` module executes a request in a subprocess, inside a
`--network none` Docker container, or inside an AWS Lambda. For Lambda the
handler in `infra/lambda/handler.py` downloads the DuckDB file from S3 to
`/tmp` when the request path is an `s3://` URI, then calls the runner. Because
callers only see `SandboxExecutor.run`, switching backends is
`LOOM_SANDBOX_BACKEND=lambda` plus the Terraform in `infra/terraform`.
