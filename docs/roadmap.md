# Loom roadmap and status report

Reporting date: 2026-09-10. Format follows a Jira epic / story status report.
Estimates are in engineering hours. "Actual" is recorded only where it was
measured; everything in this pass was built in one initial build session, so
unmeasured actuals are marked "initial pass; TBD" rather than guessed.

## Summary

| Epic | Status | Estimate (h) | Actual (h) | Blockers |
|---|---|---|---|---|
| Phase 1: Data ingestion, semantic layer, benchmark skeleton | Done (local backends), pgvector unverified | 24 | initial pass; TBD | No Postgres on the build machine |
| Phase 2: Planner, executor, sandbox | Done (subprocess), Docker/Lambda mocked | 20 | initial pass; TBD | No Docker on the build machine |
| Phase 3: Verifier and injection mode | Done | 12 | initial pass; TBD | Live catch-rate run blocked (see risks) |
| Phase 4: Narrator, API, UI | Done | 14 | initial pass; TBD | Screenshots pending a live run |
| Phase 5: Tracing, small-model comparison, CI gate, Terraform | Done (code), untested against AWS | 18 | initial pass; TBD | Terraform not applied; `ml` extra models not downloaded |

## Phase 1: Data ingestion + semantic layer + benchmark skeleton

| Story | Status | Est. | Actual | Notes |
|---|---|---|---|---|
| LOOM-101 Adapter interface (download / clean / load / describe) | Done | 2 | initial pass; TBD | `loom.data.adapters.base` |
| LOOM-102 Toronto TTC adapter (subway, bus, streetcar, delay codes) | Done | 5 | initial pass; TBD | CKAN package resources, recent years |
| LOOM-103 OpenFEMA NFIP adapter (claims slice, communities, states) | Done | 5 | initial pass; TBD | Bounded slice by loss year and state |
| LOOM-104 Profiling to `DatasetSemantics` JSON | Done | 3 | initial pass; TBD | Null fraction, distinct, min/max, samples |
| LOOM-105 Curated metric dictionary and allowed joins | Done | 3 | initial pass; TBD | 8 to 12 metrics per dataset |
| LOOM-106 Vector repository (pgvector) + DuckDB fallback | Done / pgvector unverified | 3 | initial pass; TBD | pgvector tests marked integration |
| LOOM-107 Hybrid search (RRF) + bge reranker | Done | 2 | initial pass; TBD | |
| LOOM-108 Benchmark schema, template generator, YAML | Done | 1 | initial pass; TBD | LLM generator built, not run |

## Phase 2: Planner + executor + sandbox

| Story | Status | Est. | Actual | Notes |
|---|---|---|---|---|
| LOOM-201 Agent contracts and typed state | Done | 2 | initial pass; TBD | ADR 0008 |
| LOOM-202 Planner agent + analysis-type strategies | Done | 4 | initial pass; TBD | |
| LOOM-203 Sandbox runner (SQL allow-list, pandas namespace, row limit) | Done | 4 | initial pass; TBD | |
| LOOM-204 Subprocess executor | Done | 2 | initial pass; TBD | Not a hard boundary on macOS |
| LOOM-205 Docker executor + image | Done (mocked) | 3 | initial pass; TBD | Blocked: Docker absent on build machine |
| LOOM-206 Lambda executor + handler | Done (stubbed client) | 3 | initial pass; TBD | |
| LOOM-207 Executor agent with self-correction retries | Done | 2 | initial pass; TBD | |

## Phase 3: Verifier + injection mode

| Story | Status | Est. | Actual | Notes |
|---|---|---|---|---|
| LOOM-301 Alternate-path re-derivation with tolerance | Done | 4 | initial pass; TBD | |
| LOOM-302 Deterministic checks (unit, sample, time, null, duplicate, truncation) | Done | 4 | initial pass; TBD | |
| LOOM-303 Injection modes and catch-rate runner | Done | 3 | initial pass; TBD | |
| LOOM-304 Live catch-rate measurement | Blocked | 1 | – | Invalid `ANTHROPIC_API_KEY` in the build environment |

## Phase 4: Narrator + UI

| Story | Status | Est. | Actual | Notes |
|---|---|---|---|---|
| LOOM-401 Narrator memo with UNVERIFIED enforcement | Done | 3 | initial pass; TBD | |
| LOOM-402 Charts (matplotlib + plotly) | Done | 2 | initial pass; TBD | |
| LOOM-403 Audit appendix | Done | 1 | initial pass; TBD | |
| LOOM-404 FastAPI backend | Done | 3 | initial pass; TBD | |
| LOOM-405 Streamlit UI | Done | 3 | initial pass; TBD | |
| LOOM-406 CLI | Done | 2 | initial pass; TBD | |
| LOOM-407 Screenshots | Blocked | 0.5 | – | Needs a live run |

## Phase 5: Tracing, small-model comparison, CI gate, Terraform

| Story | Status | Est. | Actual | Notes |
|---|---|---|---|---|
| LOOM-501 Langfuse tracing per node / generation with prompt version and cost | Done (no-op without keys) | 3 | initial pass; TBD | |
| LOOM-502 Zero-shot router baseline (bart-large-mnli) | Done (code) | 2 | initial pass; TBD | `ml` extra |
| LOOM-503 Small SQL model baselines (Qwen coder / sqlcoder) | Done (code) | 4 | initial pass; TBD | `ml` extra, not run |
| LOOM-504 TAPAS lookup baseline | Done (code) | 2 | initial pass; TBD | `ml` extra, not run |
| LOOM-505 Eval report + comparison table | Done | 2 | initial pass; TBD | |
| LOOM-506 CI: ruff, mypy, tests, eval gate | Done | 2 | initial pass; TBD | Gate runs only with an API key secret |
| LOOM-507 Terraform (S3, Lambda, RDS pgvector, secrets) | Done (not applied) | 3 | initial pass; TBD | Untested against a real account |

## Blockers

1. Live benchmark run blocked: the `ANTHROPIC_API_KEY` present in the build
   environment was rejected by the API (401). All LLM paths are covered by
   tests with a scripted provider; live numbers are pending.
2. Docker and Postgres were not available on the build machine. The Docker
   sandbox and the pgvector repository are validated by unit tests with mocks
   only; integration tests are marked and skipped.
3. Terraform was not installed on the build machine; the module has not been
   planned or applied.

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Unreviewed benchmark items contain wrong gold answers | Accuracy numbers are noisy | `reviewed` flag; report states reviewed count; review a sample first |
| Subprocess sandbox treated as production isolation | Model-written code escapes | Documented as dev-only; Docker / Lambda are the deployed backends |
| Public API schema drift (OpenFEMA, CKAN) | Ingestion breaks | Adapters verify columns at download; ingestion tests use fixtures |
| LLM cost of the full benchmark | Budget | CI uses a stratified sample; full run is manual |
| Pricing table goes stale | Cost metric off | Single table in `loom.llm.pricing` |
