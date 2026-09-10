# Loom

Loom is a multi-agent data-to-decision platform. A business user asks a
natural-language question over messy tabular data and Loom plans, executes,
verifies and explains the analysis, producing a short recommendation memo with
charts and an audit appendix listing every query that ran.

It is built to the bar an analytics team in a regulated company would need:
every agent has a strict output contract, every number is re-derived a second
way before it reaches the memo, every query is kept for audit, and a benchmark
with a CI regression gate guards against silent quality drops.

## The five agents

| Agent | Receives | Emits | Job |
|---|---|---|---|
| Planner | question, dataset table summary | `AnalysisPlan` | Classifies the question (lookup, trend, comparison, driver) and decomposes it into steps. |
| Semantic layer | question + plan | `SemanticContext` | Hybrid search (vector + keyword, reranked) over table, column, metric and join documents so queries use the right joins, grain and units. |
| Executor | plan + semantic context | `ExecutionResult` per step | Writes DuckDB SQL or pandas, runs it in a sandbox with row and time limits, self-corrects on errors up to N retries. |
| Verifier | executions + plan + context | `VerificationReport` | Re-derives each key number by a different query path, then checks units, sample sizes, time ranges, nulls, duplicates and truncation. |
| Narrator | plan + executions + verification | `Memo` | Writes the recommendation memo, renders charts, attaches the audit appendix. A failed verification forces an `UNVERIFIED:` headline. |

A LangGraph supervisor owns the routing between agents and checkpoints the
typed Pydantic state after every node.

## Architecture

```mermaid
graph LR
    subgraph Control plane
        S[Supervisor<br/>LangGraph StateGraph<br/>typed LoomState + checkpoints]
        P[Planner agent]
        SL[Semantic-layer agent<br/>hybrid search + bge-reranker]
        E[Executor agent<br/>SQL / pandas, N retries]
        V[Verifier agent<br/>re-derive, unit / sample / time / null / dup checks]
        N[Narrator agent<br/>memo + charts + audit appendix]
        S --> P --> SL --> E --> V --> N
    end

    subgraph Execution
        SB{Sandbox<br/>Strategy}
        D1[Docker<br/>no network, CPU / mem / time limits]
        D2[Subprocess<br/>local fallback]
        D3[AWS Lambda<br/>swap-in]
        SB --> D1
        SB --> D2
        SB --> D3
    end
    E --> SB
    V --> SB

    subgraph Data plane
        A1[Adapter: Toronto TTC delays]
        A2[Adapter: OpenFEMA NFIP claims]
        DDB[(DuckDB<br/>dataset_table)]
        PR[Profiling]
        SJ[Semantic JSON<br/>tables, columns, metrics, joins]
        LI[LlamaIndex documents]
        PG[(pgvector on Postgres<br/>DuckDB local fallback)]
        A1 --> DDB
        A2 --> DDB
        DDB --> PR --> SJ --> LI --> PG
    end
    SB --> DDB
    SL --> PG

    LLM[LLM provider<br/>Anthropic API default<br/>Amazon Bedrock via config]
    P -.-> LLM
    SL -.-> LLM
    E -.-> LLM
    V -.-> LLM
    N -.-> LLM
    LF[Langfuse tracing<br/>node, tool call, prompt version, cost]
    S -.-> LF
```

## Quickstart

```bash
# 1. Install (Python 3.12, uv)
uv sync --all-groups

# 2. Configure
cp .env.example .env            # set ANTHROPIC_API_KEY (or the Bedrock switch)

# 3. Load both public datasets into DuckDB and build the semantic layer
uv run loom ingest --dataset all
uv run loom build-semantic --dataset all

# 4. Ask a question from the CLI
uv run loom ask "Which flood zone drove the increase in total paid claims from 2018 to 2020?" --dataset nfip
uv run loom ask "Which subway line had the most delay minutes in 2024?" --dataset ttc

# 5. API and UI
uv run loom serve                # FastAPI on http://localhost:8000
uv run loom ui                   # Streamlit front end

# 6. Tests and benchmark
uv run pytest -q
uv run loom benchmark --sample 40 --injection all
```

The CLI exits with code 2 when the verifier fails, so a scripted caller cannot
mistake an unverified memo for a verified one.

## Configuration

All settings are read from `.env` (see `.env.example`). Unless noted the prefix
is `LOOM_`.

| Variable | Default | Meaning |
|---|---|---|
| `LOOM_LLM_PROVIDER` | `anthropic` | `anthropic` or `bedrock`. |
| `ANTHROPIC_API_KEY` | – | Anthropic API key (no prefix). |
| `LOOM_LLM_MODEL` | `claude-sonnet-5` | Model id for the Anthropic provider. |
| `AWS_REGION` | `us-east-1` | Region for Bedrock and Lambda (no prefix). |
| `LOOM_BEDROCK_MODEL_ID` | `anthropic.claude-sonnet-5-v1:0` | Bedrock model id. |
| `LOOM_DUCKDB_PATH` | `data/duckdb/loom.duckdb` | Analytical database file. |
| `LOOM_VECTOR_BACKEND` | `duckdb` | `pgvector` (production) or `duckdb` (local fallback). |
| `LOOM_PG_DSN` | `postgresql://loom:loom@localhost:5432/loom` | Postgres with the `vector` extension. |
| `LOOM_CHECKPOINT_PATH` | `data/checkpoints/loom.sqlite` | LangGraph checkpoint store. |
| `LOOM_SANDBOX_BACKEND` | `subprocess` | `docker`, `subprocess` or `lambda`. |
| `LOOM_SANDBOX_TIMEOUT_S` | `30` | Per-query wall-clock limit. |
| `LOOM_SANDBOX_MEMORY_MB` | `1024` | Memory limit for the sandbox. |
| `LOOM_SANDBOX_ROW_LIMIT` | `10000` | Maximum rows returned per query. |
| `LOOM_LAMBDA_FUNCTION_NAME` | `loom-executor` | Lambda executor function name. |
| `LOOM_EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Embedding model. |
| `LOOM_RERANKER_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder reranker. |
| `LOOM_EMBEDDING_BACKEND` | `hf` | `hf` or `hashing` (offline test fallback). |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` | – | Tracing; when absent tracing is a no-op. |

## Datasets

Two real public datasets ship with the project so it is clearly domain-agnostic.
Each is loaded by one adapter into DuckDB as `<dataset>_<table>`.

| Dataset id | Source | Tables |
|---|---|---|
| `ttc` | Open Data Toronto: TTC subway, bus and streetcar delay data | `ttc_subway_delays`, `ttc_bus_delays`, `ttc_streetcar_delays`, `ttc_subway_delay_codes` |
| `nfip` | OpenFEMA: NFIP claims (bounded slice) and community status book | `nfip_claims`, `nfip_communities`, `nfip_states` |

The data dictionary generated from profiling plus hand-written metric
definitions is in [`docs/data_dictionary.md`](docs/data_dictionary.md).

## Benchmark results

Numbers below are produced by `scripts/run_benchmark.py` and written into this
block by the orchestrating run. "pending" means the live run has not happened
yet for this commit.

<!-- BENCHMARK_RESULTS_START -->
**Status: benchmark built (172 questions, gold answers executed against the real DuckDB), live run pending.**
The live run needs a valid `ANTHROPIC_API_KEY`; the key present at build time was rejected by the API (401), so no
model-generated numbers are reported here. Nothing in this table is estimated.

| Difficulty | n | Execution accuracy | Numeric match | Retrieval P / R | Verifier catch rate | Zero-correction % | Cost / question | p50 latency |
|---|---|---|---|---|---|---|---|---|
| easy | 62 | pending | pending | pending | pending | pending | pending | pending |
| medium | 61 | pending | pending | pending | pending | pending | pending | pending |
| hard | 49 | pending | pending | pending | pending | pending | pending | pending |
| all | 172 | pending | pending | pending | pending | pending | pending | pending |

Benchmark composition: ttc 67 / nfip 105; lookup 62, trend 25, comparison 36, driver 49; all items generated by
the template generator and marked `reviewed: false` (add LLM candidates with `scripts/generate_benchmark.py --n-llm 30`).

To produce the numbers: `uv run python scripts/run_benchmark.py --sample 40 --injection all`, then paste
`outputs/eval/<run>/report.md` here and copy `results.json` to `benchmarks/baseline_results.json` to seed the CI gate.

Small-model vs LLM comparison: pending (`scripts/run_benchmark.py --baselines qwen,tapas,router`).
<!-- BENCHMARK_RESULTS_END -->

## Example memo

<!-- EXAMPLE_MEMO_START -->
The memo below was produced by the full pipeline on the real TTC data (planner -> semantic retrieval with bge
embeddings -> sandboxed DuckDB SQL -> independent re-derivation -> chart -> audit appendix). Because the live LLM
key was invalid at build time, the LLM turns (plan, SQL, narrative text) came from a scripted stand-in provider;
every query, number, verification check and chart is real output of the system. Full file: `docs/example_memo.md`.

# TTC subway delays by line, 2024

**Sheppard (SHP) has the longest average delay in 2024 at 9.2 minutes**

## Findings
- Among lines with at least 100 delay records, SHP averages 9.2 minutes per non-zero delay.
- Only delays with a recorded duration above zero are included.

## Recommendation

Review the delay codes on the Sheppard line to identify the incident types driving the longer average.

## Caveats
- Lines with fewer than 100 records were excluded to avoid single-incident outliers.
- sample_size_1: average/rate computed over only 3 rows

## Verification

WARN: sample_size_1: average/rate computed over only 3 rows

## Charts

![Average non-zero delay minutes by line (2024)](docs/screenshots/example_chart.png)

## Audit appendix

Every query executed, in order:

### Query 1 (attempt 1, sql, ok, 3 rows, 0.00s)

```sql
SELECT line, AVG(min_delay) AS avg_delay_minutes, COUNT(*) AS n FROM ttc_subway_delays WHERE delay_date BETWEEN DATE '2024-01-01' AND DATE '2024-12-31' AND min_delay > 0 GROUP BY line HAVING COUNT(*) >= 100 ORDER BY avg_delay_minutes DESC LIMIT 10
```

### Query 2 (attempt 1, sql, ok, 1 rows, 0.00s)

```sql
SELECT SUM(min_delay) * 1.0 / COUNT(*) AS top_line_avg_delay_minutes FROM ttc_subway_delays WHERE line = 'SHP' AND delay_date >= DATE '2024-01-01' AND delay_date < DATE '2025-01-01' AND min_delay > 0
```
<!-- EXAMPLE_MEMO_END -->

## Screenshots

Captured from the Streamlit UI (`loom ui`) reading the example run above.

| Ask a question | Read a memo | Verification report |
|---|---|---|
| ![Ask](docs/screenshots/ui_ask.jpg) | ![Memo](docs/screenshots/ui_memo.jpg) | ![Verification](docs/screenshots/ui_verification.jpg) |

## Eval harness

The benchmark lives in `benchmarks/questions.yaml`: 150 to 200 questions across
both datasets with gold SQL and gold answers, tagged `easy` (lookup), `medium`
(trend / comparison) and `hard` (multi-step driver analysis). Candidates are
generated by templates and by an LLM; every item carries `reviewed: true|false`
so human-reviewed and unreviewed questions are never mixed silently.

Metrics per question:

- Execution accuracy: predicted rows equal gold rows after normalisation.
- Numerical tolerance match: a predicted key number is within 1% of the gold.
- Semantic-layer retrieval precision and recall against expected tables and metrics.
- Verifier catch rate on deliberately injected wrong answers. Injection modes: `scale`, `sign`, `off_by_year`, `drop_rows`, `swap_unit`, `zero`. False-alarm rate is measured on uninjected runs.
- Percentage of questions with zero human correction (numeric match and verifier did not fail).
- Cost and latency per question.
- Small-model vs LLM comparison table (Qwen coder / sqlcoder for SQL, TAPAS for simple lookups, bart-large-mnli for routing), available with the `ml` extra.

The report (`report.md`) breaks everything down by difficulty, dataset and
analysis type.

CI gate rule: the `eval-gate` job fails if execution accuracy drops by more
than 5 points versus `main`, or if verifier catch rate drops at all.

## Design patterns

| Pattern | Where | Why |
|---|---|---|
| Adapter | `loom.data.adapters` | One class per data source with the same download / clean / load / describe steps; adding a dataset is one subclass plus metric definitions. |
| Strategy | `loom.agents.strategies`, `loom.sandbox`, `loom.llm` | Analysis type, sandbox backend and LLM vendor are each swapped by config without touching callers. |
| Repository | `loom.semantic.repository` | Agents talk to `VectorRepository`; pgvector in production, DuckDB locally. |
| Supervisor / Worker | `loom.agents.supervisor` | LangGraph supervisor routes between worker agents that each own one contract. |
| Factory | `loom.llm.factory`, `loom.sandbox.factory` | Pick the concrete backend from settings. |

## Repository layout

```
loom/
├── src/loom/
│   ├── config.py          # pydantic-settings, all knobs
│   ├── llm/               # provider abstraction: Anthropic, Bedrock, fake
│   ├── agents/            # planner, semantic, executor, verifier, narrator, supervisor, strategies
│   ├── semantic/          # documents, embeddings, reranker, repositories, hybrid search, ingest
│   ├── sandbox/           # runner + subprocess / Docker / Lambda executors
│   ├── data/              # adapters, profiling, DuckDB store, ingestion
│   ├── eval/              # benchmark, generators, metrics, runner, injection, report, gate, baselines
│   ├── tracing/           # Langfuse wrapper (no-op without keys)
│   ├── charts.py          # matplotlib / plotly rendering
│   ├── api/               # FastAPI
│   ├── ui/                # Streamlit
│   └── cli.py             # `loom` command
├── scripts/               # ingest, build_semantic, generate_benchmark, run_benchmark, eval_gate
├── benchmarks/            # questions.yaml, baseline_results.json
├── sandbox/               # Dockerfile for the isolated executor image
├── infra/                 # Lambda handler, Terraform
├── docs/                  # ADRs, architecture, security, roadmap, data dictionary
├── tests/
└── .github/workflows/     # ruff, mypy, tests, eval gate
```

## Limitations and honesty notes

- The NFIP claims table is a bounded slice (recent loss years, a handful of
  states, capped row count) so ingestion finishes in minutes; the full dataset
  is millions of rows. Results are about the slice, not the whole program.
- Benchmark items are marked `reviewed: false` until a human has checked them.
  The results table states how many were reviewed.
- Small-model baselines (Qwen coder, sqlcoder, TAPAS, zero-shot router) need
  the optional `ml` extra (`uv sync --extra ml`) and model downloads.
- The Docker and Lambda sandboxes and the pgvector repository are covered by
  unit tests with mocked processes and clients; integration tests are marked
  and skipped when the service is not available.
- Prices used for cost accounting are approximate list prices in
  `loom.llm.pricing`.

## License

MIT, see `LICENSE`.
