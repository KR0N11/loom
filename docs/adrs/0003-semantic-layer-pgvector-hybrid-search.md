# ADR 0003: Semantic layer in pgvector with hybrid search and a reranker

Status: Accepted. Date: 2026-09-10.

## Context

Text-to-SQL fails most often on the wrong join, the wrong grain, or a metric
computed from the wrong column. The fix is to retrieve curated metadata
(table and column descriptions from profiling, a metrics dictionary with SQL
expressions, units, grain, and allowed joins) and put it in front of the
executor.

## Decision

Flatten tables, columns, metrics and joins into `SemanticDoc`s, build them as
LlamaIndex documents, embed them with `BAAI/bge-base-en-v1.5` and store them in
Postgres with the `vector` extension behind a `VectorRepository` interface
(Repository pattern). Retrieval is hybrid: vector top-k plus keyword top-k
(Postgres full-text search, or a pure-Python BM25 in the fallback), fused with
Reciprocal Rank Fusion, then reranked by `BAAI/bge-reranker-base`. The context
builder always includes the full column list of every touched table and all
joins and metrics that reference those tables. A `DuckDBVectorRepository`
implements the same interface for laptops and CI without Postgres.

## Consequences

- Exact column names are found by keyword search even when embeddings miss them.
- Retrieval precision and recall are measurable against expected tables and
  metrics per benchmark question.
- Two repository implementations must stay behaviourally aligned; the
  repository tests run against the DuckDB backend and are marked
  `integration` for pgvector.
