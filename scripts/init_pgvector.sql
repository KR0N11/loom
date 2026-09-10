-- Schema for the Loom semantic layer on Postgres + pgvector.
-- Applied automatically by PgVectorRepository; kept here for Terraform/RDS bootstrap.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS semantic_docs (
    doc_id    TEXT PRIMARY KEY,
    dataset   TEXT NOT NULL,
    kind      TEXT NOT NULL,          -- table | column | metric | join
    name      TEXT NOT NULL,
    text      TEXT NOT NULL,
    metadata  JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(768),            -- BAAI/bge-base-en-v1.5
    tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', name || ' ' || text)) STORED
);

CREATE INDEX IF NOT EXISTS semantic_docs_dataset_idx ON semantic_docs (dataset);
CREATE INDEX IF NOT EXISTS semantic_docs_tsv_idx ON semantic_docs USING GIN (tsv);
-- HNSW index for approximate cosine search once the corpus is large enough to matter.
CREATE INDEX IF NOT EXISTS semantic_docs_embedding_idx
    ON semantic_docs USING hnsw (embedding vector_cosine_ops);
