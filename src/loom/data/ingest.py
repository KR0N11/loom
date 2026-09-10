"""End-to-end ingestion: download -> clean -> load into DuckDB -> profile -> semantic JSON.

Approach: one function per dataset so the CLI, tests and CI call the same
path. The semantic JSON written here is the single input to the semantic-layer
index build, so profiling output and curated metric definitions travel together.
"""

from __future__ import annotations

import time
from pathlib import Path

from pydantic import BaseModel, Field

from loom.config import Settings, get_settings
from loom.data.adapters import get_adapter
from loom.data.duckdb_store import connect
from loom.data.profiling import profile_dataset


class IngestReport(BaseModel):
    dataset: str
    rows_per_table: dict[str, int] = Field(default_factory=dict)
    semantic_path: Path
    elapsed_s: float


def ingest(
    dataset: str,
    settings: Settings | None = None,
    limit: int | None = None,
    raw_dir: Path | None = None,
) -> IngestReport:
    settings = settings or get_settings()
    adapter = get_adapter(dataset)
    start = time.perf_counter()
    raw_files = adapter.download(raw_dir or settings.raw_dir, limit=limit)
    frames = adapter.clean(raw_files)
    # Optional cap applied after cleaning so tests and quick runs stay small.
    if limit is not None:
        frames = {t: df.head(limit) for t, df in frames.items()}
    conn = connect(settings)
    try:
        adapter.load(conn, frames)
        semantics = profile_dataset(conn, adapter)
    finally:
        conn.close()
    semantic_path = Path(settings.semantic_dir) / f"{dataset}.json"
    semantic_path.parent.mkdir(parents=True, exist_ok=True)
    semantic_path.write_text(semantics.model_dump_json(indent=2))
    return IngestReport(
        dataset=dataset,
        rows_per_table={adapter.qualified(t): len(df) for t, df in frames.items()},
        semantic_path=semantic_path,
        elapsed_s=time.perf_counter() - start,
    )
