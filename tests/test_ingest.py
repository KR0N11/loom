"""ingest() wiring with download/clean monkeypatched (no network)."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from loom.config import Settings
from loom.data import ingest as ingest_mod
from loom.data.adapters.toronto_ttc import TorontoTTCAdapter
from loom.semantic.models import DatasetSemantics


@pytest.fixture
def offline_ttc(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_download(
        self: TorontoTTCAdapter, raw_dir: Path, limit: int | None = None
    ) -> dict[str, Path]:
        return {"subway_delays__2024": Path("x"), "subway_delay_codes__all": Path("y")}

    def fake_clean(self: TorontoTTCAdapter, raw_files: dict[str, Path]) -> dict[str, pd.DataFrame]:
        delays = pd.DataFrame(
            {
                "delay_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]).date,
                "delay_time": ["02:00", "03:00", "04:00"],
                "day_of_week": ["Monday"] * 3,
                "station": ["A", "B", "C"],
                "code": ["MUIS", "SUDP", "MUIS"],
                "min_delay": [0, 5, 12],
                "min_gap": [0, 10, 18],
                "bound": ["N", "S", None],
                "line": ["YU"] * 3,
                "vehicle": [0, 1, 2],
                "delay_hour": [2, 3, 4],
                "mode": ["subway"] * 3,
            }
        )
        codes = pd.DataFrame({"code": ["MUIS", "SUDP"], "description": ["Injured", "Disorderly"]})
        return {
            "subway_delays": delays,
            "bus_delays": delays.head(0),
            "streetcar_delays": delays.head(0),
            "subway_delay_codes": codes,
        }

    monkeypatch.setattr(TorontoTTCAdapter, "download", fake_download)
    monkeypatch.setattr(TorontoTTCAdapter, "clean", fake_clean)


# Proves: ingest loads namespaced tables, writes the semantic JSON, and reports row counts.
def test_ingest_end_to_end(settings: Settings, offline_ttc: None, tmp_path: Path) -> None:
    settings.semantic_dir = tmp_path / "semantic"
    report = ingest_mod.ingest("ttc", settings=settings)
    assert report.rows_per_table["ttc_subway_delays"] == 3
    assert report.rows_per_table["ttc_subway_delay_codes"] == 2
    conn = duckdb.connect(str(settings.duckdb_path), read_only=True)
    assert conn.execute("SELECT COUNT(*) FROM ttc_subway_delays").fetchone()[0] == 3
    conn.close()
    sem = DatasetSemantics.model_validate_json(report.semantic_path.read_text())
    assert {t.name for t in sem.tables} >= {"ttc_subway_delays", "ttc_subway_delay_codes"}
    sub = next(t for t in sem.tables if t.name == "ttc_subway_delays")
    assert sub.time_min == "2024-01-01" and sub.time_max == "2024-01-03"
    assert len(sem.metrics) == 10


# Proves: --limit caps rows per table after cleaning.
def test_ingest_limit(settings: Settings, offline_ttc: None, tmp_path: Path) -> None:
    settings.semantic_dir = tmp_path / "semantic"
    report = ingest_mod.ingest("ttc", settings=settings, limit=2)
    assert report.rows_per_table["ttc_subway_delays"] == 2


# Proves: re-running ingest replaces tables instead of appending duplicates.
def test_ingest_is_idempotent(settings: Settings, offline_ttc: None, tmp_path: Path) -> None:
    settings.semantic_dir = tmp_path / "semantic"
    ingest_mod.ingest("ttc", settings=settings)
    ingest_mod.ingest("ttc", settings=settings)
    conn = duckdb.connect(str(settings.duckdb_path), read_only=True)
    assert conn.execute("SELECT COUNT(*) FROM ttc_subway_delays").fetchone()[0] == 3
    conn.close()


# Real network pull of a tiny NFIP slice; skipped unless integration tests are requested.
@pytest.mark.integration
def test_nfip_download_small_slice(tmp_path: Path) -> None:
    from loom.data.adapters.nfip import NFIPClaimsAdapter

    files = NFIPClaimsAdapter().download(tmp_path, limit=50)
    frames = NFIPClaimsAdapter().clean(files)
    assert 0 < len(frames["claims"]) <= 50
    assert "total_paid" in frames["claims"].columns
