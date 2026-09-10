"""Adapter for Toronto Transit Commission (TTC) delay data from Open Data Toronto.

Approach (Adapter pattern): three related fact tables (subway, bus, streetcar
delays; one row per delay incident) plus a subway delay-code lookup. Raw files
are yearly XLSX exports from the CKAN portal; we take 2023 and 2024 to keep the
dataset a few hundred thousand rows. Column names are normalised so the three
modes share `delay_date`, `delay_time`, `day_of_week`, `min_delay`, `min_gap`,
`bound`, `vehicle`, `mode`; subway keeps `station`/`line`/`code`, surface modes
keep `route`/`location`/`incident`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from loom.data.adapters.base import DataSourceAdapter
from loom.data.cleaning import to_date, to_int
from loom.semantic.models import JoinDoc, MetricDoc

_CKAN = "https://ckan0.cf.opendata.inter.prod-toronto.ca/dataset"
_SUBWAY_PKG = "996cfe8d-fb35-40ce-b569-698d51fc683b"
_BUS_PKG = "e271cdae-8788-4980-96ce-6a5c95bc6618"
_STREETCAR_PKG = "b68cb71b-44a7-4394-97e2-5d2f41462a5d"

# {raw key: url}; raw key is "<table>__<year>" so clean() can regroup files per table.
RESOURCES: dict[str, str] = {
    "subway_delays__2023": f"{_CKAN}/{_SUBWAY_PKG}/resource/2fbec48b-33d9-4897-a572-96c9f002d66a/download/ttc-subway-delay-2023.xlsx",
    "subway_delays__2024": f"{_CKAN}/{_SUBWAY_PKG}/resource/2ee1a65c-da06-4ad1-bdfb-b1a57701e46a/download/ttc-subway-delay-2024.xlsx",
    "bus_delays__2023": f"{_CKAN}/{_BUS_PKG}/resource/10802a64-9ac0-4f2e-9538-04800a399d1e/download/ttc-bus-delay-data-2023.xlsx",
    "bus_delays__2024": f"{_CKAN}/{_BUS_PKG}/resource/7823b829-9952-4e4c-ac8f-0fe2ef53901c/download/ttc-bus-delay-data-2024.xlsx",
    "streetcar_delays__2023": f"{_CKAN}/{_STREETCAR_PKG}/resource/472d838d-e41a-4616-a11b-585d26d59777/download/ttc-streetcar-delay-data-2023.xlsx",
    "streetcar_delays__2024": f"{_CKAN}/{_STREETCAR_PKG}/resource/5f527714-2284-437b-958b-c02b6f21eb9d/download/ttc-streetcar-delay-data-2024.xlsx",
    "subway_delay_codes__all": f"{_CKAN}/{_SUBWAY_PKG}/resource/b2d8f5e0-0997-46b5-8abd-caa685a0290b/download/code-descriptions.csv",
}

_SUBWAY_RENAME = {
    "Date": "delay_date",
    "Time": "delay_time",
    "Day": "day_of_week",
    "Station": "station",
    "Code": "code",
    "Min Delay": "min_delay",
    "Min Gap": "min_gap",
    "Bound": "bound",
    "Line": "line",
    "Vehicle": "vehicle",
}
_BUS_RENAME = {
    "Date": "delay_date",
    "Route": "route",
    "Time": "delay_time",
    "Day": "day_of_week",
    "Location": "location",
    "Incident": "incident",
    "Min Delay": "min_delay",
    "Min Gap": "min_gap",
    "Direction": "bound",
    "Vehicle": "vehicle",
}
_STREETCAR_RENAME = {
    "Date": "delay_date",
    "Line": "route",
    "Time": "delay_time",
    "Day": "day_of_week",
    "Location": "location",
    "Incident": "incident",
    "Min Delay": "min_delay",
    "Min Gap": "min_gap",
    "Bound": "bound",
    "Vehicle": "vehicle",
}


class TorontoTTCAdapter(DataSourceAdapter):
    """TTC subway / bus / streetcar delay incidents, 2023-2024."""

    name = "ttc"
    description = "Toronto Transit Commission delay incidents (subway, bus, streetcar), 2023-2024."

    def download(self, raw_dir: Path, limit: int | None = None) -> dict[str, Path]:
        target = raw_dir / self.name
        target.mkdir(parents=True, exist_ok=True)
        out: dict[str, Path] = {}
        for key, url in RESOURCES.items():
            path = target / f"{key}{Path(url).suffix}"
            # Skip files already on disk so re-running ingestion is cheap and offline-safe.
            if not path.exists() or path.stat().st_size == 0:
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
                path.write_bytes(resp.content)
            out[key] = path
        return out

    def clean(self, raw_files: dict[str, Path]) -> dict[str, pd.DataFrame]:
        grouped: dict[str, list[pd.DataFrame]] = {}
        for key, path in raw_files.items():
            table = key.split("__")[0]
            grouped.setdefault(table, []).append(_read_raw(path))
        frames: dict[str, pd.DataFrame] = {}
        for table, parts in grouped.items():
            raw = pd.concat(parts, ignore_index=True)
            if table == "subway_delay_codes":
                frames[table] = clean_codes(raw)
            else:
                frames[table] = clean_delays(raw, table)
        return frames

    def table_descriptions(self) -> dict[str, tuple[str, str]]:
        return {
            "subway_delays": (
                "Subway and SRT delay incidents logged by TTC control, 2023-2024. "
                "Each incident has a station, a delay code (see subway_delay_codes), the delay in "
                "minutes and the resulting gap between trains.",
                "one row per delay incident",
            ),
            "bus_delays": (
                "Bus delay incidents by route and location, 2023-2024. Causes are free-text "
                "incident categories (e.g. Mechanical, Operations - Operator, Collision - TTC).",
                "one row per delay incident",
            ),
            "streetcar_delays": (
                "Streetcar delay incidents by route and location, 2023-2024. Same incident "
                "categories as bus delays.",
                "one row per delay incident",
            ),
            "subway_delay_codes": (
                "Lookup of subway delay codes to plain-English descriptions "
                "(e.g. MUPAA = Passenger Assistance Alarm Activated).",
                "one row per delay code",
            ),
        }

    def column_descriptions(self) -> dict[str, dict[str, tuple[str, str | None]]]:
        common: dict[str, tuple[str, str | None]] = {
            "delay_date": ("Calendar date of the incident.", None),
            "delay_time": ("Local clock time of the incident as HH:MM text (24h).", None),
            "delay_hour": ("Hour of day (0-23) parsed from delay_time.", "hour"),
            "day_of_week": ("Weekday name, e.g. Monday.", None),
            "min_delay": (
                "Length of the delay to the vehicle in minutes. Zero means logged but no delay.",
                "minutes",
            ),
            "min_gap": (
                "Resulting headway gap to the next vehicle in minutes.",
                "minutes",
            ),
            "bound": ("Direction of travel: N, S, E, W (null when not applicable).", None),
            "vehicle": ("Vehicle number; 0 means unknown.", None),
            "mode": ("Transit mode: subway, bus or streetcar.", None),
        }
        subway = dict(common)
        subway.update(
            {
                "station": ("Station or location name, upper case, e.g. BLOOR STATION.", None),
                "code": ("Delay code; join to subway_delay_codes.code for the description.", None),
                "line": ("Subway line code: YU (Line 1), BD (Line 2), SHP (Line 4), SRT.", None),
            }
        )
        surface = dict(common)
        surface.update(
            {
                "route": ("Route number as text, e.g. 501 (streetcar) or 36 (bus).", None),
                "location": ("Intersection or stop name where the delay occurred.", None),
                "incident": (
                    "Incident category, e.g. Mechanical, Diversion, Collision - TTC.",
                    None,
                ),
            }
        )
        return {
            "subway_delays": subway,
            "bus_delays": surface,
            "streetcar_delays": surface,
            "subway_delay_codes": {
                "code": ("Delay code, matches subway_delays.code.", None),
                "description": ("Plain-English description of the code.", None),
            },
        }

    def time_columns(self) -> dict[str, str]:
        return {t: "delay_date" for t in ("subway_delays", "bus_delays", "streetcar_delays")}

    def joins(self) -> list[JoinDoc]:
        return [
            JoinDoc(
                dataset=self.name,
                left_table="ttc_subway_delays",
                right_table="ttc_subway_delay_codes",
                on="ttc_subway_delays.code = ttc_subway_delay_codes.code",
                kind="left",
                note="Use LEFT JOIN: some codes in the incident log are missing from the lookup.",
            ),
            JoinDoc(
                dataset=self.name,
                left_table="ttc_bus_delays",
                right_table="ttc_streetcar_delays",
                on="UNION ALL on shared columns (delay_date, route, min_delay, min_gap, incident, mode)",
                kind="union",
                note="Bus and streetcar share a schema; combine with UNION ALL, not a join.",
            ),
        ]

    def metrics(self) -> list[MetricDoc]:
        ds = self.name
        return [
            MetricDoc(
                dataset=ds,
                name="delay_count",
                definition="Number of logged delay incidents.",
                sql_expression="COUNT(*)",
                unit="incidents",
                grain="any (filter by mode/line/route/date as needed)",
                tables=["ttc_subway_delays", "ttc_bus_delays", "ttc_streetcar_delays"],
                caveats="Includes rows with min_delay = 0 (logged, no measurable delay).",
            ),
            MetricDoc(
                dataset=ds,
                name="total_delay_minutes",
                definition="Sum of min_delay across incidents.",
                sql_expression="SUM(min_delay)",
                unit="minutes",
                grain="any",
                tables=["ttc_subway_delays", "ttc_bus_delays", "ttc_streetcar_delays"],
            ),
            MetricDoc(
                dataset=ds,
                name="avg_delay_minutes",
                definition="Mean min_delay per incident, including zero-minute rows.",
                sql_expression="AVG(min_delay)",
                unit="minutes",
                grain="any",
                tables=["ttc_subway_delays", "ttc_bus_delays", "ttc_streetcar_delays"],
                caveats="Zero-minute incidents pull the average down; filter min_delay > 0 for "
                "'average of real delays'.",
            ),
            MetricDoc(
                dataset=ds,
                name="avg_nonzero_delay_minutes",
                definition="Mean min_delay over incidents with a positive delay.",
                sql_expression="AVG(CASE WHEN min_delay > 0 THEN min_delay END)",
                unit="minutes",
                grain="any",
                tables=["ttc_subway_delays", "ttc_bus_delays", "ttc_streetcar_delays"],
            ),
            MetricDoc(
                dataset=ds,
                name="delays_per_day",
                definition="Incidents divided by distinct days in the filter window.",
                sql_expression="COUNT(*) * 1.0 / COUNT(DISTINCT delay_date)",
                unit="incidents per day",
                grain="any",
                tables=["ttc_subway_delays", "ttc_bus_delays", "ttc_streetcar_delays"],
                caveats="Counts only days with at least one incident.",
            ),
            MetricDoc(
                dataset=ds,
                name="avg_gap_minutes",
                definition="Mean headway gap caused by delays.",
                sql_expression="AVG(min_gap)",
                unit="minutes",
                grain="any",
                tables=["ttc_subway_delays", "ttc_bus_delays", "ttc_streetcar_delays"],
            ),
            MetricDoc(
                dataset=ds,
                name="major_delay_count",
                definition="Incidents with a delay of 15 minutes or more.",
                sql_expression="SUM(CASE WHEN min_delay >= 15 THEN 1 ELSE 0 END)",
                unit="incidents",
                grain="any",
                tables=["ttc_subway_delays", "ttc_bus_delays", "ttc_streetcar_delays"],
            ),
            MetricDoc(
                dataset=ds,
                name="major_delay_share",
                definition="Share of incidents with min_delay >= 15.",
                sql_expression="AVG(CASE WHEN min_delay >= 15 THEN 1.0 ELSE 0.0 END)",
                unit="ratio (0-1)",
                grain="any",
                tables=["ttc_subway_delays", "ttc_bus_delays", "ttc_streetcar_delays"],
            ),
            MetricDoc(
                dataset=ds,
                name="top_delay_code_share",
                definition="Share of subway incidents attributed to the single most common code.",
                sql_expression="MAX(code_count) * 1.0 / SUM(code_count) over grouped-by-code counts",
                unit="ratio (0-1)",
                grain="subway only; group by code first",
                tables=["ttc_subway_delays", "ttc_subway_delay_codes"],
                caveats="Requires a two-step query: count per code, then compare max to total.",
            ),
            MetricDoc(
                dataset=ds,
                name="incident_category_minutes",
                definition="Total delay minutes by incident category for bus/streetcar.",
                sql_expression="SUM(min_delay) GROUP BY incident",
                unit="minutes",
                grain="bus and streetcar only",
                tables=["ttc_bus_delays", "ttc_streetcar_delays"],
            ),
        ]


def _read_raw(path: Path) -> pd.DataFrame:
    # Code lookup is CSV; all delay files are XLSX with the data on the first sheet.
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path, sheet_name=0)


def clean_delays(raw: pd.DataFrame, table: str) -> pd.DataFrame:
    """Normalise one mode's delay file(s) to the shared schema."""
    rename = {
        "subway_delays": _SUBWAY_RENAME,
        "bus_delays": _BUS_RENAME,
        "streetcar_delays": _STREETCAR_RENAME,
    }[table]
    df = raw.rename(columns={c: rename[c] for c in raw.columns if c in rename})
    df = df[[c for c in rename.values() if c in df.columns]].copy()
    df["delay_date"] = to_date(df["delay_date"])
    # A row without a date cannot be placed on a timeline, so it is unusable for any question.
    df = df[df["delay_date"].notna()]
    df["delay_time"] = df["delay_time"].astype("string").str.strip()
    df["delay_hour"] = to_int(df["delay_time"].str.slice(0, 2))
    for col in ("min_delay", "min_gap", "vehicle"):
        df[col] = to_int(df[col])
    for col in ("route", "station", "location", "code", "line", "incident", "bound", "day_of_week"):
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
    df["mode"] = table.split("_")[0]
    # Exact duplicates come from the source occasionally re-exporting rows; they would double-count.
    df = df.drop_duplicates().reset_index(drop=True)
    df["delay_date"] = df["delay_date"].dt.date
    return df


def clean_codes(raw: pd.DataFrame) -> pd.DataFrame:
    cols = {c: c.strip().lower() for c in raw.columns}
    df = raw.rename(columns=cols)
    df = df[["code", "description"]].copy()
    df["code"] = df["code"].astype("string").str.strip()
    df["description"] = df["description"].astype("string").str.strip().str.title()
    return df.drop_duplicates(subset=["code"]).reset_index(drop=True)
