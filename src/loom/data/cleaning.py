"""Small, shared cleaning helpers used by every adapter.

Approach: adapters differ in *what* they rename, not *how*; keeping the
snake_case / date / money coercions here means the two datasets are cleaned by
the same code paths and tests.
"""

from __future__ import annotations

import re

import pandas as pd

_CAMEL_1 = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_2 = re.compile(r"([a-z0-9])([A-Z])")
_NON_WORD = re.compile(r"[^0-9a-zA-Z]+")


def to_snake(name: str) -> str:
    """'Min Delay' -> 'min_delay', 'amountPaidOnBuildingClaim' -> 'amount_paid_on_building_claim'."""
    name = _NON_WORD.sub("_", str(name).strip())
    name = _CAMEL_1.sub(r"\1_\2", name)
    name = _CAMEL_2.sub(r"\1_\2", name)
    return re.sub(r"_+", "_", name).strip("_").lower()


def snake_case_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: to_snake(c) for c in df.columns})


def to_date(series: pd.Series) -> pd.Series:
    """Parse to a date (no time part); unparseable values become null instead of raising."""
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()


def to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("float64")


def to_int(series: pd.Series) -> pd.Series:
    """Nullable integer so a blank cell does not turn the whole column into floats."""
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")
