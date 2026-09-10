"""Adapter cleaning logic on tiny in-memory frames (no network)."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from loom.data.adapters import get_adapter, list_adapters
from loom.data.adapters.nfip import clean_claims, clean_communities, states_frame
from loom.data.adapters.registry import UnknownDatasetError
from loom.data.adapters.toronto_ttc import clean_codes, clean_delays
from loom.data.cleaning import to_snake


def _subway_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-01", None, "2024-01-02"],
            "Time": ["02:00", "02:00", "03:00", "14:30"],
            "Day": ["Monday", "Monday", "Monday", "Tuesday"],
            "Station": ["DUNDAS STATION", "DUNDAS STATION", "BLOOR STATION", "KING STATION "],
            "Code": ["MUIS", "MUIS", "SUDP", "PUOPO"],
            "Min Delay": [0, 0, 5, 12],
            "Min Gap": [0, 0, 10, 18],
            "Bound": ["N", "N", "S", None],
            "Line": ["YU", "YU", "YU", "BD"],
            "Vehicle": [0, 0, 5491, 5555],
        }
    )


# Proves: subway rows get snake_case names, exact duplicates are removed, null dates dropped.
def test_clean_subway_dedupes_and_drops_null_dates() -> None:
    df = clean_delays(_subway_raw(), "subway_delays")
    assert list(df.columns) == [
        "delay_date",
        "delay_time",
        "day_of_week",
        "station",
        "code",
        "min_delay",
        "min_gap",
        "bound",
        "line",
        "vehicle",
        "delay_hour",
        "mode",
    ]
    assert len(df) == 2
    assert df["delay_date"].iloc[0] == dt.date(2024, 1, 1)
    assert set(df["mode"]) == {"subway"}
    assert df["delay_hour"].tolist() == [2, 14]
    assert df["station"].iloc[1] == "KING STATION"
    assert str(df["min_delay"].dtype) == "Int64"


# Proves: bus files map Direction->bound and Route/Location/Incident survive.
def test_clean_bus_maps_direction_to_bound() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["2023-05-05"],
            "Route": [36],
            "Time": ["07:15"],
            "Day": ["Friday"],
            "Location": ["FINCH STATION"],
            "Incident": ["Mechanical"],
            "Min Delay": [8],
            "Min Gap": [16],
            "Direction": ["W"],
            "Vehicle": [8914],
        }
    )
    df = clean_delays(raw, "bus_delays")
    assert df["bound"].iloc[0] == "W"
    assert df["route"].iloc[0] == "36"
    assert df["mode"].iloc[0] == "bus"


# Proves: streetcar 'Line' becomes 'route' so bus and streetcar share a schema.
def test_clean_streetcar_line_becomes_route() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["2023-05-05"],
            "Line": [505],
            "Time": ["07:15"],
            "Day": ["Friday"],
            "Location": ["DUNDAS AND MCCAUL"],
            "Incident": ["Security"],
            "Min Delay": [10],
            "Min Gap": [20],
            "Bound": ["W"],
            "Vehicle": [4416],
        }
    )
    df = clean_delays(raw, "streetcar_delays")
    assert "route" in df.columns and "line" not in df.columns


# Proves: the code lookup keeps one row per code with title-cased descriptions.
def test_clean_codes() -> None:
    raw = pd.DataFrame(
        {
            "_id": [1, 2, 3],
            "CODE": ["EUAC", "EUAC ", "EUAL"],
            "DESCRIPTION": ["AIR CONDITIONING", "AIR CONDITIONING", "ALTERNATING CURRENT"],
        }
    )
    df = clean_codes(raw)
    assert df["code"].tolist() == ["EUAC", "EUAL"]
    assert df["description"].iloc[0] == "Air Conditioning"


def _claims_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["a", "a", "b", "c"],
            "dateOfLoss": [
                "2022-09-28T00:00:00.000Z",
                "2022-09-28T00:00:00.000Z",
                None,
                "2017-08-27T00:00:00.000Z",
            ],
            "yearOfLoss": [2022, 2022, 2021, 2017],
            "state": ["FL", "FL", "TX", "TX"],
            "countyCode": ["12071", "12071", "48201", "48201"],
            "reportedZipCode": ["33901", "33901", "77001", "77002"],
            "nfipCommunityNumberCurrent": ["125124", "125124", "480296", None],
            "ratedFloodZone": ["AE", "AE", "X", "VE"],
            "floodZoneCurrent": [None, None, None, None],
            "occupancyType": [1, 1, 1, 2],
            "originalConstructionDate": ["1980-01-01T00:00:00.000Z"] * 4,
            "elevatedBuildingIndicator": [False, False, True, None],
            "primaryResidenceIndicator": [True, True, False, True],
            "postFIRMConstructionIndicator": [True, True, None, False],
            "amountPaidOnBuildingClaim": [10000.5, 10000.5, None, 250000],
            "amountPaidOnContentsClaim": [2000, 2000, None, None],
            "amountPaidOnIncreasedCostOfComplianceClaim": [None, None, None, 30000],
            "totalBuildingInsuranceCoverage": [250000, 250000, 100000, 250000],
            "totalContentsInsuranceCoverage": [100000, 100000, 0, 50000],
            "buildingDamageAmount": [12000, 12000, 0, 400000],
            "contentsDamageAmount": [2500, 2500, 0, 0],
            "netBuildingPaymentAmount": [10000.5, 10000.5, 0, 250000],
            "netContentsPaymentAmount": [2000, 2000, 0, 0],
            "causeOfDamage": ["1", "1", "4", "1"],
            "waterDepth": [3, 3, None, 8],
            "numberOfFloorsInTheInsuredBuilding": [1, 1, 2, 1],
            "policyCount": [1, 1, 1, 1],
            "rateMethod": ["1", "1", "7", "1"],
            "floodEvent": ["Hurricane Ian", "Hurricane Ian", None, "Hurricane Harvey"],
        }
    )


# Proves: claims get snake_case names, total_paid = building + contents + ICC with nulls as 0,
# duplicate ids collapse, null loss dates are dropped, SFHA flag derived from zone prefix.
def test_clean_claims_derives_total_paid_and_dedupes() -> None:
    df = clean_claims(_claims_raw())
    assert len(df) == 2
    assert "amount_paid_on_building_claim" in df.columns
    row_a = df[df["id"] == "a"].iloc[0]
    row_c = df[df["id"] == "c"].iloc[0]
    assert row_a["total_paid"] == pytest.approx(12000.5)
    assert row_c["total_paid"] == pytest.approx(280000.0)
    assert bool(row_a["is_sfha"]) is True and bool(row_c["is_sfha"]) is True
    assert row_a["date_of_loss"] == dt.date(2022, 9, 28)
    assert str(df["year_of_loss"].dtype) == "Int64"


# Proves: community book columns are snake_case with DATE-typed map dates.
def test_clean_communities() -> None:
    raw = pd.DataFrame(
        {
            "communityIdNumber": ["120001", "120001"],
            "communityName": ["ALACHUA COUNTY*"] * 2,
            "county": ["ALACHUA COUNTY"] * 2,
            "state": ["FL"] * 2,
            "initialFloodHazardBoundaryMap": ["1974-06-07T00:00:00.000Z"] * 2,
            "initialFloodInsuranceRateMap": [None] * 2,
            "participatingInNFIP": [True] * 2,
            "tribal": [False] * 2,
            "classRating": [None, None],
            "sanction": [False] * 2,
        }
    )
    df = clean_communities(raw)
    assert len(df) == 1
    assert df["initial_flood_hazard_boundary_map"].iloc[0] == dt.date(1974, 6, 7)
    assert pd.isna(df["initial_flood_insurance_rate_map"].iloc[0])


# Proves: the states dimension covers every slice state exactly once with a FEMA region.
def test_states_frame() -> None:
    df = states_frame()
    assert df["state"].is_unique
    assert set(["FL", "TX", "LA", "NJ", "NY", "NC", "SC"]) <= set(df["state"])
    assert df.loc[df["state"] == "TX", "fema_region"].iloc[0] == 6


# Proves: camelCase and spaced names both become snake_case.
def test_to_snake() -> None:
    assert to_snake("amountPaidOnBuildingClaim") == "amount_paid_on_building_claim"
    assert to_snake("Min Delay") == "min_delay"
    assert to_snake("initialFIRM") == "initial_firm"


# Proves: both adapters are registered and an unknown name fails loudly.
def test_registry() -> None:
    assert set(list_adapters()) == {"ttc", "nfip"}
    assert get_adapter("ttc").qualified("subway_delays") == "ttc_subway_delays"
    with pytest.raises(UnknownDatasetError):
        get_adapter("nope")


# Proves: every metric and join references only qualified tables the adapter loads.
@pytest.mark.parametrize("name", ["ttc", "nfip"])
def test_metrics_and_joins_reference_known_tables(name: str) -> None:
    adapter = get_adapter(name)
    known = {adapter.qualified(t) for t in adapter.table_descriptions()}
    for m in adapter.metrics():
        assert set(m.tables) <= known, m.name
    for j in adapter.joins():
        assert {j.left_table, j.right_table} <= known
