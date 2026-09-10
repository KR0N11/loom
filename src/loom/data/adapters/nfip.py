"""Adapter for OpenFEMA National Flood Insurance Program (NFIP) redacted claims.

Approach (Adapter pattern): the claims dataset is ~2.6M rows, so we pull a
bounded, filtered slice through the public v2 OData-style API (no key) with
`$select`/`$filter`/`$top`/`$skip` paging and store it as JSON Lines. Two
related tables give the joins a purpose: the NFIP Community Status Book
(community id -> participation dates) and a hand-written states dimension
(state -> name, FEMA region). Money columns become DOUBLE and `total_paid`
(building + contents + ICC) is derived once here so every metric agrees.

Note: OpenFEMA marks v2 claims as deprecated (frozen 2026-06-01). The base URL
is a module constant so it can be repointed to the successor endpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

from loom.data.adapters.base import DataSourceAdapter
from loom.data.cleaning import snake_case_columns, to_date, to_float, to_int
from loom.semantic.models import JoinDoc, MetricDoc

CLAIMS_URL = "https://www.fema.gov/api/open/v2/FimaNfipClaims"
COMMUNITIES_URL = "https://www.fema.gov/api/open/v1/NfipCommunityStatusBook"
PAGE_SIZE = 10_000
DEFAULT_LIMIT = 150_000
SLICE_STATES = ("FL", "TX", "LA", "NJ", "NY", "NC", "SC")
SLICE_MIN_YEAR = 2015

# Verified against a live $top=1 response; every name exists in the v2 schema.
CLAIM_COLUMNS = [
    "id",
    "dateOfLoss",
    "yearOfLoss",
    "state",
    "countyCode",
    "reportedZipCode",
    "nfipCommunityNumberCurrent",
    "ratedFloodZone",
    "floodZoneCurrent",
    "occupancyType",
    "originalConstructionDate",
    "elevatedBuildingIndicator",
    "primaryResidenceIndicator",
    "postFIRMConstructionIndicator",
    "amountPaidOnBuildingClaim",
    "amountPaidOnContentsClaim",
    "amountPaidOnIncreasedCostOfComplianceClaim",
    "totalBuildingInsuranceCoverage",
    "totalContentsInsuranceCoverage",
    "buildingDamageAmount",
    "contentsDamageAmount",
    "netBuildingPaymentAmount",
    "netContentsPaymentAmount",
    "causeOfDamage",
    "waterDepth",
    "numberOfFloorsInTheInsuredBuilding",
    "policyCount",
    "rateMethod",
    "floodEvent",
]
COMMUNITY_COLUMNS = [
    "communityIdNumber",
    "communityName",
    "county",
    "state",
    "initialFloodHazardBoundaryMap",
    "initialFloodInsuranceRateMap",
    "participatingInNFIP",
    "tribal",
    "classRating",
    "sanction",
]

_MONEY = [
    "amount_paid_on_building_claim",
    "amount_paid_on_contents_claim",
    "amount_paid_on_increased_cost_of_compliance_claim",
    "total_building_insurance_coverage",
    "total_contents_insurance_coverage",
    "building_damage_amount",
    "contents_damage_amount",
    "net_building_payment_amount",
    "net_contents_payment_amount",
]

# FEMA regions, from fema.gov/about/organization/regions.
_FEMA_REGIONS: dict[int, list[tuple[str, str]]] = {
    1: [
        ("CT", "Connecticut"),
        ("ME", "Maine"),
        ("MA", "Massachusetts"),
        ("NH", "New Hampshire"),
        ("RI", "Rhode Island"),
        ("VT", "Vermont"),
    ],
    2: [
        ("NJ", "New Jersey"),
        ("NY", "New York"),
        ("PR", "Puerto Rico"),
        ("VI", "U.S. Virgin Islands"),
    ],
    3: [
        ("DE", "Delaware"),
        ("DC", "District of Columbia"),
        ("MD", "Maryland"),
        ("PA", "Pennsylvania"),
        ("VA", "Virginia"),
        ("WV", "West Virginia"),
    ],
    4: [
        ("AL", "Alabama"),
        ("FL", "Florida"),
        ("GA", "Georgia"),
        ("KY", "Kentucky"),
        ("MS", "Mississippi"),
        ("NC", "North Carolina"),
        ("SC", "South Carolina"),
        ("TN", "Tennessee"),
    ],
    5: [
        ("IL", "Illinois"),
        ("IN", "Indiana"),
        ("MI", "Michigan"),
        ("MN", "Minnesota"),
        ("OH", "Ohio"),
        ("WI", "Wisconsin"),
    ],
    6: [
        ("AR", "Arkansas"),
        ("LA", "Louisiana"),
        ("NM", "New Mexico"),
        ("OK", "Oklahoma"),
        ("TX", "Texas"),
    ],
    7: [("IA", "Iowa"), ("KS", "Kansas"), ("MO", "Missouri"), ("NE", "Nebraska")],
    8: [
        ("CO", "Colorado"),
        ("MT", "Montana"),
        ("ND", "North Dakota"),
        ("SD", "South Dakota"),
        ("UT", "Utah"),
        ("WY", "Wyoming"),
    ],
    9: [
        ("AZ", "Arizona"),
        ("CA", "California"),
        ("HI", "Hawaii"),
        ("NV", "Nevada"),
        ("GU", "Guam"),
        ("AS", "American Samoa"),
        ("MP", "Northern Mariana Islands"),
    ],
    10: [("AK", "Alaska"), ("ID", "Idaho"), ("OR", "Oregon"), ("WA", "Washington")],
}


class NFIPClaimsAdapter(DataSourceAdapter):
    """NFIP flood insurance claims (bounded slice) + communities + states."""

    name = "nfip"
    description = (
        "OpenFEMA NFIP flood insurance claims for 7 high-exposure states since 2015, "
        "with the NFIP Community Status Book and a states/FEMA-region dimension."
    )

    def __init__(self, session: requests.Session | None = None) -> None:
        self._session = session or requests.Session()

    def download(self, raw_dir: Path, limit: int | None = None) -> dict[str, Path]:
        target = raw_dir / self.name
        target.mkdir(parents=True, exist_ok=True)
        limit = limit or DEFAULT_LIMIT
        claims_path = target / f"claims_{limit}.jsonl"
        # Cached slices are keyed by limit so a bigger pull never reuses a smaller file.
        if not claims_path.exists():
            self._paginate(
                CLAIMS_URL,
                "FimaNfipClaims",
                claims_path,
                select=CLAIM_COLUMNS,
                filter_=self._claims_filter(),
                limit=limit,
                order_by="id",
            )
        communities_path = target / "communities.jsonl"
        if not communities_path.exists():
            states = ",".join(f"'{s}'" for s in SLICE_STATES)
            self._paginate(
                COMMUNITIES_URL,
                "NfipCommunityStatusBook",
                communities_path,
                select=COMMUNITY_COLUMNS,
                filter_=f"state in ({states})",
                limit=None,
                order_by="communityIdNumber",
                page_size=1000,
            )
        return {"claims": claims_path, "communities": communities_path}

    @staticmethod
    def _claims_filter() -> str:
        states = ",".join(f"'{s}'" for s in SLICE_STATES)
        return f"yearOfLoss ge {SLICE_MIN_YEAR} and state in ({states})"

    def _paginate(
        self,
        url: str,
        entity: str,
        out_path: Path,
        select: list[str],
        filter_: str,
        limit: int | None,
        order_by: str,
        page_size: int = PAGE_SIZE,
    ) -> int:
        """Walk $skip pages and stream rows to JSONL; returns rows written."""
        written = 0
        tmp = out_path.with_suffix(".partial")
        with tmp.open("w") as fh:
            skip = 0
            while True:
                top = page_size if limit is None else min(page_size, limit - written)
                # Stop when the caller's cap is reached; otherwise the API decides.
                if top <= 0:
                    break
                params: dict[str, str | int] = {
                    "$select": ",".join(select),
                    "$filter": filter_,
                    "$top": top,
                    "$skip": skip,
                    "$orderby": order_by,
                    "$format": "json",
                }
                resp = self._session.get(url, params=params, timeout=180)
                resp.raise_for_status()
                rows = resp.json().get(entity, [])
                for row in rows:
                    fh.write(json.dumps(row) + "\n")
                written += len(rows)
                skip += len(rows)
                # A short page means the server ran out of matching rows.
                if len(rows) < top:
                    break
        tmp.rename(out_path)
        return written

    def clean(self, raw_files: dict[str, Path]) -> dict[str, pd.DataFrame]:
        claims = clean_claims(pd.read_json(raw_files["claims"], lines=True))
        communities = clean_communities(pd.read_json(raw_files["communities"], lines=True))
        return {"claims": claims, "communities": communities, "states": states_frame()}

    def table_descriptions(self) -> dict[str, tuple[str, str]]:
        return {
            "claims": (
                "NFIP flood insurance claims (redacted) for FL, TX, LA, NJ, NY, NC, SC with "
                "year of loss 2015 or later. Amounts paid, coverage, damage estimates, flood zone "
                "and building attributes per claim. A bounded slice of the full 2.6M-row dataset.",
                "one row per claim (one policy, one loss date)",
            ),
            "communities": (
                "NFIP Community Status Book: every community in the slice states with its NFIP "
                "participation status and initial flood-map dates.",
                "one row per NFIP community",
            ),
            "states": (
                "States and territories with their full name and FEMA region number.",
                "one row per state",
            ),
        }

    def column_descriptions(self) -> dict[str, dict[str, tuple[str, str | None]]]:
        return {
            "claims": {
                "id": ("Unique claim identifier (UUID).", None),
                "date_of_loss": ("Date the flood loss occurred.", None),
                "year_of_loss": ("Calendar year of date_of_loss.", "year"),
                "state": (
                    "Two-letter state code of the insured property; join to states.state.",
                    None,
                ),
                "county_code": ("5-digit FIPS county code.", None),
                "reported_zip_code": ("ZIP code of the insured property.", None),
                "nfip_community_number_current": (
                    "Current NFIP community id; join to communities.community_id_number.",
                    None,
                ),
                "rated_flood_zone": (
                    "Flood zone used for rating. Zones starting with A or V are Special Flood "
                    "Hazard Areas (SFHA); X/B/C are lower risk.",
                    None,
                ),
                "flood_zone_current": ("Flood zone on the current effective map.", None),
                "occupancy_type": (
                    "Occupancy code: 1 single-family, 2 two-to-four family, 3 other residential, "
                    "4 non-residential, 6 other residential building, 11-18 condo/RCBAP types.",
                    None,
                ),
                "original_construction_date": ("Date the building was originally built.", None),
                "elevated_building_indicator": ("True if the building is elevated.", None),
                "primary_residence_indicator": ("True if the insured's primary residence.", None),
                "post_firm_construction_indicator": (
                    "True if built after the community's first Flood Insurance Rate Map.",
                    None,
                ),
                "amount_paid_on_building_claim": ("Dollars paid for building damage.", "USD"),
                "amount_paid_on_contents_claim": ("Dollars paid for contents damage.", "USD"),
                "amount_paid_on_increased_cost_of_compliance_claim": (
                    "Dollars paid under Increased Cost of Compliance (ICC) coverage.",
                    "USD",
                ),
                "total_paid": (
                    "Derived: building + contents + ICC amounts paid (nulls treated as 0).",
                    "USD",
                ),
                "total_building_insurance_coverage": ("Building coverage limit.", "USD"),
                "total_contents_insurance_coverage": ("Contents coverage limit.", "USD"),
                "building_damage_amount": ("Adjuster's estimate of building damage.", "USD"),
                "contents_damage_amount": ("Adjuster's estimate of contents damage.", "USD"),
                "net_building_payment_amount": ("Net building payment after adjustments.", "USD"),
                "net_contents_payment_amount": ("Net contents payment after adjustments.", "USD"),
                "cause_of_damage": (
                    "Cause code as text: 0 other, 1 tidal water overflow, 2 stream/river overflow, "
                    "3 alluvial fan, 4 accumulation of rainfall/snowmelt, 7 erosion, 8 mudflow, "
                    "9 earth movement.",
                    None,
                ),
                "water_depth": ("Depth of flood water in the building.", "feet"),
                "number_of_floors_in_the_insured_building": ("Number of floors.", "floors"),
                "policy_count": ("Number of policies/units on the claim (usually 1).", "policies"),
                "rate_method": ("Rating method code.", None),
                "flood_event": (
                    "Named flood event, e.g. Hurricane Ian; null for unnamed events.",
                    None,
                ),
                "is_sfha": (
                    "Derived: rated_flood_zone starts with A or V (Special Flood Hazard Area).",
                    None,
                ),
            },
            "communities": {
                "community_id_number": ("6-character NFIP community id; join key to claims.", None),
                "community_name": ("Community name as listed by FEMA.", None),
                "county": ("County name.", None),
                "state": ("Two-letter state code.", None),
                "initial_flood_hazard_boundary_map": (
                    "Date of the first flood hazard boundary map.",
                    None,
                ),
                "initial_flood_insurance_rate_map": (
                    "Date of the first Flood Insurance Rate Map (FIRM).",
                    None,
                ),
                "participating_in_nfip": (
                    "True if the community currently participates in NFIP.",
                    None,
                ),
                "tribal": ("True for tribal communities.", None),
                "class_rating": (
                    "Community Rating System class (1 best, 10 none); null if not rated.",
                    None,
                ),
                "sanction": ("True if the community is sanctioned / suspended.", None),
            },
            "states": {
                "state": ("Two-letter state or territory code.", None),
                "state_name": ("Full state name.", None),
                "fema_region": ("FEMA region number (1-10).", None),
            },
        }

    def time_columns(self) -> dict[str, str]:
        return {"claims": "date_of_loss"}

    def joins(self) -> list[JoinDoc]:
        return [
            JoinDoc(
                dataset=self.name,
                left_table="nfip_claims",
                right_table="nfip_states",
                on="nfip_claims.state = nfip_states.state",
                kind="left",
                note="Every claim state exists in states; use for state names and FEMA regions.",
            ),
            JoinDoc(
                dataset=self.name,
                left_table="nfip_claims",
                right_table="nfip_communities",
                on="nfip_claims.nfip_community_number_current = nfip_communities.community_id_number",
                kind="left",
                note="LEFT JOIN: some claims have a null or retired community number.",
            ),
        ]

    def metrics(self) -> list[MetricDoc]:
        ds = self.name
        return [
            MetricDoc(
                dataset=ds,
                name="claim_count",
                definition="Number of claims.",
                sql_expression="COUNT(*)",
                unit="claims",
                grain="any",
                tables=["nfip_claims"],
            ),
            MetricDoc(
                dataset=ds,
                name="total_paid",
                definition="Total dollars paid: building + contents + ICC.",
                sql_expression="SUM(total_paid)",
                unit="USD",
                grain="any",
                tables=["nfip_claims"],
                caveats="Closed-without-payment claims contribute 0, not null.",
            ),
            MetricDoc(
                dataset=ds,
                name="avg_claim_severity",
                definition="Mean total_paid per claim including zero-paid claims.",
                sql_expression="AVG(total_paid)",
                unit="USD per claim",
                grain="any",
                tables=["nfip_claims"],
                caveats="Use AVG(CASE WHEN total_paid > 0 THEN total_paid END) for paid claims only.",
            ),
            MetricDoc(
                dataset=ds,
                name="paid_claim_share",
                definition="Share of claims with any payment.",
                sql_expression="AVG(CASE WHEN total_paid > 0 THEN 1.0 ELSE 0.0 END)",
                unit="ratio (0-1)",
                grain="any",
                tables=["nfip_claims"],
            ),
            MetricDoc(
                dataset=ds,
                name="claim_frequency_per_policy",
                definition="Claims per policy on the claim records.",
                sql_expression="COUNT(*) * 1.0 / SUM(policy_count)",
                unit="claims per policy",
                grain="any",
                tables=["nfip_claims"],
                caveats="policy_count is per-claim, not the exposure base; this is not a true "
                "frequency. True frequency needs the policies dataset, which is not loaded.",
            ),
            MetricDoc(
                dataset=ds,
                name="building_loss_ratio",
                definition="Building dollars paid divided by building coverage limit.",
                sql_expression="SUM(amount_paid_on_building_claim) / NULLIF(SUM(total_building_insurance_coverage), 0)",
                unit="ratio",
                grain="any",
                tables=["nfip_claims"],
                caveats="Coverage is per claim, not per policy-year, so this overstates a true loss ratio.",
            ),
            MetricDoc(
                dataset=ds,
                name="sfha_claim_share",
                definition="Share of claims in Special Flood Hazard Areas (rated_flood_zone starting with A or V).",
                sql_expression="AVG(CASE WHEN is_sfha THEN 1.0 ELSE 0.0 END)",
                unit="ratio (0-1)",
                grain="any",
                tables=["nfip_claims"],
            ),
            MetricDoc(
                dataset=ds,
                name="median_water_depth",
                definition="Median reported flood water depth.",
                sql_expression="MEDIAN(water_depth)",
                unit="feet",
                grain="any",
                tables=["nfip_claims"],
                caveats="water_depth is null for many claims; the median ignores nulls.",
            ),
            MetricDoc(
                dataset=ds,
                name="avg_building_damage",
                definition="Mean adjuster estimate of building damage.",
                sql_expression="AVG(building_damage_amount)",
                unit="USD",
                grain="any",
                tables=["nfip_claims"],
            ),
            MetricDoc(
                dataset=ds,
                name="payment_to_damage_ratio",
                definition="Building dollars paid divided by estimated building damage.",
                sql_expression="SUM(amount_paid_on_building_claim) / NULLIF(SUM(building_damage_amount), 0)",
                unit="ratio",
                grain="any",
                tables=["nfip_claims"],
            ),
            MetricDoc(
                dataset=ds,
                name="claims_per_community",
                definition="Claims divided by distinct participating communities in scope.",
                sql_expression="COUNT(*) * 1.0 / COUNT(DISTINCT nfip_community_number_current)",
                unit="claims per community",
                grain="any",
                tables=["nfip_claims", "nfip_communities"],
            ),
        ]


def clean_claims(raw: pd.DataFrame) -> pd.DataFrame:
    df = snake_case_columns(raw)
    for col in ("date_of_loss", "original_construction_date"):
        if col in df.columns:
            df[col] = to_date(df[col])
    # A claim without a loss date cannot be placed in any trend or time filter.
    df = df[df["date_of_loss"].notna()]
    for col in _MONEY:
        if col in df.columns:
            df[col] = to_float(df[col])
    for col in (
        "year_of_loss",
        "water_depth",
        "number_of_floors_in_the_insured_building",
        "policy_count",
    ):
        if col in df.columns:
            df[col] = to_int(df[col])
    for col in (
        "elevated_building_indicator",
        "primary_residence_indicator",
        "post_firm_construction_indicator",
    ):
        if col in df.columns:
            df[col] = df[col].astype("boolean")
    for col in (
        "state",
        "county_code",
        "reported_zip_code",
        "nfip_community_number_current",
        "rated_flood_zone",
        "flood_zone_current",
        "cause_of_damage",
        "rate_method",
        "flood_event",
        "id",
    ):
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
    # total_paid is the headline money number; deriving it once here keeps every metric consistent.
    df["total_paid"] = (
        df["amount_paid_on_building_claim"].fillna(0)
        + df["amount_paid_on_contents_claim"].fillna(0)
        + df["amount_paid_on_increased_cost_of_compliance_claim"].fillna(0)
    )
    df["is_sfha"] = (
        df["rated_flood_zone"].str.upper().str.match(r"^[AV]").fillna(False).astype(bool)
    )
    # The API can return the same claim id on overlapping pages; keep one copy.
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    for col in ("date_of_loss", "original_construction_date"):
        df[col] = df[col].dt.date
    return df


def clean_communities(raw: pd.DataFrame) -> pd.DataFrame:
    df = snake_case_columns(raw)
    for col in ("initial_flood_hazard_boundary_map", "initial_flood_insurance_rate_map"):
        if col in df.columns:
            df[col] = to_date(df[col]).dt.date
    for col in ("participating_in_nfip", "tribal", "sanction"):
        if col in df.columns:
            df[col] = df[col].astype("boolean")
    if "class_rating" in df.columns:
        df["class_rating"] = to_int(df["class_rating"])
    for col in ("community_id_number", "community_name", "county", "state"):
        df[col] = df[col].astype("string").str.strip()
    return df.drop_duplicates(subset=["community_id_number"]).reset_index(drop=True)


def states_frame() -> pd.DataFrame:
    rows = [
        {"state": code, "state_name": name, "fema_region": region}
        for region, members in _FEMA_REGIONS.items()
        for code, name in members
    ]
    return pd.DataFrame(rows).sort_values("state").reset_index(drop=True)
