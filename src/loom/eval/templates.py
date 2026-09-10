"""Hand-written, guaranteed-correct question templates per dataset.

Approach: each template pairs a natural-language question with DuckDB SQL over
the ingested tables. `{param}` slots are filled from real values sampled at
generation time, so one template yields several grounded questions. Templates
that fail against the live schema are skipped with a warning, never guessed.
"""

from __future__ import annotations

from loom.eval.benchmark import Difficulty
from loom.eval.generate import Template

E, M, H = Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD

# ---------- demo (tests only; matches tests/conftest.py tiny_db) ----------

DEMO_TEMPLATES: list[Template] = [
    Template(
        key="total-by-province",
        dataset="demo",
        difficulty=E,
        analysis_type="lookup",
        question="What is the total sales amount in {province}?",
        sql="SELECT sum(amount) AS total_amount FROM demo_sales WHERE province = '{province}'",
        expected_tables=["demo_sales"],
        params={"province": "SELECT DISTINCT province FROM demo_sales"},
        max_variants=3,
    ),
    Template(
        key="monthly-trend",
        dataset="demo",
        difficulty=M,
        analysis_type="trend",
        question="How did monthly sales amount change over 2024?",
        sql=(
            "SELECT date_trunc('month', sale_date) AS month, sum(amount) AS total_amount "
            "FROM demo_sales GROUP BY 1 ORDER BY 1"
        ),
        expected_tables=["demo_sales"],
        max_variants=1,
    ),
    Template(
        key="province-compare",
        dataset="demo",
        difficulty=M,
        analysis_type="comparison",
        question="Compare total sales between {province_a} and {province_b}.",
        sql=(
            "SELECT sum(CASE WHEN province = '{province_a}' THEN amount END) AS total_a, "
            "sum(CASE WHEN province = '{province_b}' THEN amount END) AS total_b "
            "FROM demo_sales"
        ),
        expected_tables=["demo_sales"],
        params={
            "province_a": "SELECT DISTINCT province FROM demo_sales",
            "province_b": "SELECT DISTINCT province FROM demo_sales",
        },
        max_variants=2,
    ),
    Template(
        key="driver-share",
        dataset="demo",
        difficulty=H,
        analysis_type="driver",
        question="Which province contributed the largest share of total sales, and what share?",
        sql=(
            "WITH tot AS (SELECT sum(amount) AS t FROM demo_sales) "
            "SELECT province, sum(amount) / tot.t AS share FROM demo_sales, tot "
            "GROUP BY province, tot.t ORDER BY share DESC LIMIT 1"
        ),
        expected_tables=["demo_sales"],
        max_variants=1,
    ),
]

# ---------- TTC delays ----------

_TTC_YEARS = "SELECT DISTINCT year(delay_date) FROM ttc_subway_delays ORDER BY 1"
_TTC_LINES = (
    "SELECT line FROM ttc_subway_delays WHERE line IN ('YU','BD','SHP') GROUP BY line ORDER BY 1"
)
_TTC_STATIONS = (
    "SELECT station FROM ttc_subway_delays WHERE station LIKE '% STATION' "
    "GROUP BY station ORDER BY count(*) DESC LIMIT 8"
)
_TTC_DAYS = "SELECT DISTINCT day_of_week FROM ttc_subway_delays"
_TTC_BUS_ROUTES = "SELECT route FROM ttc_bus_delays GROUP BY route ORDER BY count(*) DESC LIMIT 8"
_TTC_INCIDENTS = (
    "SELECT incident FROM ttc_bus_delays GROUP BY incident ORDER BY count(*) DESC LIMIT 6"
)
_TTC_MONTHS = "SELECT DISTINCT strftime(delay_date, '%Y-%m') FROM ttc_subway_delays ORDER BY 1"

TTC_TEMPLATES: list[Template] = [
    # easy lookups
    Template(
        key="subway-count-year",
        dataset="ttc",
        difficulty=E,
        analysis_type="lookup",
        question="How many subway delay incidents were recorded in {year}?",
        sql="SELECT count(*) AS delay_count FROM ttc_subway_delays WHERE year(delay_date) = {year}",
        expected_tables=["ttc_subway_delays"],
        expected_metrics=["delay_count"],
        params={"year": _TTC_YEARS},
        max_variants=4,
    ),
    Template(
        key="subway-total-minutes-line-year",
        dataset="ttc",
        difficulty=E,
        analysis_type="lookup",
        question="What was the total delay minutes on subway line {line} in {year}?",
        sql=(
            "SELECT sum(min_delay) AS total_delay_minutes FROM ttc_subway_delays "
            "WHERE line = '{line}' AND year(delay_date) = {year}"
        ),
        expected_tables=["ttc_subway_delays"],
        expected_metrics=["total_delay_minutes"],
        params={"line": _TTC_LINES, "year": _TTC_YEARS},
        max_variants=5,
    ),
    Template(
        key="subway-avg-delay-station",
        dataset="ttc",
        difficulty=E,
        analysis_type="lookup",
        question="What is the average delay in minutes at {station} across all years?",
        sql=(
            "SELECT avg(min_delay) AS avg_delay_minutes FROM ttc_subway_delays "
            "WHERE station = '{station}'"
        ),
        expected_tables=["ttc_subway_delays"],
        expected_metrics=["avg_delay_minutes"],
        params={"station": _TTC_STATIONS},
        max_variants=5,
    ),
    Template(
        key="subway-max-delay-year",
        dataset="ttc",
        difficulty=E,
        analysis_type="lookup",
        question="What was the single longest subway delay in minutes in {year}?",
        sql="SELECT max(min_delay) AS max_delay_minutes FROM ttc_subway_delays WHERE year(delay_date) = {year}",
        expected_tables=["ttc_subway_delays"],
        params={"year": _TTC_YEARS},
        max_variants=3,
    ),
    Template(
        key="bus-count-route-year",
        dataset="ttc",
        difficulty=E,
        analysis_type="lookup",
        question="How many bus delays did route {route} have in {year}?",
        sql=(
            "SELECT count(*) AS delay_count FROM ttc_bus_delays "
            "WHERE route = '{route}' AND year(delay_date) = {year}"
        ),
        expected_tables=["ttc_bus_delays"],
        expected_metrics=["delay_count"],
        params={"route": _TTC_BUS_ROUTES, "year": _TTC_YEARS},
        max_variants=5,
    ),
    Template(
        key="bus-total-minutes-incident",
        dataset="ttc",
        difficulty=E,
        analysis_type="lookup",
        question="How many total delay minutes were caused by '{incident}' incidents on buses?",
        sql=(
            "SELECT sum(min_delay) AS total_delay_minutes FROM ttc_bus_delays "
            "WHERE incident = '{incident}'"
        ),
        expected_tables=["ttc_bus_delays"],
        expected_metrics=["total_delay_minutes"],
        params={"incident": _TTC_INCIDENTS},
        max_variants=4,
    ),
    Template(
        key="streetcar-count-year",
        dataset="ttc",
        difficulty=E,
        analysis_type="lookup",
        question="How many streetcar delays were logged in {year}?",
        sql="SELECT count(*) AS delay_count FROM ttc_streetcar_delays WHERE year(delay_date) = {year}",
        expected_tables=["ttc_streetcar_delays"],
        expected_metrics=["delay_count"],
        params={"year": _TTC_YEARS},
        max_variants=3,
    ),
    Template(
        key="subway-code-description-count",
        dataset="ttc",
        difficulty=E,
        analysis_type="lookup",
        question="Which subway delay code description had the most incidents in {year}?",
        sql=(
            "SELECT c.description, count(*) AS delay_count FROM ttc_subway_delays d "
            "JOIN ttc_subway_delay_codes c ON d.code = c.code "
            "WHERE year(d.delay_date) = {year} GROUP BY 1 ORDER BY 2 DESC LIMIT 1"
        ),
        expected_tables=["ttc_subway_delays", "ttc_subway_delay_codes"],
        expected_metrics=["delay_count"],
        params={"year": _TTC_YEARS},
        max_variants=3,
    ),
    # medium trends / comparisons
    Template(
        key="subway-monthly-trend-year",
        dataset="ttc",
        difficulty=M,
        analysis_type="trend",
        question="How did the number of subway delays per month trend through {year}?",
        sql=(
            "SELECT strftime(delay_date, '%Y-%m') AS month, count(*) AS delay_count "
            "FROM ttc_subway_delays WHERE year(delay_date) = {year} GROUP BY 1 ORDER BY 1"
        ),
        expected_tables=["ttc_subway_delays"],
        expected_metrics=["delay_count"],
        params={"year": _TTC_YEARS},
        max_variants=5,
    ),
    Template(
        key="subway-monthly-minutes-line",
        dataset="ttc",
        difficulty=M,
        analysis_type="trend",
        question="Show the monthly total delay minutes on line {line} over {year}.",
        sql=(
            "SELECT strftime(delay_date, '%Y-%m') AS month, sum(min_delay) AS total_delay_minutes "
            "FROM ttc_subway_delays WHERE line = '{line}' AND year(delay_date) = {year} "
            "GROUP BY 1 ORDER BY 1"
        ),
        expected_tables=["ttc_subway_delays"],
        expected_metrics=["total_delay_minutes"],
        params={"line": _TTC_LINES, "year": _TTC_YEARS},
        max_variants=6,
    ),
    Template(
        key="bus-monthly-avg-delay",
        dataset="ttc",
        difficulty=M,
        analysis_type="trend",
        question="What was the average bus delay in minutes by month in {year}?",
        sql=(
            "SELECT strftime(delay_date, '%Y-%m') AS month, avg(min_delay) AS avg_delay_minutes "
            "FROM ttc_bus_delays WHERE year(delay_date) = {year} GROUP BY 1 ORDER BY 1"
        ),
        expected_tables=["ttc_bus_delays"],
        expected_metrics=["avg_delay_minutes"],
        params={"year": _TTC_YEARS},
        max_variants=4,
    ),
    Template(
        key="subway-line-compare-year",
        dataset="ttc",
        difficulty=M,
        analysis_type="comparison",
        question="Compare the average delay minutes on line {line_a} versus line {line_b} in {year}.",
        sql=(
            "SELECT avg(CASE WHEN line = '{line_a}' THEN min_delay END) AS avg_delay_a, "
            "avg(CASE WHEN line = '{line_b}' THEN min_delay END) AS avg_delay_b "
            "FROM ttc_subway_delays WHERE year(delay_date) = {year}"
        ),
        expected_tables=["ttc_subway_delays"],
        expected_metrics=["avg_delay_minutes"],
        params={"line_a": _TTC_LINES, "line_b": _TTC_LINES, "year": _TTC_YEARS},
        max_variants=6,
    ),
    Template(
        key="weekday-weekend-compare",
        dataset="ttc",
        difficulty=M,
        analysis_type="comparison",
        question="Did subway delays happen more often on {day_a} or {day_b} in {year}?",
        sql=(
            "SELECT sum(CASE WHEN day_of_week = '{day_a}' THEN 1 ELSE 0 END) AS count_a, "
            "sum(CASE WHEN day_of_week = '{day_b}' THEN 1 ELSE 0 END) AS count_b "
            "FROM ttc_subway_delays WHERE year(delay_date) = {year}"
        ),
        expected_tables=["ttc_subway_delays"],
        expected_metrics=["delay_count"],
        params={"day_a": _TTC_DAYS, "day_b": _TTC_DAYS, "year": _TTC_YEARS},
        max_variants=6,
    ),
    Template(
        key="mode-compare-year",
        dataset="ttc",
        difficulty=M,
        analysis_type="comparison",
        question="In {year}, which had more total delay minutes: buses or streetcars?",
        sql=(
            "SELECT 'bus' AS mode, sum(min_delay) AS total_delay_minutes FROM ttc_bus_delays "
            "WHERE year(delay_date) = {year} "
            "UNION ALL SELECT 'streetcar', sum(min_delay) FROM ttc_streetcar_delays "
            "WHERE year(delay_date) = {year} ORDER BY 2 DESC"
        ),
        expected_tables=["ttc_bus_delays", "ttc_streetcar_delays"],
        expected_metrics=["total_delay_minutes"],
        params={"year": _TTC_YEARS},
        max_variants=4,
    ),
    Template(
        key="yoy-subway-minutes",
        dataset="ttc",
        difficulty=M,
        analysis_type="comparison",
        question="How did total subway delay minutes change from {year_a} to {year_b}?",
        sql=(
            "SELECT sum(CASE WHEN year(delay_date) = {year_a} THEN min_delay END) AS minutes_a, "
            "sum(CASE WHEN year(delay_date) = {year_b} THEN min_delay END) AS minutes_b, "
            "sum(CASE WHEN year(delay_date) = {year_b} THEN min_delay END) - "
            "sum(CASE WHEN year(delay_date) = {year_a} THEN min_delay END) AS change_minutes "
            "FROM ttc_subway_delays"
        ),
        expected_tables=["ttc_subway_delays"],
        expected_metrics=["total_delay_minutes"],
        params={"year_a": _TTC_YEARS, "year_b": _TTC_YEARS},
        max_variants=4,
    ),
    Template(
        key="hourly-peak",
        dataset="ttc",
        difficulty=M,
        analysis_type="trend",
        question="At which hour of the day do subway delays peak on line {line}, and how many?",
        sql=(
            "SELECT delay_hour, count(*) AS delay_count FROM ttc_subway_delays "
            "WHERE line = '{line}' AND delay_hour IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 1"
        ),
        expected_tables=["ttc_subway_delays"],
        expected_metrics=["delay_count"],
        params={"line": _TTC_LINES},
        max_variants=5,
    ),
    # hard driver analyses
    Template(
        key="driver-code-change",
        dataset="ttc",
        difficulty=H,
        analysis_type="driver",
        question=(
            "Which subway delay code contributed most to the change in total delay minutes "
            "from {year_a} to {year_b}, and by how many minutes?"
        ),
        sql=(
            "WITH by_code AS ("
            "SELECT d.code, c.description, "
            "sum(CASE WHEN year(delay_date) = {year_a} THEN min_delay ELSE 0 END) AS minutes_a, "
            "sum(CASE WHEN year(delay_date) = {year_b} THEN min_delay ELSE 0 END) AS minutes_b "
            "FROM ttc_subway_delays d LEFT JOIN ttc_subway_delay_codes c ON d.code = c.code "
            "WHERE year(delay_date) IN ({year_a}, {year_b}) GROUP BY 1, 2) "
            "SELECT code, description, minutes_b - minutes_a AS delta_minutes "
            "FROM by_code ORDER BY abs(minutes_b - minutes_a) DESC LIMIT 1"
        ),
        expected_tables=["ttc_subway_delays", "ttc_subway_delay_codes"],
        expected_metrics=["total_delay_minutes"],
        params={"year_a": _TTC_YEARS, "year_b": _TTC_YEARS},
        max_variants=5,
    ),
    Template(
        key="driver-station-share-line",
        dataset="ttc",
        difficulty=H,
        analysis_type="driver",
        question=(
            "Which station accounts for the largest share of delay minutes on line {line} "
            "in {year}, and what is that share?"
        ),
        sql=(
            "WITH tot AS (SELECT sum(min_delay) AS t FROM ttc_subway_delays "
            "WHERE line = '{line}' AND year(delay_date) = {year}) "
            "SELECT station, sum(min_delay) AS station_minutes, sum(min_delay) / tot.t AS share "
            "FROM ttc_subway_delays, tot WHERE line = '{line}' AND year(delay_date) = {year} "
            "GROUP BY station, tot.t ORDER BY share DESC LIMIT 1"
        ),
        expected_tables=["ttc_subway_delays"],
        expected_metrics=["total_delay_minutes"],
        params={"line": _TTC_LINES, "year": _TTC_YEARS},
        max_variants=7,
    ),
    Template(
        key="driver-bus-incident-mix",
        dataset="ttc",
        difficulty=H,
        analysis_type="driver",
        question=(
            "Break down the change in bus delay minutes from {year_a} to {year_b} by incident "
            "type: which incident type drove the biggest change?"
        ),
        sql=(
            "WITH by_inc AS (SELECT incident, "
            "sum(CASE WHEN year(delay_date) = {year_a} THEN min_delay ELSE 0 END) AS minutes_a, "
            "sum(CASE WHEN year(delay_date) = {year_b} THEN min_delay ELSE 0 END) AS minutes_b "
            "FROM ttc_bus_delays WHERE year(delay_date) IN ({year_a}, {year_b}) GROUP BY 1) "
            "SELECT incident, minutes_b - minutes_a AS delta_minutes FROM by_inc "
            "ORDER BY abs(minutes_b - minutes_a) DESC LIMIT 1"
        ),
        expected_tables=["ttc_bus_delays"],
        expected_metrics=["total_delay_minutes"],
        params={"year_a": _TTC_YEARS, "year_b": _TTC_YEARS},
        max_variants=5,
    ),
    Template(
        key="driver-long-delay-share",
        dataset="ttc",
        difficulty=H,
        analysis_type="driver",
        question=(
            "Of all subway delay minutes in {year}, what share came from incidents of "
            "20 minutes or more, and how many such incidents were there?"
        ),
        sql=(
            "SELECT sum(CASE WHEN min_delay >= 20 THEN min_delay ELSE 0 END) / sum(min_delay) "
            "AS long_delay_share, sum(CASE WHEN min_delay >= 20 THEN 1 ELSE 0 END) AS long_delay_count "
            "FROM ttc_subway_delays WHERE year(delay_date) = {year}"
        ),
        expected_tables=["ttc_subway_delays"],
        expected_metrics=["total_delay_minutes", "delay_count"],
        params={"year": _TTC_YEARS},
        max_variants=5,
    ),
    Template(
        key="driver-route-concentration",
        dataset="ttc",
        difficulty=H,
        analysis_type="driver",
        question=("In {year}, how many bus routes account for half of all bus delay minutes?"),
        sql=(
            "WITH r AS (SELECT route, sum(min_delay) AS m FROM ttc_bus_delays "
            "WHERE year(delay_date) = {year} GROUP BY 1), "
            "c AS (SELECT route, m, sum(m) OVER (ORDER BY m DESC, route) AS cum, "
            "sum(m) OVER () AS total FROM r) "
            "SELECT count(*) AS routes_for_half FROM c WHERE cum - m < total / 2"
        ),
        expected_tables=["ttc_bus_delays"],
        expected_metrics=["total_delay_minutes"],
        params={"year": _TTC_YEARS},
        max_variants=5,
    ),
]

# ---------- NFIP claims ----------

_NFIP_STATES = "SELECT state FROM nfip_claims GROUP BY state ORDER BY count(*) DESC LIMIT 7"
_NFIP_YEARS = "SELECT DISTINCT year_of_loss FROM nfip_claims WHERE year_of_loss BETWEEN 2015 AND 2024 ORDER BY 1"
_NFIP_ZONES = (
    "SELECT rated_flood_zone FROM nfip_claims WHERE rated_flood_zone IS NOT NULL "
    "GROUP BY 1 ORDER BY count(*) DESC LIMIT 5"
)
_NFIP_EVENTS = (
    "SELECT flood_event FROM nfip_claims WHERE flood_event IS NOT NULL "
    "GROUP BY 1 ORDER BY count(*) DESC LIMIT 5"
)

NFIP_TEMPLATES: list[Template] = [
    # easy lookups
    Template(
        key="claim-count-state-year",
        dataset="nfip",
        difficulty=E,
        analysis_type="lookup",
        question="How many NFIP claims were filed in {state} for losses in {year}?",
        sql=(
            "SELECT count(*) AS claim_count FROM nfip_claims "
            "WHERE state = '{state}' AND year_of_loss = {year}"
        ),
        expected_tables=["nfip_claims"],
        expected_metrics=["claim_count"],
        params={"state": _NFIP_STATES, "year": _NFIP_YEARS},
        max_variants=5,
    ),
    Template(
        key="total-paid-state",
        dataset="nfip",
        difficulty=E,
        analysis_type="lookup",
        question="What is the total amount paid on NFIP claims in {state}?",
        sql="SELECT sum(total_paid) AS total_paid FROM nfip_claims WHERE state = '{state}'",
        expected_tables=["nfip_claims"],
        expected_metrics=["total_paid"],
        params={"state": _NFIP_STATES},
        max_variants=4,
    ),
    Template(
        key="avg-severity-year",
        dataset="nfip",
        difficulty=E,
        analysis_type="lookup",
        question="What was the average claim payment (severity) for losses in {year}?",
        sql=(
            "SELECT avg(total_paid) AS avg_claim_severity FROM nfip_claims "
            "WHERE year_of_loss = {year} AND total_paid > 0"
        ),
        expected_tables=["nfip_claims"],
        expected_metrics=["avg_claim_severity"],
        params={"year": _NFIP_YEARS},
        max_variants=4,
    ),
    Template(
        key="max-building-paid-state",
        dataset="nfip",
        difficulty=E,
        analysis_type="lookup",
        question="What was the largest single building claim payment in {state}?",
        sql=(
            "SELECT max(amount_paid_on_building_claim) AS max_building_paid FROM nfip_claims "
            "WHERE state = '{state}'"
        ),
        expected_tables=["nfip_claims"],
        params={"state": _NFIP_STATES},
        max_variants=3,
    ),
    Template(
        key="claims-by-zone-count",
        dataset="nfip",
        difficulty=E,
        analysis_type="lookup",
        question="How many claims were in rated flood zone {zone}?",
        sql="SELECT count(*) AS claim_count FROM nfip_claims WHERE rated_flood_zone = '{zone}'",
        expected_tables=["nfip_claims"],
        expected_metrics=["claim_count"],
        params={"zone": _NFIP_ZONES},
        max_variants=4,
    ),
    Template(
        key="event-total-paid",
        dataset="nfip",
        difficulty=E,
        analysis_type="lookup",
        question="How much was paid in total for claims tied to the flood event '{event}'?",
        sql="SELECT sum(total_paid) AS total_paid FROM nfip_claims WHERE flood_event = '{event}'",
        expected_tables=["nfip_claims"],
        expected_metrics=["total_paid"],
        params={"event": _NFIP_EVENTS},
        max_variants=4,
    ),
    Template(
        key="state-name-claims",
        dataset="nfip",
        difficulty=E,
        analysis_type="lookup",
        question="Which state (by full name) had the most claims in {year}?",
        sql=(
            "SELECT s.state_name, count(*) AS claim_count FROM nfip_claims c "
            "JOIN nfip_states s ON c.state = s.state WHERE c.year_of_loss = {year} "
            "GROUP BY 1 ORDER BY 2 DESC LIMIT 1"
        ),
        expected_tables=["nfip_claims", "nfip_states"],
        expected_metrics=["claim_count"],
        params={"year": _NFIP_YEARS},
        max_variants=4,
    ),
    Template(
        key="median-water-depth-state",
        dataset="nfip",
        difficulty=E,
        analysis_type="lookup",
        question="What is the median reported water depth for claims in {state}?",
        sql=(
            "SELECT median(water_depth) AS median_water_depth FROM nfip_claims "
            "WHERE state = '{state}' AND water_depth IS NOT NULL"
        ),
        expected_tables=["nfip_claims"],
        expected_metrics=["median_water_depth"],
        params={"state": _NFIP_STATES},
        max_variants=3,
    ),
    # medium trends / comparisons
    Template(
        key="yearly-paid-trend-state",
        dataset="nfip",
        difficulty=M,
        analysis_type="trend",
        question="How has the total paid on claims in {state} trended by year of loss?",
        sql=(
            "SELECT year_of_loss, sum(total_paid) AS total_paid FROM nfip_claims "
            "WHERE state = '{state}' AND year_of_loss >= 2015 GROUP BY 1 ORDER BY 1"
        ),
        expected_tables=["nfip_claims"],
        expected_metrics=["total_paid"],
        params={"state": _NFIP_STATES},
        max_variants=6,
    ),
    Template(
        key="yearly-claim-count-trend",
        dataset="nfip",
        difficulty=M,
        analysis_type="trend",
        question="How did the number of claims per year of loss change from 2015 onward?",
        sql=(
            "SELECT year_of_loss, count(*) AS claim_count FROM nfip_claims "
            "WHERE year_of_loss >= 2015 GROUP BY 1 ORDER BY 1"
        ),
        expected_tables=["nfip_claims"],
        expected_metrics=["claim_count"],
        max_variants=3,
    ),
    Template(
        key="monthly-claims-year",
        dataset="nfip",
        difficulty=M,
        analysis_type="trend",
        question="Which month of {year} had the most claims by date of loss, and how many?",
        sql=(
            "SELECT strftime(date_of_loss, '%Y-%m') AS month, count(*) AS claim_count "
            "FROM nfip_claims WHERE year_of_loss = {year} GROUP BY 1 ORDER BY 2 DESC LIMIT 1"
        ),
        expected_tables=["nfip_claims"],
        expected_metrics=["claim_count"],
        params={"year": _NFIP_YEARS},
        max_variants=5,
    ),
    Template(
        key="state-compare-severity",
        dataset="nfip",
        difficulty=M,
        analysis_type="comparison",
        question="Compare average claim severity in {state_a} versus {state_b} for losses in {year}.",
        sql=(
            "SELECT avg(CASE WHEN state = '{state_a}' THEN total_paid END) AS severity_a, "
            "avg(CASE WHEN state = '{state_b}' THEN total_paid END) AS severity_b "
            "FROM nfip_claims WHERE year_of_loss = {year} AND total_paid > 0"
        ),
        expected_tables=["nfip_claims"],
        expected_metrics=["avg_claim_severity"],
        params={"state_a": _NFIP_STATES, "state_b": _NFIP_STATES, "year": _NFIP_YEARS},
        max_variants=6,
    ),
    Template(
        key="sfha-vs-non-sfha",
        dataset="nfip",
        difficulty=M,
        analysis_type="comparison",
        question=(
            "In {state}, compare average claim payment for properties in Special Flood Hazard "
            "Areas (zones starting with A or V) versus outside them."
        ),
        sql=(
            "SELECT avg(CASE WHEN rated_flood_zone LIKE 'A%' OR rated_flood_zone LIKE 'V%' "
            "THEN total_paid END) AS sfha_avg_paid, "
            "avg(CASE WHEN NOT (rated_flood_zone LIKE 'A%' OR rated_flood_zone LIKE 'V%') "
            "THEN total_paid END) AS non_sfha_avg_paid "
            "FROM nfip_claims WHERE state = '{state}' AND rated_flood_zone IS NOT NULL "
            "AND total_paid > 0"
        ),
        expected_tables=["nfip_claims"],
        expected_metrics=["avg_claim_severity", "sfha_claim_share"],
        params={"state": _NFIP_STATES},
        max_variants=5,
    ),
    Template(
        key="yoy-total-paid",
        dataset="nfip",
        difficulty=M,
        analysis_type="comparison",
        question="How did total claims paid change from {year_a} to {year_b}?",
        sql=(
            "SELECT sum(CASE WHEN year_of_loss = {year_a} THEN total_paid END) AS paid_a, "
            "sum(CASE WHEN year_of_loss = {year_b} THEN total_paid END) AS paid_b, "
            "sum(CASE WHEN year_of_loss = {year_b} THEN total_paid END) - "
            "sum(CASE WHEN year_of_loss = {year_a} THEN total_paid END) AS change_paid "
            "FROM nfip_claims"
        ),
        expected_tables=["nfip_claims"],
        expected_metrics=["total_paid"],
        params={"year_a": _NFIP_YEARS, "year_b": _NFIP_YEARS},
        max_variants=5,
    ),
    Template(
        key="elevated-vs-not",
        dataset="nfip",
        difficulty=M,
        analysis_type="comparison",
        question="Do elevated buildings have lower average claim payments than non-elevated ones in {state}?",
        sql=(
            "SELECT avg(CASE WHEN elevated_building_indicator THEN total_paid END) AS elevated_avg_paid, "
            "avg(CASE WHEN NOT elevated_building_indicator THEN total_paid END) AS non_elevated_avg_paid "
            "FROM nfip_claims WHERE state = '{state}' AND total_paid > 0 "
            "AND elevated_building_indicator IS NOT NULL"
        ),
        expected_tables=["nfip_claims"],
        expected_metrics=["avg_claim_severity"],
        params={"state": _NFIP_STATES},
        max_variants=5,
    ),
    # hard driver analyses
    Template(
        key="driver-zone-change",
        dataset="nfip",
        difficulty=H,
        analysis_type="driver",
        question=(
            "Which rated flood zone contributed most to the change in total paid "
            "from {year_a} to {year_b}, and by how much?"
        ),
        sql=(
            "WITH by_zone AS (SELECT rated_flood_zone, "
            "sum(CASE WHEN year_of_loss = {year_a} THEN total_paid ELSE 0 END) AS paid_a, "
            "sum(CASE WHEN year_of_loss = {year_b} THEN total_paid ELSE 0 END) AS paid_b "
            "FROM nfip_claims WHERE year_of_loss IN ({year_a}, {year_b}) "
            "AND rated_flood_zone IS NOT NULL GROUP BY 1) "
            "SELECT rated_flood_zone, paid_b - paid_a AS delta_paid FROM by_zone "
            "ORDER BY abs(paid_b - paid_a) DESC LIMIT 1"
        ),
        expected_tables=["nfip_claims"],
        expected_metrics=["total_paid"],
        params={"year_a": _NFIP_YEARS, "year_b": _NFIP_YEARS},
        max_variants=6,
    ),
    Template(
        key="driver-state-share-year",
        dataset="nfip",
        difficulty=H,
        analysis_type="driver",
        question="Which state accounted for the largest share of total paid in {year}, and what share?",
        sql=(
            "WITH tot AS (SELECT sum(total_paid) AS t FROM nfip_claims WHERE year_of_loss = {year}) "
            "SELECT state, sum(total_paid) AS state_paid, sum(total_paid) / tot.t AS share "
            "FROM nfip_claims, tot WHERE year_of_loss = {year} GROUP BY state, tot.t "
            "ORDER BY share DESC LIMIT 1"
        ),
        expected_tables=["nfip_claims"],
        expected_metrics=["total_paid"],
        params={"year": _NFIP_YEARS},
        max_variants=6,
    ),
    Template(
        key="driver-severity-vs-frequency",
        dataset="nfip",
        difficulty=H,
        analysis_type="driver",
        question=(
            "Was the change in total paid in {state} from {year_a} to {year_b} driven more by "
            "claim count or by average severity? Give both ratios."
        ),
        sql=(
            "WITH y AS (SELECT year_of_loss, count(*) AS n, avg(total_paid) AS sev "
            "FROM nfip_claims WHERE state = '{state}' AND year_of_loss IN ({year_a}, {year_b}) "
            "AND total_paid > 0 GROUP BY 1) "
            "SELECT max(CASE WHEN year_of_loss = {year_b} THEN n END) * 1.0 / "
            "max(CASE WHEN year_of_loss = {year_a} THEN n END) AS count_ratio, "
            "max(CASE WHEN year_of_loss = {year_b} THEN sev END) / "
            "max(CASE WHEN year_of_loss = {year_a} THEN sev END) AS severity_ratio FROM y"
        ),
        expected_tables=["nfip_claims"],
        expected_metrics=["claim_count", "avg_claim_severity"],
        params={"state": _NFIP_STATES, "year_a": _NFIP_YEARS, "year_b": _NFIP_YEARS},
        max_variants=7,
    ),
    Template(
        key="driver-loss-ratio-zone",
        dataset="nfip",
        difficulty=H,
        analysis_type="driver",
        question=(
            "In {state}, which rated flood zone has the highest building loss ratio "
            "(building paid over building coverage), among zones with at least 100 claims?"
        ),
        sql=(
            "SELECT rated_flood_zone, sum(amount_paid_on_building_claim) / "
            "sum(total_building_insurance_coverage) AS building_loss_ratio, count(*) AS claim_count "
            "FROM nfip_claims WHERE state = '{state}' AND total_building_insurance_coverage > 0 "
            "AND rated_flood_zone IS NOT NULL GROUP BY 1 HAVING count(*) >= 100 "
            "ORDER BY 2 DESC LIMIT 1"
        ),
        expected_tables=["nfip_claims"],
        expected_metrics=["building_loss_ratio", "claim_count"],
        params={"state": _NFIP_STATES},
        max_variants=6,
    ),
    Template(
        key="driver-event-share",
        dataset="nfip",
        difficulty=H,
        analysis_type="driver",
        question=(
            "What share of all claims paid in {year} came from the single largest named flood "
            "event, and which event was it?"
        ),
        sql=(
            "WITH tot AS (SELECT sum(total_paid) AS t FROM nfip_claims WHERE year_of_loss = {year}) "
            "SELECT flood_event, sum(total_paid) / tot.t AS event_share FROM nfip_claims, tot "
            "WHERE year_of_loss = {year} AND flood_event IS NOT NULL GROUP BY flood_event, tot.t "
            "ORDER BY event_share DESC LIMIT 1"
        ),
        expected_tables=["nfip_claims"],
        expected_metrics=["total_paid"],
        params={"year": _NFIP_YEARS},
        max_variants=6,
    ),
]

NFIP_TEMPLATES += [
    Template(
        key="community-most-claims-state",
        dataset="nfip",
        difficulty=E,
        analysis_type="lookup",
        question="Which NFIP community in {state} has the most claims, and how many?",
        sql=(
            "SELECT m.community_name, count(*) AS claim_count FROM nfip_claims c "
            "JOIN nfip_communities m ON c.nfip_community_number_current = m.community_id_number "
            "WHERE c.state = '{state}' GROUP BY 1 ORDER BY 2 DESC LIMIT 1"
        ),
        expected_tables=["nfip_claims", "nfip_communities"],
        expected_metrics=["claim_count", "claims_per_community"],
        params={"state": _NFIP_STATES},
        max_variants=4,
    ),
    Template(
        key="driver-community-share-state-year",
        dataset="nfip",
        difficulty=H,
        analysis_type="driver",
        question=(
            "In {state} for losses in {year}, which community accounted for the largest share "
            "of total paid, and what was that share?"
        ),
        sql=(
            "WITH tot AS (SELECT sum(total_paid) AS t FROM nfip_claims "
            "WHERE state = '{state}' AND year_of_loss = {year}) "
            "SELECT m.community_name, sum(c.total_paid) / tot.t AS share FROM nfip_claims c "
            "JOIN nfip_communities m ON c.nfip_community_number_current = m.community_id_number, tot "
            "WHERE c.state = '{state}' AND c.year_of_loss = {year} "
            "GROUP BY m.community_name, tot.t ORDER BY share DESC LIMIT 1"
        ),
        expected_tables=["nfip_claims", "nfip_communities"],
        expected_metrics=["total_paid"],
        params={"state": _NFIP_STATES, "year": _NFIP_YEARS},
        max_variants=6,
    ),
]

TEMPLATE_SETS: dict[str, list[Template]] = {
    "demo": DEMO_TEMPLATES,
    "ttc": TTC_TEMPLATES,
    "nfip": NFIP_TEMPLATES,
}
