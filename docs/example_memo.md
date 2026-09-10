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

![Average non-zero delay minutes by line (2024)](chart_1.png)

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

