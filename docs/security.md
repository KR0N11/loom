# Security notes

## Threat model

The untrusted input is model-written SQL and pandas code produced from a user
question. Goals: it must not read or write the host filesystem beyond the
read-only DuckDB file, must not reach the network, and must not consume
unbounded CPU, memory or time.

## What each sandbox backend guarantees

| Backend | Filesystem | Network | CPU / memory | Time | Use |
|---|---|---|---|---|---|
| `docker` | Read-only root, DuckDB mounted read-only | `--network none` | `--cpus`, `--memory`, `--pids-limit` | Parent timeout, container removed | Production on a container host |
| `lambda` | Lambda `/tmp` only, S3 read via IAM | Only what the function's VPC / IAM allows | Function memory size | Function timeout | Production on AWS |
| `subprocess` | Host filesystem as the running user (only the runner's own allow-list and restricted builtins stand in the way) | Best effort: proxy variables removed, `unshare -n` on Linux when available; no isolation on macOS | Memory rlimit on Linux only | Parent timeout | Development and CI only |

The subprocess backend is a convenience, not a security boundary. Do not
deploy it for untrusted users.

## Runner allow-list (applies inside every backend)

- SQL: one statement only, after comment stripping. Anything starting with
  `INSERT`, `UPDATE`, `DELETE`, `CREATE`, `DROP`, `ALTER`, `COPY`, `ATTACH`,
  `INSTALL`, `LOAD`, `PRAGMA`, `SET`, `EXPORT` or `IMPORT` is refused before
  it reaches DuckDB. The database is opened read-only. Results are wrapped in
  a `LIMIT row_limit + 1` so truncation is detected and reported.
- pandas: the code runs with a restricted namespace (`pd`, the requested
  tables as DataFrames, a safe subset of builtins). `open`, `__import__`,
  `exec`, `eval`, `compile`, `input` and `globals` are absent. The code must
  assign a DataFrame to `result`.

## Secrets

No secrets are committed. `.env.example` lists every variable; `.env` is
ignored by git. In AWS, the Anthropic key and the database password live in
Secrets Manager (see `infra/terraform`). CI reads `ANTHROPIC_API_KEY` from
repository secrets and only runs the eval job when it is present.

## Data

Both datasets are public. The NFIP claims slice contains no direct personal
identifiers beyond what OpenFEMA publishes (zip code, county). Memos and
query logs are written under `outputs/`, which is git-ignored.
