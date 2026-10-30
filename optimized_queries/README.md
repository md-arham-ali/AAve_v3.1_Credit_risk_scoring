# optimized_queries — Aave V3.1 raw extraction (Dune, credit-optimized)

DuneSQL queries that extract **raw** Aave V3 (Ethereum) variables for downstream ML/DL credit-risk
modelling. No derived/risk metrics are computed in SQL — only raw numerics, timestamps, and
numerically-encoded categoricals. See `VERIFICATION.md` for the metric↔table↔field audit and
corrections.

## Files

Data queries (one logical table each):

| # | File | Table grain | Source(s) |
|---|------|-------------|-----------|
| 1 | `01_supply_withdraw_metrics.sql` | (time_bucket, asset) | pool_evt_supply, pool_evt_withdraw |
| 2 | `02_borrow_repay_metrics.sql` | (time_bucket, asset) | pool_evt_borrow, pool_evt_repay |
| 3 | `03_reserve_state_rates.sql` | (time_bucket, asset) | pool_evt_reservedataupdated |
| 4 | `04_reserve_config_metrics.sql` | (time_bucket, asset) | 5 × poolconfigurator_evt_* |
| 5 | `05_liquidation_metrics.sql` | (time_bucket, asset) | pool_evt_liquidationcall |
| 6 | `06_flashloan_metrics.sql` | (time_bucket, asset) | pool_evt_flashloan |
| 7 | `07_oracle_price_metrics.sql` | (time_bucket, asset) | aaveoracle_call_getassetprice |
| 8 | `08_user_account_metrics.sql` | (time_bucket) | pool_call_getuseraccountdata |
| 9 | `09_collateral_toggle_metrics.sql` | (time_bucket, asset) | pool_evt_reserveusedascollateral{enabled,disabled} |

Decimal references (one per part, schema `asset, asset_symbol, metric, decimals, unit`):
`decimal_reference_part1.sql` … `part6`, `part9_collateral_toggle`.

Run order is independent — each query stands alone. All joins happen later in Python on
`(time_bucket, asset)` (Part 6 joins on `time_bucket` only).

## Bucket width — `{{bucket_hours}}`

Parameterised 2026-09-04. Every query buckets its events with the same expression:

```sql
date_add('hour',
         {{bucket_hours}} * CAST(floor(hour(<time_col>) / {{bucket_hours}}) AS bigint),
         date_trunc('day', <time_col>))                          AS time_bucket
```

**Allowed widths: 1, 2, 3, 4, 6, 8, 12, 24 — the divisors of 24, and nothing else.** The
formula restarts at every midnight (`date_trunc('day', …)` is its anchor), while `04`'s grid
CTE strides continuously from `{{start_date}} 00:00`. The two agree only when the width
divides 24. At width 5 the formula's last bucket of each day is short and the grid drifts to
01:00, 06:00, … — `reserve_config` then keys on timestamps no other table produces and joins
to nothing, with no error raised anywhere.

On Dune, declare it as name `bucket_hours`, Type **List**, values `1, 2, 3, 4, 6, 8, 12, 24`,
default `2`; the dropdown is what makes a bad width unselectable. Type **Number** also
substitutes correctly but drops that guard. Client-side the same rule is enforced by
`src/dune_fetch.VALID_BUCKET_HOURS`, which rejects a bad width *before* the execute call, so
it costs no credits.

### Where it appears — 11 lines across the 9 files

| File | Line(s) | Note |
|------|---------|------|
| `01_supply_withdraw_metrics.sql` | 31 | |
| `02_borrow_repay_metrics.sql` | 33 | |
| `03_reserve_state_rates.sql` | 4 | |
| `04_reserve_config_metrics.sql` | 25, 37, 57 | **three** spots — see below |
| `05_liquidation_metrics.sql` | 26 | uses `EXTRACT(hour FROM …)`, not `hour(…)` |
| `06_flashloan_metrics.sql` | 4 | |
| `07_oracle_price_metrics.sql` | 92 | |
| `08_user_account_metrics.sql` | 3 | column is `call_block_time` |
| `09_collateral_toggle_metrics.sql` | 12 | |

`04` is the exception: two copies of the formula (`ev_b` at 25, `chg` at 57) **plus** the grid
spine at 37, which takes `INTERVAL '1' HOUR * {{bucket_hours}}` instead of the formula. A
find-and-replace across the folder misses both `04`'s second copy and `05`'s `EXTRACT` form —
that is why this table lists lines rather than saying "one per file".

### Mixed widths are undetectable downstream

All nine tables join on `(time_bucket, asset)`. If one runs at a width the others don't, the
join just yields fewer rows — nothing raises, and the panel looks structurally fine. Change
the width for **all nine at once, or not at all**. `notebooks/process.ipynb` tracks which
queries have the parameter live on Dune in its `BUCKET_PARAM_ON_DUNE` set, withholds
`bucket_hours` from the rest (Dune ignores an undeclared parameter rather than rejecting it,
so sending it blindly would look like it worked), and prints the split on every run.

Two other bucket widths legitimately exist and are **not** this parameter:

- `05` lines 7/16 pre-truncate each liquidation leg to the **hour** before re-binning at line
  26. Correct at any whole-hour width — leave it alone.
- `07` reads `prices.hour`, an **hourly** feed. `price_points` counts observations per bucket,
  so it scales with the width on its own (≤ `bucket_hours`).

## Credit-optimization strategy (per Dune `writing-efficient-queries`)

1. **Partition pruning.** Every query filters the partition column
   (`evt_block_date` / `call_block_date`) to the 3 target months. This is the single biggest
   lever on data scanned → credits. The exact `2025-11-01 ≤ date < 2026-02-01` window aligns to
   day boundaries, so date-level pruning is exact.
2. **One scan per table, one GROUP BY.** Tables sharing a grain are folded into a single query
   via `UNION ALL` + conditional aggregation (supply/withdraw, borrow/repay, the 5 config tables,
   collateral toggle). No re-scans, no self-joins, no cross-part joins.
3. **Project only needed columns** — never `SELECT *`. Decoded event tables are columnar; fewer
   columns = less I/O.
4. **`approx_distinct()` (HLL)** for unique-wallet counts instead of exact `COUNT(DISTINCT)` —
   cheaper, and keeps raw wallet values out of the output (rule-compliant).
5. **`max_by(value, key)` for end-of-period state** (rates, indexes, caps, last borrow rate,
   oracle price) — one pass, no window functions / no extra sort stage.
6. **Aggregate at the source.** ~90 days × (24 / `bucket_hours`) buckets/day × N assets is a few
   thousand output rows per table, so result-set export cost is negligible. Narrowing the bucket
   multiplies the row count — width 1 emits 24× the rows of width 24 over the same window.

### Running them
- App: paste each file, **Run**. Pick the smallest engine that finishes (Free/Small is enough for
  these pruned scans); the per-execution credit cost is shown in the run panel.
- API/MCP: create + execute each query, then page results with the results endpoint. Re-running
  the **same saved query** (rather than creating new ones) reuses cached results when inputs are
  unchanged.

### Notes
- `stable_borrow_*` metrics are expected ≈ 0 (stable-rate mode is deprecated on Aave V3 mainnet).
- Health factor for no-debt accounts is `uint256` max — `min/max_health_factor` keep it raw;
  handle the sentinel during preprocessing.
- `04` reserve_config is a forward-filled config **STATE** panel (dense on the bucket grid, current
  value per parameter carried forward from full history; unset params = 0 on-chain default).
  `05` liquidation is per-asset `(time_bucket, asset)` — each liquidation is split into a
  collateral leg + a debt leg so it joins/scales like every other table. Both were previously
  the cause of the ">70% NULL / few rows / can't join" problem and are no longer sparse.
- **`asset_symbol`** — every asset-level query (`01`–`07`, `09`) emits the on-chain token
  symbol from `tokens.erc20` (`blockchain = 'ethereum'`), including `05` (single `asset_symbol`,
  per-asset grain). The lookup is a `LEFT JOIN` applied **after** aggregation (on the small
  per-bucket result), so partition pruning and the heavy event scan are unchanged — added cost is
  just one scan of the small `tokens.erc20` reference (already used by `decimal_reference_*`).
  Assets absent from `tokens.erc20` get `symbol = NULL`. It is a readability/QA label — keep
  joining on `asset` (address) downstream. `08` has no asset dimension, so it has no symbol.
