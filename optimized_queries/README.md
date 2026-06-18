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
6. **Aggregate at the source.** ~90 days × 4 buckets/day × N assets is a few thousand output rows
   per table, so result-set export cost is negligible.

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
- `04` reserve_config is a forward-filled config **STATE** panel (dense on the 6h grid, current
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
