-- =====================================================================
-- Query 1 / Part 1 — Supply + Withdraw metrics
-- Grain / composite key : (time_bucket, asset)
-- Sources               : aave_v3_ethereum.pool_evt_supply
--                         aave_v3_ethereum.pool_evt_withdraw
-- Window                : 2025-11-01 00:00 (incl) .. 2026-02-01 00:00 (excl), UTC
-- Bucket                : fixed 6-hour slots (00 / 06 / 12 / 18 UTC)
--
-- Credit-optimization notes
--   * Two tables sharing the (time_bucket, asset) key are folded into ONE
--     query via UNION ALL -> single GROUP BY (no join, no second execution).
--   * Partition pruning: WHERE on evt_block_date (the partition column) keeps
--     the scan to the 3 target months only.
--   * Only the columns actually needed are projected from each table.
--   * approx_distinct() (HLL) is used for unique-user counts (cheaper + wallet
--     addresses never leave the query as raw output).
--   * Raw values only — no derived/risk metrics computed in SQL.
--   * Missing buckets are simply absent (no fill); NULL preserved per metric
--     because conditional aggregates return NULL when a side has no rows.
--   * asset_symbol — on-chain token symbol from tokens.erc20 (blockchain =
--     'ethereum'). Resolved via a LEFT JOIN applied AFTER aggregation, on the small
--     per-bucket result set, so the heavy event scan is unchanged; assets missing
--     from tokens.erc20 get symbol = NULL. The symbol is a readability/QA label —
--     keep joining on `asset` (address) downstream, not on the symbol.
-- =====================================================================
WITH events AS (
    SELECT
        reserve            AS asset,
        evt_block_time,
        evt_block_number,
        evt_index,
        amount,
        "user"             AS actor,
        'supply'           AS kind
    FROM aave_v3_ethereum.pool_evt_supply
    WHERE evt_block_date >= DATE '2025-11-01'
      AND evt_block_date <  DATE '2026-02-01'

    UNION ALL

    SELECT
        reserve            AS asset,
        evt_block_time,
        evt_block_number,
        evt_index,
        amount,
        "user"             AS actor,
        'withdraw'         AS kind
    FROM aave_v3_ethereum.pool_evt_withdraw
    WHERE evt_block_date >= DATE '2025-11-01'
      AND evt_block_date <  DATE '2026-02-01'
),
agg AS (
    SELECT
        date_add('hour',
                 6 * CAST(floor(hour(evt_block_time) / 6) AS bigint),
                 date_trunc('day', evt_block_time))                            AS time_bucket,
        asset,
        -- cumulative raw flows in the bucket
        SUM(CASE WHEN kind = 'supply'   THEN amount END)                       AS supply_amount_raw,
        SUM(CASE WHEN kind = 'withdraw' THEN amount END)                       AS withdrawal_amount_raw,
        -- signed net flow: supply inflow (+), withdrawal outflow (-)
        SUM(CASE WHEN kind = 'supply'   THEN  CAST(amount AS int256)
                 WHEN kind = 'withdraw' THEN -CAST(amount AS int256) END)      AS net_supply_flow_raw,
        -- activity counts
        COUNT(CASE WHEN kind = 'supply'   THEN 1 END)                          AS supply_tx_count,
        COUNT(CASE WHEN kind = 'withdraw' THEN 1 END)                          AS withdrawal_tx_count,
        approx_distinct(CASE WHEN kind = 'supply'   THEN actor END)            AS unique_suppliers,
        approx_distinct(CASE WHEN kind = 'withdraw' THEN actor END)            AS unique_withdraw_users,
        -- end-of-period block markers (for downstream ordering / dedup)
        MAX(CASE WHEN kind = 'supply'   THEN evt_block_number END)             AS latest_supply_block,
        MAX(CASE WHEN kind = 'withdraw' THEN evt_block_number END)             AS latest_withdraw_block
    FROM events
    GROUP BY 1, 2
)
SELECT
    agg.time_bucket,
    agg.asset,
    tok.symbol                                                                AS asset_symbol,
    agg.supply_amount_raw,
    agg.withdrawal_amount_raw,
    agg.net_supply_flow_raw,
    agg.supply_tx_count,
    agg.withdrawal_tx_count,
    agg.unique_suppliers,
    agg.unique_withdraw_users,
    agg.latest_supply_block,
    agg.latest_withdraw_block
FROM agg
LEFT JOIN tokens.erc20 tok
       ON tok.blockchain = 'ethereum'
      AND tok.contract_address = agg.asset
ORDER BY 1, 2
