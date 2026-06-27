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
WITH agg AS (
    SELECT
        date_add('hour',
                 2 * CAST(floor(hour(evt_block_time) / 2) AS bigint),
                 date_trunc('day', evt_block_time))                                AS time_bucket,
        asset,
        SUM(amount)                                                               AS flashloan_amount_raw,
        SUM(premium)                                                              AS flashloan_premium_raw,
        COUNT(*)                                                                  AS flashloan_tx_count,
        approx_distinct(initiator)                                               AS unique_flashloan_initiators,
        COUNT(CASE WHEN interestRateMode = 0 THEN 1 END)                          AS no_open_debt_flashloan_tx_count,
        COUNT(CASE WHEN interestRateMode = 1 THEN 1 END)                          AS stable_flashloan_tx_count,
        COUNT(CASE WHEN interestRateMode = 2 THEN 1 END)                          AS variable_flashloan_tx_count,
        MAX(evt_block_number)                                                     AS latest_flashloan_block
    FROM aave_v3_ethereum.pool_evt_flashloan
    WHERE evt_block_date >= DATE '2025-04-01'
      AND evt_block_date < DATE '2026-04-01'
    GROUP BY 1, 2
)
SELECT
    agg.time_bucket,
    agg.asset,
    tok.symbol                                                                    AS asset_symbol,
    CAST(agg.flashloan_amount_raw AS DOUBLE) / POW(10, tok.decimals)            AS flashloan_amount,
    CAST(agg.flashloan_premium_raw AS DOUBLE) / POW(10, tok.decimals)           AS flashloan_premium,
    agg.flashloan_tx_count,
    agg.unique_flashloan_initiators,
    agg.no_open_debt_flashloan_tx_count,
    agg.stable_flashloan_tx_count,
    agg.variable_flashloan_tx_count,
    agg.latest_flashloan_block
FROM agg
LEFT JOIN tokens.erc20 tok
       ON tok.blockchain = 'ethereum'
      AND tok.contract_address = agg.asset
ORDER BY 1, 2
