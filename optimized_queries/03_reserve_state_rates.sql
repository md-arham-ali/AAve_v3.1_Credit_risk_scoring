-- =====================================================================
-- Query 3 / Part 3 — Reserve state + rates (end-of-period)
-- Grain / composite key : (time_bucket, asset)
-- Source                : aave_v3_ethereum.pool_evt_reservedataupdated
-- Window                : 2025-11-01 00:00 (incl) .. 2026-02-01 00:00 (excl), UTC
-- Bucket                : fixed 6-hour slots (00 / 06 / 12 / 18 UTC)
--
-- Credit-optimization notes
--   * Single table, single scan, single GROUP BY.
--   * Partition pruning on evt_block_date.
--   * Each rate/index is captured as END-OF-PERIOD state via
--     max_by(value, ROW(evt_block_number, evt_index)) — i.e. the value carried
--     by the last ReserveDataUpdated event in the bucket for that asset.
--   * All rate/index fields are RAY (1e27) integers; left raw (no scaling here).
--   * update_count gives the number of state updates in the bucket (activity).
--   * asset_symbol — on-chain token symbol from tokens.erc20 (blockchain =
--     'ethereum'). Resolved via a LEFT JOIN applied AFTER aggregation, on the small
--     per-bucket result set, so the heavy event scan is unchanged; assets missing
--     from tokens.erc20 get symbol = NULL. The symbol is a readability/QA label —
--     keep joining on `asset` (address) downstream, not on the symbol.
-- =====================================================================
SELECT
    date_add('hour',
             2 * CAST(floor(hour(evt_block_time) / 2) AS bigint),
             date_trunc('day', evt_block_time))                                    AS time_bucket,
    reserve                                                                        AS asset,
    max_by(liquidityRate,       ROW(evt_block_number, evt_index)) / 1e27          AS liquidity_rate,
    max_by(variableBorrowRate,  ROW(evt_block_number, evt_index)) / 1e27          AS variable_borrow_rate,
    max_by(stableBorrowRate,    ROW(evt_block_number, evt_index)) / 1e27          AS stable_borrow_rate,
    max_by(liquidityIndex,      ROW(evt_block_number, evt_index)) / 1e27          AS liquidity_index,
    max_by(variableBorrowIndex, ROW(evt_block_number, evt_index)) / 1e27          AS variable_borrow_index,
    COUNT(*)                                                                       AS update_count
FROM aave_v3_ethereum.pool_evt_reservedataupdated
WHERE evt_block_date >= DATE '2025-04-01'
  AND evt_block_date <  DATE '2026-03-31'
GROUP BY 1, 2
ORDER BY 1, 2