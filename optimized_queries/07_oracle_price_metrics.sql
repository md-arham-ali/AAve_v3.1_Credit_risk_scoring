-- =====================================================================
-- Query 7 / Part 5B — Oracle asset price metrics (end-of-period)
-- Grain / composite key : (time_bucket, asset)
-- Source                : aave_v3_ethereum.aaveoracle_call_getassetprice  (CALL table)
-- Window                : 2025-11-01 00:00 (incl) .. 2026-02-01 00:00 (excl), UTC
-- Bucket                : fixed 6-hour slots (00 / 06 / 12 / 18 UTC)
--
-- Credit-optimization notes
--   * CALL table -> time/partition columns are call_block_* (use call_block_date
--     for partition pruning, call_block_time for bucketing).
--   * Only successful calls are kept (call_success) so output_0 is a valid price.
--   * oracle_asset_price = END-OF-PERIOD price via max_by(output_0, key) where key
--     = (call_block_number, call_tx_index) -> the last observed price in the bucket.
--   * `asset` is the input token address (categorical key, not a raw value column).
--   * Aave oracle prices are quoted in the base currency with 8 decimals (see
--     decimal_reference_part5.sql).
--   * asset_symbol — on-chain token symbol from tokens.erc20 (blockchain =
--     'ethereum'). Resolved via a LEFT JOIN applied AFTER aggregation, on the small
--     per-bucket result set, so the heavy call scan is unchanged; assets missing
--     from tokens.erc20 get symbol = NULL. The symbol is a readability/QA label —
--     keep joining on `asset` (address) downstream, not on the symbol.
-- =====================================================================
WITH agg AS (
    SELECT
        date_add('hour',
                 6 * CAST(floor(hour(call_block_time) / 6) AS bigint),
                 date_trunc('day', call_block_time))                               AS time_bucket,
        asset,
        max_by(output_0, ROW(call_block_number, call_tx_index))                   AS oracle_asset_price,
        MAX(call_block_number)                                                    AS latest_oracle_block,
        COUNT(*)                                                                  AS oracle_call_count
    FROM aave_v3_ethereum.aaveoracle_call_getassetprice
    WHERE call_block_date >= DATE '2025-11-01'
      AND call_block_date <  DATE '2026-02-01'
      AND call_success
    GROUP BY 1, 2
)
SELECT
    agg.time_bucket,
    agg.asset,
    tok.symbol                                                                    AS asset_symbol,
    agg.oracle_asset_price,
    agg.latest_oracle_block,
    agg.oracle_call_count
FROM agg
LEFT JOIN tokens.erc20 tok
       ON tok.blockchain = 'ethereum'
      AND tok.contract_address = agg.asset
ORDER BY 1, 2
