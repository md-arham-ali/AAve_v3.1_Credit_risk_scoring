-- =====================================================================
-- Query 6 / Part 5A — Flashloan metrics
-- Grain / composite key : (time_bucket, asset)
-- Source                : aave_v3_ethereum.pool_evt_flashloan
-- Window                : 2025-11-01 00:00 (incl) .. 2026-02-01 00:00 (excl), UTC
-- Bucket                : fixed 6-hour slots (00 / 06 / 12 / 18 UTC)
--
-- Credit-optimization notes
--   * Single table, single scan, single GROUP BY; partition-pruned on evt_block_date.
--   * initiator (wallet) used only inside approx_distinct(); target (address) excluded.
--   * interestRateMode on a flashloan: 0 = no open debt (loan repaid same tx, the
--     dominant case), 1 = stable, 2 = variable. All three counts are exposed raw.
--   * Raw cumulative sums only — flashloan_volume / fee_revenue are DERIVED downstream.
--   * asset_symbol — on-chain token symbol from tokens.erc20 (blockchain =
--     'ethereum'). Resolved via a LEFT JOIN applied AFTER aggregation, on the small
--     per-bucket result set, so the heavy event scan is unchanged; assets missing
--     from tokens.erc20 get symbol = NULL. The symbol is a readability/QA label —
--     keep joining on `asset` (address) downstream, not on the symbol.
-- =====================================================================
WITH agg AS (
    SELECT
        date_add('hour',
                 6 * CAST(floor(hour(evt_block_time) / 6) AS bigint),
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
    WHERE evt_block_date >= DATE '2025-11-01'
      AND evt_block_date <  DATE '2026-02-01'
    GROUP BY 1, 2
)
SELECT
    agg.time_bucket,
    agg.asset,
    tok.symbol                                                                    AS asset_symbol,
    agg.flashloan_amount_raw,
    agg.flashloan_premium_raw,
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
