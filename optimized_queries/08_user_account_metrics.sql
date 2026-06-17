-- =====================================================================
-- Query 8 / Part 6 — User account state metrics (market-wide, per bucket)
-- Grain / primary key   : time_bucket  (NO asset dimension)
-- Source                : aave_v3_ethereum.pool_call_getuseraccountdata  (CALL table)
-- Window                : 2025-11-01 00:00 (incl) .. 2026-02-01 00:00 (excl), UTC
-- Bucket                : fixed 6-hour slots (00 / 06 / 12 / 18 UTC)
--
-- Credit-optimization notes
--   * Single CALL table, single scan, single GROUP BY; pruned on call_block_date.
--   * Wallet column is `user` (NOT `input_user`); used only inside approx_distinct().
--   * Only successful calls are aggregated (call_success).
--   * Raw aggregates only: avg_health_factor is DERIVED downstream, so we expose
--     MIN/MAX of healthFactor instead.
--   * Decimal notes (decimal_reference_part6.sql):
--       *_base columns        -> USD base currency, 8 decimals
--       *_liquidation_threshold / *_ltv -> basis points, 4 decimals (1e4)
--       health factor         -> WAD, 18 decimals (1e18 = HF 1.0). No-debt accounts
--                                report healthFactor = uint256 max (handle downstream).
-- =====================================================================
SELECT
    date_add('hour',
             6 * CAST(floor(hour(call_block_time) / 6) AS bigint),
             date_trunc('day', call_block_time))                               AS time_bucket,
    AVG(CAST(output_totalCollateralBase        AS double))                    AS avg_total_collateral_base,
    AVG(CAST(output_totalDebtBase              AS double))                    AS avg_total_debt_base,
    AVG(CAST(output_availableBorrowsBase       AS double))                    AS avg_available_borrows_base,
    AVG(CAST(output_currentLiquidationThreshold AS double))                   AS avg_current_liquidation_threshold,
    AVG(CAST(output_ltv                        AS double))                    AS avg_ltv,
    MIN(output_healthFactor)                                                  AS min_health_factor,
    MAX(output_healthFactor)                                                  AS max_health_factor,
    approx_distinct("user")                                                   AS sampled_user_count,
    COUNT(*)                                                                  AS account_data_call_count
FROM aave_v3_ethereum.pool_call_getuseraccountdata
WHERE call_block_date >= DATE '2025-11-01'
  AND call_block_date <  DATE '2026-02-01'
  AND call_success
GROUP BY 1
ORDER BY 1
