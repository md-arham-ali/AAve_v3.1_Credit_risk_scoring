-- =====================================================================
-- Query 9 / Extra — Collateral usage toggle metrics
--   (covers Preferred_metrics: collateral_enabled_user / collateral_enabled_asset)
-- Grain / composite key : (time_bucket, asset)
-- Sources               : aave_v3_ethereum.pool_evt_reserveusedascollateralenabled
--                         aave_v3_ethereum.pool_evt_reserveusedascollateraldisabled
-- Window                : 2025-11-01 00:00 (incl) .. 2026-02-01 00:00 (excl), UTC
-- Bucket                : fixed 6-hour slots (00 / 06 / 12 / 18 UTC)
--
-- Credit-optimization notes
--   * Both event tables folded into ONE query via UNION ALL -> single GROUP BY.
--   * Partition-pruned on evt_block_date.
--   * `user` (wallet) used only inside approx_distinct(); not exported raw.
--   * Pure activity counts — these are leading indicators of collateral structure
--     changes that downstream risk models can use.
--   * asset_symbol — on-chain token symbol from tokens.erc20 (blockchain =
--     'ethereum'). Resolved via a LEFT JOIN applied AFTER aggregation, on the small
--     per-bucket result set, so the heavy event scan is unchanged; assets missing
--     from tokens.erc20 get symbol = NULL. The symbol is a readability/QA label —
--     keep joining on `asset` (address) downstream, not on the symbol.
-- =====================================================================
WITH events AS (
    SELECT reserve AS asset, evt_block_time, evt_block_number, "user" AS actor, 'enabled' AS kind
    FROM aave_v3_ethereum.pool_evt_reserveusedascollateralenabled
    WHERE evt_block_date >= DATE '2025-11-01' AND evt_block_date < DATE '2026-02-01'
    UNION ALL
    SELECT reserve AS asset, evt_block_time, evt_block_number, "user" AS actor, 'disabled' AS kind
    FROM aave_v3_ethereum.pool_evt_reserveusedascollateraldisabled
    WHERE evt_block_date >= DATE '2025-11-01' AND evt_block_date < DATE '2026-02-01'
),
agg AS (
    SELECT
        date_add('hour',
                 6 * CAST(floor(hour(evt_block_time) / 6) AS bigint),
                 date_trunc('day', evt_block_time))                                AS time_bucket,
        asset,
        COUNT(CASE WHEN kind = 'enabled'  THEN 1 END)                             AS collateral_enabled_count,
        COUNT(CASE WHEN kind = 'disabled' THEN 1 END)                             AS collateral_disabled_count,
        approx_distinct(CASE WHEN kind = 'enabled'  THEN actor END)               AS unique_collateral_enable_users,
        approx_distinct(CASE WHEN kind = 'disabled' THEN actor END)               AS unique_collateral_disable_users,
        MAX(evt_block_number)                                                     AS latest_collateral_toggle_block
    FROM events
    GROUP BY 1, 2
)
SELECT
    agg.time_bucket,
    agg.asset,
    tok.symbol                                                                    AS asset_symbol,
    agg.collateral_enabled_count,
    agg.collateral_disabled_count,
    agg.unique_collateral_enable_users,
    agg.unique_collateral_disable_users,
    agg.latest_collateral_toggle_block
FROM agg
LEFT JOIN tokens.erc20 tok
       ON tok.blockchain = 'ethereum'
      AND tok.contract_address = agg.asset
ORDER BY 1, 2
