-- =====================================================================
-- Query 5 / Part 4B — Liquidation metrics
-- Grain / composite key : (time_bucket, asset)
-- Source                : aave_v3_ethereum.pool_evt_liquidationcall
-- Window                : 2025-11-01 00:00 (incl) .. 2026-02-01 00:00 (excl), UTC
-- Bucket                : fixed 6-hour slots (00 / 06 / 12 / 18 UTC)
--
-- WHY THIS IS PER-ASSET (was (time_bucket, collateral_asset, debt_asset))
--   A liquidation touches TWO assets: collateral seized + debt covered. The old
--   3-column grain could NOT join the rest of the panel (everything else is keyed
--   (time_bucket, asset)) and could NOT be decimal-scaled (one row carried two
--   amounts in two different tokens). This query re-keys to (time_bucket, asset) by
--   emitting each liquidation as TWO legs — a collateral leg (asset = collateralAsset)
--   and a debt leg (asset = debtAsset) — then aggregating per asset. Each amount is
--   now denominated in ITS OWN row asset's token units, so `scale_by_decimals`
--   (keyed on asset) is correct, and the table joins on (time_bucket, asset).
--
-- Credit-optimization notes
--   * Single source table; partition-pruned on evt_block_date. The UNION ALL is two
--     projections of the SAME pruned scan (collateral leg + debt leg), one GROUP BY.
--   * Wallet fields (user, liquidator) NEVER leave the query as raw output — consumed
--     only inside approx_distinct() HLL counters (deduped across the 2 legs of an
--     event, so each distinct actor is counted once per asset).
--   * Amounts COALESCE to 0 (not NULL): a row exists because the asset took part in a
--     liquidation; "0 seized as collateral" / "0 debt covered" is the true value for
--     the role the asset did NOT play. Counts are 0, never NULL, by construction.
--   * asset_symbol — on-chain token symbol from tokens.erc20 (blockchain = 'ethereum'),
--     one LEFT JOIN after aggregation on the small per-bucket result; assets absent
--     from tokens.erc20 get symbol = NULL. Readability/QA label — keep joining on the
--     `asset` address downstream.
--
-- Columns
--   liquidated_collateral_raw    Σ liquidatedCollateralAmount where asset = collateral
--                                (collateral-asset token units → see decimal_reference_part4)
--   liquidation_debt_covered_raw Σ debtToCover           where asset = debt
--                                (debt-asset token units)
--   as_collateral_tx_count       # liquidation legs where this asset was the collateral
--   as_debt_tx_count             # liquidation legs where this asset was the debt
--   liquidation_tx_count         total legs touching this asset (= collateral + debt)
--   receive_atoken_count         collateral legs where the liquidator took aTokens
--   unique_liquidated_users      approx-distinct borrowers liquidated (HLL)
--   unique_liquidators           approx-distinct liquidators (HLL)
--   latest_liquidation_block     end-of-period block marker
-- =====================================================================
WITH legs AS (
    -- collateral leg: this asset was SEIZED as collateral
    SELECT
        date_add('hour',
                 6 * CAST(floor(hour(evt_block_time) / 6) AS bigint),
                 date_trunc('day', evt_block_time))                                AS time_bucket,
        collateralAsset                                                            AS asset,
        'collateral'                                                               AS role,
        liquidatedCollateralAmount                                                 AS collateral_seized,
        CAST(NULL AS uint256)                                                      AS debt_covered,
        receiveAToken,
        "user",
        liquidator,
        evt_block_number
    FROM aave_v3_ethereum.pool_evt_liquidationcall
    WHERE evt_block_date >= DATE '2025-11-01'
      AND evt_block_date <  DATE '2026-02-01'

    UNION ALL

    -- debt leg: this asset's debt was COVERED (repaid) by the liquidator
    SELECT
        date_add('hour',
                 6 * CAST(floor(hour(evt_block_time) / 6) AS bigint),
                 date_trunc('day', evt_block_time))                                AS time_bucket,
        debtAsset                                                                  AS asset,
        'debt'                                                                     AS role,
        CAST(NULL AS uint256)                                                      AS collateral_seized,
        debtToCover                                                                AS debt_covered,
        receiveAToken,
        "user",
        liquidator,
        evt_block_number
    FROM aave_v3_ethereum.pool_evt_liquidationcall
    WHERE evt_block_date >= DATE '2025-11-01'
      AND evt_block_date <  DATE '2026-02-01'
),
agg AS (
    SELECT
        time_bucket,
        asset,
        COALESCE(SUM(collateral_seized), uint256 '0')                              AS liquidated_collateral_raw,
        COALESCE(SUM(debt_covered),      uint256 '0')                              AS liquidation_debt_covered_raw,
        COUNT(CASE WHEN role = 'collateral' THEN 1 END)                            AS as_collateral_tx_count,
        COUNT(CASE WHEN role = 'debt'       THEN 1 END)                            AS as_debt_tx_count,
        COUNT(*)                                                                   AS liquidation_tx_count,
        COUNT(CASE WHEN role = 'collateral' AND receiveAToken THEN 1 END)          AS receive_atoken_count,
        approx_distinct("user")                                                    AS unique_liquidated_users,
        approx_distinct(liquidator)                                                AS unique_liquidators,
        MAX(evt_block_number)                                                      AS latest_liquidation_block
    FROM legs
    GROUP BY 1, 2
)
SELECT
    agg.time_bucket,
    agg.asset,
    tok.symbol                                                                     AS asset_symbol,
    agg.liquidated_collateral_raw,
    agg.liquidation_debt_covered_raw,
    agg.as_collateral_tx_count,
    agg.as_debt_tx_count,
    agg.liquidation_tx_count,
    agg.receive_atoken_count,
    agg.unique_liquidated_users,
    agg.unique_liquidators,
    agg.latest_liquidation_block
FROM agg
LEFT JOIN tokens.erc20 tok
       ON tok.blockchain = 'ethereum'
      AND tok.contract_address = agg.asset
ORDER BY 1, 2
