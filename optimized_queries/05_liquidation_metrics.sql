-- Query 5 / Part 4B — Liquidation metrics — grain (time_bucket, asset)
-- Each liquidation is split into a collateral leg (asset=collateralAsset) and a debt
-- leg (asset=debtAsset), then aggregated per asset, so it joins/scales like every
-- other (time_bucket, asset) table. Amounts COALESCE to 0; counts are 0 not NULL.
WITH legs AS (
    SELECT
        date_trunc('hour', evt_block_time) AS time_bucket,
        collateralAsset AS asset, 'collateral' AS role,
        liquidatedCollateralAmount AS collateral_seized,
        CAST(NULL AS uint256) AS debt_covered,
        receiveAToken, "user", liquidator, evt_block_number
    FROM aave_v3_ethereum.pool_evt_liquidationcall
    WHERE evt_block_date >= DATE '{{start_date}}' AND evt_block_date < DATE '{{end_date}}'
    UNION ALL
    SELECT
        date_trunc('hour', evt_block_time) AS time_bucket,
        debtAsset AS asset, 'debt' AS role,
        CAST(NULL AS uint256) AS collateral_seized,
        debtToCover AS debt_covered,
        receiveAToken, "user", liquidator, evt_block_number
    FROM aave_v3_ethereum.pool_evt_liquidationcall
    WHERE evt_block_date >= DATE '{{start_date}}' AND evt_block_date < DATE '{{end_date}}'
),
buckets AS (
    SELECT
        date_add('hour', {{bucket_hours}} * CAST(floor(EXTRACT(hour FROM time_bucket) / {{bucket_hours}}) AS bigint), date_trunc('day', time_bucket)) AS time_bucket_2h,
        asset, role, collateral_seized, debt_covered, receiveAToken, "user", liquidator, evt_block_number
    FROM legs
),
agg AS (
    SELECT
        time_bucket_2h AS time_bucket, asset,
        COALESCE(SUM(collateral_seized), uint256 '0')                              AS liquidated_collateral_raw,
        COALESCE(SUM(debt_covered),      uint256 '0')                              AS liquidation_debt_covered_raw,
        COUNT(CASE WHEN role = 'collateral' THEN 1 END)                            AS as_collateral_tx_count,
        COUNT(CASE WHEN role = 'debt'       THEN 1 END)                            AS as_debt_tx_count,
        COUNT(*)                                                                   AS liquidation_tx_count,
        COUNT(CASE WHEN role = 'collateral' AND receiveAToken THEN 1 END)          AS receive_atoken_count,
        approx_distinct("user")                                                    AS unique_liquidated_users,
        approx_distinct(liquidator)                                                AS unique_liquidators,
        MAX(evt_block_number)                                                      AS latest_liquidation_block
    FROM buckets
    GROUP BY 1, 2
)
SELECT
    agg.time_bucket, agg.asset, tok.symbol AS asset_symbol,
    agg.liquidated_collateral_raw, agg.liquidation_debt_covered_raw,
    agg.as_collateral_tx_count, agg.as_debt_tx_count, agg.liquidation_tx_count,
    agg.receive_atoken_count, agg.unique_liquidated_users, agg.unique_liquidators,
    agg.latest_liquidation_block
FROM agg
LEFT JOIN tokens.erc20 tok
       ON tok.blockchain = 'ethereum' AND tok.contract_address = agg.asset
ORDER BY 1, 2