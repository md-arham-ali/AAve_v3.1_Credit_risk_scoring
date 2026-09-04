WITH events AS (
    SELECT reserve AS asset, evt_block_time, evt_block_number, "user" AS actor, 'enabled' AS kind
    FROM aave_v3_ethereum.pool_evt_reserveusedascollateralenabled
    WHERE evt_block_date >= DATE '{{start_date}}' AND evt_block_date < DATE '{{end_date}}'
    UNION ALL
    SELECT reserve AS asset, evt_block_time, evt_block_number, "user" AS actor, 'disabled' AS kind
    FROM aave_v3_ethereum.pool_evt_reserveusedascollateraldisabled
    WHERE evt_block_date >= DATE '{{start_date}}' AND evt_block_date < DATE '{{end_date}}'
),
agg AS (
    SELECT
        date_add('hour', {{bucket_hours}} * CAST(floor(hour(evt_block_time) / {{bucket_hours}}) AS bigint), date_trunc('day', evt_block_time)) AS time_bucket,
        asset,
        COUNT(CASE WHEN kind = 'enabled' THEN 1 END) AS collateral_enabled_count,
        COUNT(CASE WHEN kind = 'disabled' THEN 1 END) AS collateral_disabled_count,
        approx_distinct(CASE WHEN kind = 'enabled' THEN actor END) AS unique_collateral_enable_users,
        approx_distinct(CASE WHEN kind = 'disabled' THEN actor END) AS unique_collateral_disable_users,
        MAX(evt_block_number) AS latest_collateral_toggle_block
    FROM events
    GROUP BY 1, 2
)
SELECT
    agg.time_bucket,
    agg.asset,
    tok.symbol AS asset_symbol,
    agg.collateral_enabled_count,
    agg.collateral_disabled_count,
    agg.unique_collateral_enable_users,
    agg.unique_collateral_disable_users,
    agg.latest_collateral_toggle_block
FROM agg
LEFT JOIN tokens.erc20 tok
       ON tok.blockchain = '`ethereum'
      AND tok.contract_address = agg.asset
ORDER BY 1, 2