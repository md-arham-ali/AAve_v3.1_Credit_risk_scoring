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
    WHERE evt_block_date >= DATE '{{start_date}}'
      AND evt_block_date <= DATE '{{end_date}}'

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
    WHERE evt_block_date >= DATE '{{start_date}}'
      AND evt_block_date <= DATE '{{end_date}}'
),
agg AS (
    SELECT
        date_add('hour',
                 {{bucket_hours}} * CAST(floor(hour(evt_block_time) / {{bucket_hours}}) AS bigint),
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