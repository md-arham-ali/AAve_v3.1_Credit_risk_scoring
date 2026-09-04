WITH events AS (
    SELECT
        reserve                       AS asset,
        evt_block_time,
        evt_block_number,
        evt_index,
        amount,
        "user"                        AS actor,
        interestRateMode              AS irm,
        borrowRate                    AS borrow_rate,
        'borrow'                      AS kind
    FROM aave_v3_ethereum.pool_evt_borrow
    WHERE evt_block_date >= DATE '{{start_date}}'
      AND evt_block_date <  DATE '{{end_date}}'
    UNION ALL
    SELECT
        reserve                       AS asset,
        evt_block_time,
        evt_block_number,
        evt_index,
        amount,
        repayer                       AS actor,
        CAST(NULL AS integer)         AS irm,
        CAST(NULL AS uint256)         AS borrow_rate,
        'repay'                       AS kind
    FROM aave_v3_ethereum.pool_evt_repay
    WHERE evt_block_date >= DATE '{{start_date}}'
      AND evt_block_date <  DATE '{{end_date}}'
),
agg AS (
    SELECT
        date_add('hour',
                 {{bucket_hours}} * CAST(floor(hour(evt_block_time) / {{bucket_hours}}) AS bigint),
                 date_trunc('day', evt_block_time))                            AS time_bucket,
        asset,
        SUM(CASE WHEN kind = 'borrow' THEN amount END)                         AS borrow_amount_raw,
        SUM(CASE WHEN kind = 'repay'  THEN amount END)                         AS repay_amount_raw,
        SUM(CASE WHEN kind = 'borrow' THEN  CAST(amount AS int256)
                 WHEN kind = 'repay'  THEN -CAST(amount AS int256) END)        AS net_debt_flow_raw,
        COUNT(CASE WHEN kind = 'borrow' THEN 1 END)                            AS borrow_tx_count,
        COUNT(CASE WHEN kind = 'repay'  THEN 1 END)                            AS repay_tx_count,
        COUNT(CASE WHEN kind = 'borrow' AND irm = 1 THEN 1 END)                AS stable_borrow_tx_count,
        COUNT(CASE WHEN kind = 'borrow' AND irm = 2 THEN 1 END)                AS variable_borrow_tx_count,
        approx_distinct(CASE WHEN kind = 'borrow' THEN actor END)              AS unique_borrowers,
        approx_distinct(CASE WHEN kind = 'repay'  THEN actor END)              AS unique_repayers,
        max_by(borrow_rate,
               CASE WHEN kind = 'borrow' THEN ROW(evt_block_number, evt_index) END)
                                                                               AS last_borrow_rate,
        MAX(CASE WHEN kind = 'borrow' THEN evt_block_number END)               AS latest_borrow_block,
        MAX(CASE WHEN kind = 'repay'  THEN evt_block_number END)               AS latest_repay_block
    FROM events
    GROUP BY 1, 2
)
SELECT
    agg.time_bucket,
    agg.asset,
    tok.symbol                                                                 AS asset_symbol,
    agg.borrow_amount_raw                                                      AS borrow_amount,
    agg.repay_amount_raw                                                       AS repay_amount,
    agg.net_debt_flow_raw                                                      AS net_debt_flow,
    agg.borrow_tx_count,
    agg.repay_tx_count,
    agg.stable_borrow_tx_count,
    agg.variable_borrow_tx_count,
    agg.unique_borrowers,
    agg.unique_repayers,
    agg.last_borrow_rate,
    agg.latest_borrow_block,
    agg.latest_repay_block
FROM agg
LEFT JOIN tokens.erc20 tok
       ON tok.blockchain = 'ethereum'
      AND tok.contract_address = agg.asset
ORDER BY 1, 2