WITH agg AS (
    SELECT
        date_add('hour',
                 2 * CAST(floor(hour(evt_block_time) / 2) AS bigint),
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
    WHERE evt_block_date >= DATE '2025-04-01'
      AND evt_block_date < DATE '2026-04-01'
    GROUP BY 1, 2
)
SELECT
    agg.time_bucket,
    agg.asset,
    tok.symbol                                                                    AS asset_symbol,
    agg.flashloan_amount_raw                                                    AS flashloan_amount,
    agg.flashloan_premium_raw                                                   AS flashloan_premium,
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