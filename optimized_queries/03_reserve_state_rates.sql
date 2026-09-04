-- =====================================================================
SELECT
    date_add('hour',
             {{bucket_hours}} * CAST(floor(hour(evt_block_time) / {{bucket_hours}}) AS bigint),
             date_trunc('day', evt_block_time))                                    AS time_bucket,
    reserve                                                                        AS asset,
    max_by(liquidityRate,       ROW(evt_block_number, evt_index))                 AS liquidity_rate,
    max_by(variableBorrowRate,  ROW(evt_block_number, evt_index))                 AS variable_borrow_rate,
    max_by(stableBorrowRate,    ROW(evt_block_number, evt_index))                 AS stable_borrow_rate,
    max_by(liquidityIndex,      ROW(evt_block_number, evt_index))                 AS liquidity_index,
    max_by(variableBorrowIndex, ROW(evt_block_number, evt_index))                 AS variable_borrow_index,
    COUNT(*)                                                                       AS update_count
FROM aave_v3_ethereum.pool_evt_reservedataupdated
WHERE evt_block_date >= DATE '{{start_date}}'
  AND evt_block_date <  DATE '{{end_date}}'
GROUP BY 1, 2
ORDER BY 1, 2