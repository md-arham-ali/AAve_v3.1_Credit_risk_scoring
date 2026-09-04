SELECT
    date_add('hour',
             {{bucket_hours}} * CAST(floor(hour(call_block_time) / {{bucket_hours}}) AS bigint),
             date_trunc('day', call_block_time))                               AS time_bucket,
    AVG(CAST(output_totalCollateralBase        AS double))                    AS avg_total_collateral_base,
    AVG(CAST(output_totalDebtBase              AS double))                    AS avg_total_debt_base,
    AVG(CAST(output_availableBorrowsBase       AS double))                    AS avg_available_borrows_base,
    AVG(CAST(output_currentLiquidationThreshold AS double))                   AS avg_current_liquidation_threshold,
    AVG(CAST(output_ltv                        AS double))                    AS avg_ltv,
    CAST(MIN(output_healthFactor) AS double)                                  AS min_health_factor,
    CAST(MAX(output_healthFactor) AS double)                                  AS max_health_factor,
    approx_distinct("user")                                                   AS sampled_user_count,
    COUNT(*)                                                                  AS account_data_call_count
FROM aave_v3_ethereum.pool_call_getuseraccountdata
WHERE call_block_date >= DATE '{{start_date}}'
  AND call_block_date <  DATE '{{end_date}}'
  AND call_success
GROUP BY 1
ORDER BY 1