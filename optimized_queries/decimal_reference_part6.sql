-- =====================================================================
-- Decimal reference — Part 6 (User account state)
-- Schema: (asset, asset_symbol, metric, decimals, unit)   [asset/asset_symbol always
--          NULL — no asset dim; columns kept for a stable cross-table schema]
--   *_base                 -> USD base currency, 8 decimals
--   *_liquidation_threshold / *_ltv -> basis points, 4 decimals (1e4)
--   health factor          -> WAD, 18 decimals (1e18 = HF 1.0)
-- =====================================================================
SELECT CAST(NULL AS varbinary) AS asset, CAST(NULL AS varchar) AS asset_symbol, f.metric, f.decimals, f.unit
FROM (VALUES
    ('avg_total_collateral_base',         8,  'usd_base_8dp'),
    ('avg_total_debt_base',               8,  'usd_base_8dp'),
    ('avg_available_borrows_base',        8,  'usd_base_8dp'),
    ('avg_current_liquidation_threshold', 4,  'basis_points'),
    ('avg_ltv',                           4,  'basis_points'),
    ('min_health_factor',                 18, 'wad'),
    ('max_health_factor',                 18, 'wad'),
    ('sampled_user_count',                0,  'count'),
    ('account_data_call_count',           0,  'count')
) AS f(metric, decimals, unit)
ORDER BY 3, 1
