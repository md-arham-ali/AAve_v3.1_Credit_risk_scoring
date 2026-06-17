-- =====================================================================
-- Decimal reference — Query 9 (Collateral usage toggle)
-- Schema: (asset, asset_symbol, metric, decimals, unit)   [asset/asset_symbol always
--          NULL — all metrics asset-independent; columns kept for a stable schema]
-- All metrics are counts / block numbers -> decimals 0.
-- =====================================================================
SELECT CAST(NULL AS varbinary) AS asset, CAST(NULL AS varchar) AS asset_symbol, f.metric, f.decimals, f.unit
FROM (VALUES
    ('collateral_enabled_count',          0, 'count'),
    ('collateral_disabled_count',         0, 'count'),
    ('unique_collateral_enable_users',    0, 'count'),
    ('unique_collateral_disable_users',   0, 'count'),
    ('latest_collateral_toggle_block',    0, 'block_number')
) AS f(metric, decimals, unit)
ORDER BY 3, 1
