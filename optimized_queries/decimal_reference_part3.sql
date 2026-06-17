-- =====================================================================
-- Decimal reference — Part 3 (Reserve state + rates)
-- Schema: (asset, asset_symbol, metric, decimals, unit)
-- All Part 3 metrics have fixed (asset-independent) decimals:
--   rates  -> RAY  (1e27)
--   indexes-> RAY  (1e27)
-- asset / asset_symbol are always NULL here (decimals do not vary by token); the
-- columns exist only to keep one stable schema across all decimal-reference tables.
-- =====================================================================
SELECT CAST(NULL AS varbinary) AS asset, CAST(NULL AS varchar) AS asset_symbol, f.metric, f.decimals, f.unit
FROM (VALUES
    ('liquidity_rate',        27, 'ray'),
    ('variable_borrow_rate',  27, 'ray'),
    ('stable_borrow_rate',    27, 'ray'),
    ('liquidity_index',       27, 'ray'),
    ('variable_borrow_index', 27, 'ray'),
    ('update_count',          0,  'count')
) AS f(metric, decimals, unit)
ORDER BY 3, 1
