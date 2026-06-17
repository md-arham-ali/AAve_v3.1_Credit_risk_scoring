-- =====================================================================
-- Decimal reference — Part 5 (Flashloan + Oracle)
-- Schema: (asset, asset_symbol, metric, decimals, unit)
--   * flashloan_amount_raw / flashloan_premium_raw -> flashloan asset token decimals.
--   * oracle_asset_price -> 8 decimals (Aave oracle base-currency unit = 1e8),
--     asset-independent.
--   * asset_symbol -> on-chain token symbol from tokens.erc20 (NULL on fixed rows and
--     for assets absent from tokens.erc20).
-- =====================================================================
WITH fl_assets AS (
    SELECT DISTINCT asset
    FROM aave_v3_ethereum.pool_evt_flashloan
    WHERE evt_block_date >= DATE '2025-11-01' AND evt_block_date < DATE '2026-02-01'
),
tok AS (
    SELECT contract_address AS asset, decimals, symbol
    FROM tokens.erc20
    WHERE blockchain = 'ethereum'
)
-- flashloan raw amounts: per-asset token decimals
SELECT a.asset, t.symbol AS asset_symbol, m.metric, t.decimals, 'raw_token_amount' AS unit
FROM fl_assets a
LEFT JOIN tok t ON t.asset = a.asset
CROSS JOIN (VALUES
    ('flashloan_amount_raw'),
    ('flashloan_premium_raw')
) AS m(metric)

UNION ALL
-- fixed-decimal flashloan + oracle metrics
SELECT CAST(NULL AS varbinary) AS asset, CAST(NULL AS varchar) AS asset_symbol, f.metric, f.decimals, f.unit
FROM (VALUES
    ('flashloan_tx_count',               0, 'count'),
    ('unique_flashloan_initiators',      0, 'count'),
    ('no_open_debt_flashloan_tx_count',  0, 'count'),
    ('stable_flashloan_tx_count',        0, 'count'),
    ('variable_flashloan_tx_count',      0, 'count'),
    ('latest_flashloan_block',           0, 'block_number'),
    ('oracle_asset_price',               8, 'usd_price_8dp'),
    ('latest_oracle_block',              0, 'block_number'),
    ('oracle_call_count',                0, 'count')
) AS f(metric, decimals, unit)
ORDER BY 3, 1
