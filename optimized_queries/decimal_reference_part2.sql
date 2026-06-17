-- =====================================================================
-- Decimal reference — Part 2 (Borrow + Repay)
-- Schema: (asset, asset_symbol, metric, decimals, unit)
--   * asset / asset_symbol NOT NULL -> per-token rows (decimals + on-chain symbol
--     from tokens.erc20); NULL on fixed metrics and assets absent from tokens.erc20.
-- =====================================================================
WITH part_assets AS (
    SELECT reserve AS asset
    FROM aave_v3_ethereum.pool_evt_borrow
    WHERE evt_block_date >= DATE '2025-11-01' AND evt_block_date < DATE '2026-02-01'
    UNION
    SELECT reserve AS asset
    FROM aave_v3_ethereum.pool_evt_repay
    WHERE evt_block_date >= DATE '2025-11-01' AND evt_block_date < DATE '2026-02-01'
),
tok AS (
    SELECT contract_address AS asset, decimals, symbol
    FROM tokens.erc20
    WHERE blockchain = 'ethereum'
)
-- raw-amount metrics: per-asset token decimals
SELECT a.asset, t.symbol AS asset_symbol, m.metric, t.decimals, 'raw_token_amount' AS unit
FROM part_assets a
LEFT JOIN tok t ON t.asset = a.asset
CROSS JOIN (VALUES
    ('borrow_amount_raw'),
    ('repay_amount_raw'),
    ('net_debt_flow_raw')
) AS m(metric)

UNION ALL
-- fixed-decimal metrics
SELECT CAST(NULL AS varbinary) AS asset, CAST(NULL AS varchar) AS asset_symbol, f.metric, f.decimals, f.unit
FROM (VALUES
    ('borrow_tx_count',          0,  'count'),
    ('repay_tx_count',           0,  'count'),
    ('stable_borrow_tx_count',   0,  'count'),
    ('variable_borrow_tx_count', 0,  'count'),
    ('unique_borrowers',         0,  'count'),
    ('unique_repayers',          0,  'count'),
    ('last_borrow_rate',         27, 'ray'),
    ('latest_borrow_block',      0,  'block_number'),
    ('latest_repay_block',       0,  'block_number')
) AS f(metric, decimals, unit)
ORDER BY 3, 1
