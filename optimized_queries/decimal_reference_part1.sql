-- =====================================================================
-- Decimal reference — Part 1 (Supply + Withdraw)
-- Schema: (asset, asset_symbol, metric, decimals, unit)
--   * asset IS NULL  -> the metric's decimals are asset-independent (fixed).
--   * asset NOT NULL -> per-token decimals for raw-amount metrics (from tokens.erc20).
--   * asset_symbol   -> on-chain token symbol from tokens.erc20 (NULL on fixed /
--                       asset-independent rows, and for assets absent from tokens.erc20).
-- Token-amount metrics use the underlying reserve token's decimals.
-- =====================================================================
WITH part_assets AS (
    SELECT reserve AS asset
    FROM aave_v3_ethereum.pool_evt_supply
    WHERE evt_block_date >= DATE '2025-11-01' AND evt_block_date < DATE '2026-02-01'
    UNION
    SELECT reserve AS asset
    FROM aave_v3_ethereum.pool_evt_withdraw
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
    ('supply_amount_raw'),
    ('withdrawal_amount_raw'),
    ('net_supply_flow_raw')
) AS m(metric)

UNION ALL
-- fixed-decimal metrics: asset-independent
SELECT CAST(NULL AS varbinary) AS asset, CAST(NULL AS varchar) AS asset_symbol, f.metric, f.decimals, f.unit
FROM (VALUES
    ('supply_tx_count',        0, 'count'),
    ('withdrawal_tx_count',    0, 'count'),
    ('unique_suppliers',       0, 'count'),
    ('unique_withdraw_users',  0, 'count'),
    ('latest_supply_block',    0, 'block_number'),
    ('latest_withdraw_block',  0, 'block_number')
) AS f(metric, decimals, unit)
ORDER BY 3, 1
