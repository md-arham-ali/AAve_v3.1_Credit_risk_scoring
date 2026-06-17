-- =====================================================================
-- Decimal reference — Part 4 (Reserve config + Liquidations)
-- Schema: (asset, asset_symbol, metric, decimals, unit)
--   * asset / asset_symbol NOT NULL -> per-token liquidation-amount rows (decimals +
--     on-chain symbol from tokens.erc20); NULL on fixed config/liquidation metrics.
--
-- Aave V3 on-chain encoding (config metrics):
--   * supply_cap / borrow_cap            -> decimals 0  (stored in WHOLE TOKENS).
--   * debt_ceiling                       -> decimals 2  (Aave V3 DEBT_CEILING_DECIMALS).
--   * reserve_factor / liquidation_threshold / liquidation_bonus / ltv -> bps (1e4).
--   (old_* columns were dropped — the config table is now a STATE panel, current value
--    only; see 04_reserve_config_metrics.sql.)
--
-- Liquidation raw amounts are now PER-ASSET (query 05 re-keyed to (time_bucket, asset)):
--   * liquidated_collateral_raw     -> the row asset's token decimals (collateral seized).
--   * liquidation_debt_covered_raw  -> the row asset's token decimals (debt covered).
--   Both metrics share one asset universe = collateral assets UNION debt assets.
-- =====================================================================
WITH liq_assets AS (
    SELECT DISTINCT collateralAsset AS asset
    FROM aave_v3_ethereum.pool_evt_liquidationcall
    WHERE evt_block_date >= DATE '2025-11-01' AND evt_block_date < DATE '2026-02-01'
    UNION
    SELECT DISTINCT debtAsset AS asset
    FROM aave_v3_ethereum.pool_evt_liquidationcall
    WHERE evt_block_date >= DATE '2025-11-01' AND evt_block_date < DATE '2026-02-01'
),
tok AS (
    SELECT contract_address AS asset, decimals, symbol
    FROM tokens.erc20
    WHERE blockchain = 'ethereum'
)
-- per-asset liquidation amounts: both metrics use the row asset's token decimals
SELECT a.asset, t.symbol AS asset_symbol, m.metric, t.decimals, 'raw_token_amount' AS unit
FROM liq_assets a
LEFT JOIN tok t ON t.asset = a.asset
CROSS JOIN (VALUES
    ('liquidated_collateral_raw'),
    ('liquidation_debt_covered_raw')
) AS m(metric)

UNION ALL
-- fixed-decimal config metrics + liquidation/config count metrics (asset-independent)
SELECT CAST(NULL AS varbinary) AS asset, CAST(NULL AS varchar) AS asset_symbol, f.metric, f.decimals, f.unit
FROM (VALUES
    ('supply_cap',                0, 'whole_tokens'),
    ('borrow_cap',                0, 'whole_tokens'),
    ('debt_ceiling',              2, 'usd_2dp'),
    ('reserve_factor',            4, 'basis_points'),
    ('liquidation_threshold',     4, 'basis_points'),
    ('liquidation_bonus',         4, 'basis_points'),
    ('ltv',                       4, 'basis_points'),
    ('config_event_count',        0, 'count'),
    ('as_collateral_tx_count',    0, 'count'),
    ('as_debt_tx_count',          0, 'count'),
    ('liquidation_tx_count',      0, 'count'),
    ('receive_atoken_count',      0, 'count'),
    ('unique_liquidated_users',   0, 'count'),
    ('unique_liquidators',        0, 'count'),
    ('latest_liquidation_block',  0, 'block_number')
) AS f(metric, decimals, unit)
ORDER BY 3, 1
