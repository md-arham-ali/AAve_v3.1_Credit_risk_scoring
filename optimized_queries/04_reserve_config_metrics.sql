-- Query 4 / Part 4A — Reserve configuration STATE panel — grain (time_bucket, asset)
-- Forward-fills each config parameter (last value at/before each 2h bucket) from the
-- FULL poolconfigurator_evt_* history, then clips to the window. Dense panel; unset
-- params COALESCE to 0 (on-chain default). old_* dropped (state, not deltas).
WITH ev AS (
    SELECT asset, 'supply_cap' AS param, CAST(newSupplyCap AS uint256) AS value,
           evt_block_time, evt_block_number, evt_index
    FROM aave_v3_ethereum.poolconfigurator_evt_supplycapchanged
    WHERE evt_block_date < DATE '2026-04-01'
    UNION ALL SELECT asset, 'borrow_cap', CAST(newBorrowCap AS uint256), evt_block_time, evt_block_number, evt_index
    FROM aave_v3_ethereum.poolconfigurator_evt_borrowcapchanged WHERE evt_block_date < DATE '2026-04-01'
    UNION ALL SELECT asset, 'debt_ceiling', CAST(newDebtCeiling AS uint256), evt_block_time, evt_block_number, evt_index
    FROM aave_v3_ethereum.poolconfigurator_evt_debtceilingchanged WHERE evt_block_date < DATE '2026-04-01'
    UNION ALL SELECT asset, 'reserve_factor', CAST(newReserveFactor AS uint256), evt_block_time, evt_block_number, evt_index
    FROM aave_v3_ethereum.poolconfigurator_evt_reservefactorchanged WHERE evt_block_date < DATE '2026-04-01'
    UNION ALL SELECT asset, 'liquidation_threshold', CAST(liquidationThreshold AS uint256), evt_block_time, evt_block_number, evt_index
    FROM aave_v3_ethereum.poolconfigurator_evt_collateralconfigurationchanged WHERE evt_block_date < DATE '2026-04-01'
    UNION ALL SELECT asset, 'liquidation_bonus', CAST(liquidationBonus AS uint256), evt_block_time, evt_block_number, evt_index
    FROM aave_v3_ethereum.poolconfigurator_evt_collateralconfigurationchanged WHERE evt_block_date < DATE '2026-04-01'
    UNION ALL SELECT asset, 'ltv', CAST(ltv AS uint256), evt_block_time, evt_block_number, evt_index
    FROM aave_v3_ethereum.poolconfigurator_evt_collateralconfigurationchanged WHERE evt_block_date < DATE '2026-04-01'
),
ev_b AS (
    SELECT asset, param,
        date_add('hour', 2 * CAST(floor(hour(evt_block_time) / 2) AS bigint),
                 date_trunc('day', evt_block_time)) AS evt_bucket,
        max_by(value, ROW(evt_block_number, evt_index)) AS value
    FROM ev GROUP BY 1, 2, 3
),
iv AS (
    SELECT asset, param, value, evt_bucket AS valid_from,
        LEAD(evt_bucket) OVER (PARTITION BY asset, param ORDER BY evt_bucket) AS valid_to
    FROM ev_b
),
grid AS (
    SELECT g AS time_bucket
    FROM UNNEST(sequence(TIMESTAMP '2025-04-01 00:00:00', TIMESTAMP '2026-03-31 22:00:00', INTERVAL '2' HOUR)) AS t(g)
),
state_long AS (
    SELECT g.time_bucket, iv.asset, iv.param, iv.value
    FROM iv JOIN grid g
      ON g.time_bucket >= iv.valid_from AND (iv.valid_to IS NULL OR g.time_bucket < iv.valid_to)
),
state_wide AS (
    SELECT time_bucket, asset,
        COALESCE(max(CASE WHEN param = 'supply_cap'            THEN value END), uint256 '0') AS supply_cap,
        COALESCE(max(CASE WHEN param = 'borrow_cap'            THEN value END), uint256 '0') AS borrow_cap,
        COALESCE(max(CASE WHEN param = 'debt_ceiling'          THEN value END), uint256 '0') AS debt_ceiling,
        COALESCE(max(CASE WHEN param = 'reserve_factor'        THEN value END), uint256 '0') AS reserve_factor,
        COALESCE(max(CASE WHEN param = 'liquidation_threshold' THEN value END), uint256 '0') AS liquidation_threshold,
        COALESCE(max(CASE WHEN param = 'liquidation_bonus'     THEN value END), uint256 '0') AS liquidation_bonus,
        COALESCE(max(CASE WHEN param = 'ltv'                   THEN value END), uint256 '0') AS ltv
    FROM state_long GROUP BY 1, 2
),
chg AS (
    SELECT
        date_add('hour', 2 * CAST(floor(hour(evt_block_time) / 2) AS bigint),
                 date_trunc('day', evt_block_time)) AS time_bucket,
        asset, COUNT(*) AS config_event_count
    FROM ev
    WHERE evt_block_time >= TIMESTAMP '2025-04-01 00:00:00' AND evt_block_time < TIMESTAMP '2026-04-01 00:00:00'
    GROUP BY 1, 2
)
SELECT
    s.time_bucket, s.asset, tok.symbol AS asset_symbol,
    CAST(s.supply_cap AS double) AS supply_cap,
    CAST(s.borrow_cap AS double) AS borrow_cap,
    CAST(s.debt_ceiling AS double) AS debt_ceiling,
    CAST(s.reserve_factor AS double) AS reserve_factor,
    CAST(s.liquidation_threshold AS double) AS liquidation_threshold,
    CAST(s.liquidation_bonus AS double) AS liquidation_bonus,
    CAST(s.ltv AS double) AS ltv,
    COALESCE(c.config_event_count, 0) AS config_event_count
FROM state_wide s
LEFT JOIN chg c ON c.asset = s.asset AND c.time_bucket = s.time_bucket
LEFT JOIN tokens.erc20 tok ON tok.blockchain = 'ethereum' AND tok.contract_address = s.asset
ORDER BY 1, 2