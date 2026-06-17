-- =====================================================================
-- Query 4 / Part 4A — Reserve configuration metrics (config STATE panel)
-- Grain / composite key : (time_bucket, asset)
-- Sources               : aave_v3_ethereum.poolconfigurator_evt_supplycapchanged
--                         aave_v3_ethereum.poolconfigurator_evt_borrowcapchanged
--                         aave_v3_ethereum.poolconfigurator_evt_debtceilingchanged
--                         aave_v3_ethereum.poolconfigurator_evt_reservefactorchanged
--                         aave_v3_ethereum.poolconfigurator_evt_collateralconfigurationchanged
-- Window                : 2025-11-01 00:00 (incl) .. 2026-02-01 00:00 (excl), UTC
-- Bucket                : fixed 6-hour slots (00 / 06 / 12 / 18 UTC)
--
-- WHY THIS IS A STATE PANEL (was a sparse change-log)
--   The poolconfigurator_evt_* tables only fire when governance CHANGES a parameter.
--   Emitting one row per change (the old design) produced: (a) only ~50 rows in the
--   window, (b) 76–90% NULLs — each change row populated only the one parameter that
--   changed, and (c) no baseline for assets unchanged in-window. That is a delta log,
--   not the config STATE a credit-risk model needs.
--
--   This query reconstructs the PREVAILING configuration of every reserve on the 6h
--   grid. Config is STATE (a step function that persists until changed), so the value
--   at bucket B is the last event value at/before B — forward-filled PER PARAMETER
--   from the FULL event history (no lower date bound), then clipped to the window.
--   Result: a dense (time_bucket, asset) panel with every parameter carried forward.
--   NOTE: this intentionally overrides the repo's earlier "config is sparse, do not
--   fill" rule, because forward-filling STATE is the real on-chain value, not a fill.
--
-- Decimal note (see decimal_reference_part4.sql for the authoritative mapping):
--   * supply_cap / borrow_cap -> WHOLE TOKENS (decimals 0), NOT raw token amount.
--   * debt_ceiling            -> 2 decimals (Aave V3 DEBT_CEILING_DECIMALS = 2).
--   * reserve_factor / liquidation_threshold / liquidation_bonus / ltv -> bps (1e4).
--
-- Encoding / NULL handling
--   * Each parameter is read as its NEW value; the last event of its own type within
--     a 6h bucket wins (max_by on (block, log index)); LEAD builds the validity
--     interval [valid_from, valid_to) that the value covers on the grid.
--   * collateralconfigurationchanged carries THREE params (ltv, liquidation_threshold,
--     liquidation_bonus) — split into three rows of the normalized stream.
--   * Unset parameters COALESCE to 0 — the on-chain default. 0 is meaningful: an
--     uncapped reserve has cap 0; a non-isolation reserve has debt_ceiling 0; a
--     non-collateral reserve has ltv / threshold / bonus 0. So the panel is NULL-free.
--   * config_event_count = # of config-change events for that asset IN that bucket
--     (0 when the row is a carried-forward state with no change). A useful "did config
--     change here?" signal; the rest of the row is the end-of-bucket state regardless.
--   * old_* columns are intentionally dropped — a state panel needs the current value,
--     and the previous value is just the prior bucket's row.
--   * A row appears for an asset from its FIRST-EVER config event onward (assets are
--     configured at listing, almost always before the window, so coverage is full).
-- =====================================================================
WITH ev AS (                                            -- normalized NEW-value stream, FULL history
    SELECT asset, 'supply_cap' AS param, CAST(newSupplyCap AS uint256) AS value,
           evt_block_time, evt_block_number, evt_index
    FROM aave_v3_ethereum.poolconfigurator_evt_supplycapchanged
    WHERE evt_block_date < DATE '2026-02-01'

    UNION ALL
    SELECT asset, 'borrow_cap', CAST(newBorrowCap AS uint256),
           evt_block_time, evt_block_number, evt_index
    FROM aave_v3_ethereum.poolconfigurator_evt_borrowcapchanged
    WHERE evt_block_date < DATE '2026-02-01'

    UNION ALL
    SELECT asset, 'debt_ceiling', CAST(newDebtCeiling AS uint256),
           evt_block_time, evt_block_number, evt_index
    FROM aave_v3_ethereum.poolconfigurator_evt_debtceilingchanged
    WHERE evt_block_date < DATE '2026-02-01'

    UNION ALL
    SELECT asset, 'reserve_factor', CAST(newReserveFactor AS uint256),
           evt_block_time, evt_block_number, evt_index
    FROM aave_v3_ethereum.poolconfigurator_evt_reservefactorchanged
    WHERE evt_block_date < DATE '2026-02-01'

    UNION ALL
    SELECT asset, 'liquidation_threshold', CAST(liquidationThreshold AS uint256),
           evt_block_time, evt_block_number, evt_index
    FROM aave_v3_ethereum.poolconfigurator_evt_collateralconfigurationchanged
    WHERE evt_block_date < DATE '2026-02-01'

    UNION ALL
    SELECT asset, 'liquidation_bonus', CAST(liquidationBonus AS uint256),
           evt_block_time, evt_block_number, evt_index
    FROM aave_v3_ethereum.poolconfigurator_evt_collateralconfigurationchanged
    WHERE evt_block_date < DATE '2026-02-01'

    UNION ALL
    SELECT asset, 'ltv', CAST(ltv AS uint256),
           evt_block_time, evt_block_number, evt_index
    FROM aave_v3_ethereum.poolconfigurator_evt_collateralconfigurationchanged
    WHERE evt_block_date < DATE '2026-02-01'
),
ev_b AS (                                               -- last value per (asset, param, event-bucket)
    SELECT
        asset, param,
        date_add('hour',
                 6 * CAST(floor(hour(evt_block_time) / 6) AS bigint),
                 date_trunc('day', evt_block_time))                                AS evt_bucket,
        max_by(value, ROW(evt_block_number, evt_index))                            AS value
    FROM ev
    GROUP BY 1, 2, 3
),
iv AS (                                                 -- step-function validity interval per value
    SELECT
        asset, param, value,
        evt_bucket                                                                 AS valid_from,
        LEAD(evt_bucket) OVER (PARTITION BY asset, param ORDER BY evt_bucket)       AS valid_to
    FROM ev_b
),
grid AS (                                               -- the window's fixed 6h bucket grid
    SELECT g AS time_bucket
    FROM UNNEST(sequence(TIMESTAMP '2025-11-01 00:00:00',
                         TIMESTAMP '2026-01-31 18:00:00',
                         INTERVAL '6' HOUR)) AS t(g)
),
state_long AS (                                         -- prevailing value of each param on the grid
    SELECT g.time_bucket, iv.asset, iv.param, iv.value
    FROM iv
    JOIN grid g
      ON g.time_bucket >= iv.valid_from
     AND (iv.valid_to IS NULL OR g.time_bucket < iv.valid_to)
),
state_wide AS (                                         -- pivot to one row per (time_bucket, asset)
    SELECT
        time_bucket,
        asset,
        COALESCE(max(CASE WHEN param = 'supply_cap'            THEN value END), uint256 '0') AS supply_cap,
        COALESCE(max(CASE WHEN param = 'borrow_cap'            THEN value END), uint256 '0') AS borrow_cap,
        COALESCE(max(CASE WHEN param = 'debt_ceiling'          THEN value END), uint256 '0') AS debt_ceiling,
        COALESCE(max(CASE WHEN param = 'reserve_factor'        THEN value END), uint256 '0') AS reserve_factor,
        COALESCE(max(CASE WHEN param = 'liquidation_threshold' THEN value END), uint256 '0') AS liquidation_threshold,
        COALESCE(max(CASE WHEN param = 'liquidation_bonus'     THEN value END), uint256 '0') AS liquidation_bonus,
        COALESCE(max(CASE WHEN param = 'ltv'                   THEN value END), uint256 '0') AS ltv
    FROM state_long
    GROUP BY 1, 2
),
chg AS (                                                -- in-window config-change count per bucket/asset
    SELECT
        date_add('hour',
                 6 * CAST(floor(hour(evt_block_time) / 6) AS bigint),
                 date_trunc('day', evt_block_time))                                AS time_bucket,
        asset,
        COUNT(*)                                                                   AS config_event_count
    FROM ev
    WHERE evt_block_time >= TIMESTAMP '2025-11-01 00:00:00'
      AND evt_block_time <  TIMESTAMP '2026-02-01 00:00:00'
    GROUP BY 1, 2
)
SELECT
    s.time_bucket,
    s.asset,
    tok.symbol                                                                     AS asset_symbol,
    s.supply_cap,
    s.borrow_cap,
    s.debt_ceiling,
    s.reserve_factor,
    s.liquidation_threshold,
    s.liquidation_bonus,
    s.ltv,
    COALESCE(c.config_event_count, 0)                                              AS config_event_count
FROM state_wide s
LEFT JOIN chg c
       ON c.asset = s.asset
      AND c.time_bucket = s.time_bucket
LEFT JOIN tokens.erc20 tok
       ON tok.blockchain = 'ethereum'
      AND tok.contract_address = s.asset
ORDER BY 1, 2
