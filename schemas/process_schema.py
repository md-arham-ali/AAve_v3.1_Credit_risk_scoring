"""Reference column structure for the tables `process.ipynb` extracts from Dune.

Hand-authored on purpose, and hand-authored HERE rather than in `src/`: this is data with
commentary attached, the same category as `domains/credit_risk/model_config.py` and
deliberately not the same as `column_priority.json` (a generated lookup with nothing to say).

Captured from the real files of BOTH windows on disk (2024-04→2025-03 and 2025-04→2026-03),
full reads — a partial read infers different dtypes on a column that only goes null later on.
Checked by `src/schema_check.py` at the moment each table is produced or loaded.

WHY SOME DTYPES ARE TUPLES
    `borrow_repay`, `reserve_config` and `reserve_state_rates` are the three tables
    `normalize.ipynb` rescales IN PLACE, so the same file has two legitimate structural
    states: raw as extracted (huge integers -> `object`, bps -> `int64`) and rescaled
    (`float64`). Both are accepted here because both are real. That is not a hole in the
    gate: whether a file SHOULD still be raw is a question about values, and values are what
    `normalize.ipynb`'s `looks_raw_*` guards and `adv_validation`'s Tier-1 un-scaled-wei
    check are for. This file only asserts structure.

FINDING RECORDED WHILE CAPTURING THIS
    `normalize.ipynb`'s TABLES entry for `flashloan` names `flashloan_amount` /
    `flashloan_premium`, but the columns on disk are `flashloan_amount_raw` /
    `flashloan_premium_raw` — in BOTH windows. `looks_raw_token_amounts()` ignores missing
    columns and returns False, so that table is reported "already normalized" and silently
    skipped on every run. Nothing is broken today because `transform.ipynb` scales flashloan
    itself (`scale_by_decimals(df_0_flashloan, ...)`), which is why this went unnoticed —
    but the normalize entry is dead configuration. Left as-is, deliberately, so this pass
    stays a schema pass; recorded in `context/bugs.md`.

The decimal-reference tables are here too. `process.ipynb` does not re-fetch them (they are
in DISABLED_QUERY_IDS), so its own check never sees them — but `normalize.ipynb` loads
`decimal_reference_part4` and checks it against this same file, and a schema with no frame is
allowed by `check_frames()`.
"""

import schema_check as sc

SCHEMAS = {
    "borrow_repay": {"columns": {
        "time_bucket": "object",
        "asset": "object",
        "asset_symbol": "object",
        "borrow_amount": ("float64", "object"),
        "repay_amount": ("float64", "object"),
        "net_debt_flow": ("float64", "object"),
        "borrow_tx_count": "int64",
        "repay_tx_count": "int64",
        "stable_borrow_tx_count": "int64",
        "variable_borrow_tx_count": "int64",
        "unique_borrowers": "int64",
        "unique_repayers": "int64",
        "last_borrow_rate": "object",
        "latest_borrow_block": "float64",
        "latest_repay_block": "float64",
    }},
    "collateral_toggle": {"columns": {
        "time_bucket": "object",
        "asset": "object",
        "asset_symbol": "float64",  # fully null in every window on disk — float64 IS the nullability signal
        "collateral_enabled_count": "int64",
        "collateral_disabled_count": "int64",
        "unique_collateral_enable_users": "int64",
        "unique_collateral_disable_users": "int64",
        "latest_collateral_toggle_block": "int64",
    }},
    "decimal_reference_part1": {"columns": {
        "asset": "object",
        "asset_symbol": "object",
        "metric": "object",
        "decimals": "int64",
        "unit": "object",
    }},
    "decimal_reference_part2": {"columns": {
        "asset": "object",
        "asset_symbol": "object",
        "metric": "object",
        "decimals": "int64",
        "unit": "object",
    }},
    "decimal_reference_part3": {"columns": {
        "asset": "float64",  # these parts carry no per-asset rows, so both columns are all-null
        "asset_symbol": "float64",
        "metric": "object",
        "decimals": "int64",
        "unit": "object",
    }},
    "decimal_reference_part4": {"columns": {
        "asset": "object",
        "asset_symbol": "object",
        "metric": "object",
        "decimals": "int64",
        "unit": "object",
    }},
    "decimal_reference_part5": {"columns": {
        "asset": "object",
        "asset_symbol": "object",
        "metric": "object",
        "decimals": "int64",
        "unit": "object",
    }},
    "decimal_reference_part6": {"columns": {
        "asset": "float64",  # ditto
        "asset_symbol": "float64",
        "metric": "object",
        "decimals": "int64",
        "unit": "object",
    }},
    "decimal_reference_part9_collateral_toggle": {"columns": {
        "asset": "float64",  # ditto
        "asset_symbol": "float64",
        "metric": "object",
        "decimals": "int64",
        "unit": "object",
    }},
    "flashloan": {"columns": {
        "time_bucket": "object",
        "asset": "object",
        "asset_symbol": "object",
        "flashloan_amount_raw": "object",  # still _raw on disk — see the note above about normalize.ipynb
        "flashloan_premium_raw": "object",
        "flashloan_tx_count": "int64",
        "unique_flashloan_initiators": "int64",
        "no_open_debt_flashloan_tx_count": "int64",
        "stable_flashloan_tx_count": "int64",
        "variable_flashloan_tx_count": "int64",
        "latest_flashloan_block": "int64",
    }},
    "liquidation": {"columns": {
        "time_bucket": "object",
        "asset": "object",
        "asset_symbol": "object",
        "liquidated_collateral_raw": "object",
        "liquidation_debt_covered_raw": "object",
        "as_collateral_tx_count": "int64",
        "as_debt_tx_count": "int64",
        "liquidation_tx_count": "int64",
        "receive_atoken_count": "int64",
        "unique_liquidated_users": "int64",
        "unique_liquidators": "int64",
        "latest_liquidation_block": "int64",
    }},
    "oracle_price_usd_eth_weth_6h": {"columns": {
        "time_bucket": "object",
        "asset": "object",
        "symbol": "object",
        "decimals": "int64",
        "avg_price_usd": "float64",
        "avg_price_eth": "float64",
        "price_points": "int64",
    }},
    "reserve_config": {"columns": {
        "time_bucket": "object",
        "asset": "object",
        "asset_symbol": "object",
        "supply_cap": "int64",
        "borrow_cap": "int64",
        "debt_ceiling": ("float64", "int64"),
        "reserve_factor": ("float64", "int64"),
        "liquidation_threshold": ("float64", "int64"),
        "liquidation_bonus": ("float64", "int64"),
        "ltv": ("float64", "int64"),
        "config_event_count": "int64",
    }},
    "reserve_state_rates": {"columns": {
        "time_bucket": "object",
        "asset": "object",
        "liquidity_rate": ("float64", "object"),
        "variable_borrow_rate": ("float64", "object"),
        "stable_borrow_rate": ("float64", "object"),
        "liquidity_index": ("float64", "object"),
        "variable_borrow_index": ("float64", "object"),
        "update_count": "int64",
    }},
    "supply_withdraw": {"columns": {
        "time_bucket": "object",
        "asset": "object",
        "asset_symbol": "object",
        "supply_amount_raw": "object",
        "withdrawal_amount_raw": "object",
        "net_supply_flow_raw": "object",
        "supply_tx_count": "int64",
        "withdrawal_tx_count": "int64",
        "unique_suppliers": "int64",
        "unique_withdraw_users": "int64",
        "latest_supply_block": "float64",  # float64, not int64: null in buckets with no supply event
        "latest_withdraw_block": "float64",
    }},
    "user_account": {"columns": {
        "time_bucket": "object",
        "avg_total_collateral_base": "float64",
        "avg_total_debt_base": "float64",
        "avg_available_borrows_base": "float64",
        "avg_current_liquidation_threshold": "float64",
        "avg_ltv": "float64",
        "min_health_factor": "float64",  # DOUBLE from Dune's engine, ~62 digits lost upstream (E45)
        "max_health_factor": "float64",
        "sampled_user_count": "int64",
        "account_data_call_count": "int64",
    }},
}

# Fails at IMPORT time on a typo'd entry, rather than at whichever gate first touches that
# table — same reasoning as data_validation.py's TABLE_LABELS uniqueness assert.
sc.assert_schema_file(SCHEMAS, name="process_schema")
