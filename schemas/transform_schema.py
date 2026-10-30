"""Reference column structure for what `transform.ipynb` writes.

15 tables in one flat dict — the 8 that go to `transformed_data/batch_<span>/` and the 7 that
go to `processed_data/batch_<span>/transform/`. One dict because the two sets share no names
and both are produced by the same notebook in the same run; splitting them would only add a
second import to every call site.

Captured from the real 2025-04→2026-03 batch on disk, full reads.

THE ONE PLACE THIS CAN BITE, and what to do about it
    These are the dtypes pandas infers when READING the CSVs back. `transform.ipynb`'s output
    gate runs a step earlier, against the frames still in memory, and the two can in principle
    disagree (an int column written and re-read as float, say). They agree for every table
    here today. If a first run trips on one, the assertion names the column and both dtypes:
    widen that entry to a tuple, e.g. ("int64", "float64"), and note why. `feature_addition`'s
    input gate cannot hit this — it checks frames that came from pd.read_csv, which is exactly
    what was captured here.

`reserve_panel` is `DF_common_1` under a second name: `transform.ipynb` writes the same frame
to both folders (the processed_data copy is the per-stage intermediate). Two entries, one
structure — deliberate, because the two folders are read by different consumers.
"""

import schema_check as sc

SCHEMAS = {
    "DF_common_1": {"columns": {
        "time_bucket": "object",
        "asset": "object",
        "asset_symbol": "object",
        # object in memory, float64 on disk — and BOTH are this column, which is why the
        # tuple. transform.ipynb computes it as `.astype(float) / 10**27`; 10**27 exceeds
        # int64, so numpy cannot hold the divisor and falls back to element-wise OBJECT
        # division. The values are Python floats, bit-identical to `/ 1e27`, and the CSV
        # round-trip restores float64 — which is why this went unseen until a gate compared
        # the in-memory frame to a disk-captured reference. `/ 1e27` in cell 12 would make
        # both sides float64 and let this entry tighten back to a plain "float64";
        # deliberately not changed in a schema pass. See bugs.md 7b.7.
        "last_borrow_rate": ("float64", "object"),
        "liquidity_rate": "float64",
        "variable_borrow_rate": "float64",
        "liquidity_index": "float64",
        "variable_borrow_index": "float64",
        "update_count_liquidity": "int64",
    }},
    "DF_common_final_1": {"columns": {
        "time_bucket": "object",  # canonical '<ts>.fff UTC' string, not a datetime — see canonicalize_keys
        "supply_tx_count": "int64",
        "withdrawal_tx_count": "int64",
        "unique_suppliers": "int64",
        "unique_withdraw_users": "int64",
        "supply_amount_value_usd": "float64",
        "supply_amount_value_eth": "float64",
        "withdrawal_amount_value_usd": "float64",
        "withdrawal_amount_value_eth": "float64",
        "borrow_tx_count": "float64",
        "repay_tx_count": "float64",
        "variable_borrow_tx_count": "float64",
        "unique_borrowers": "float64",
        "unique_repayers": "float64",
        "borrow_amount_value_usd": "float64",
        "borrow_amount_value_eth": "float64",
        "repay_amount_value_usd": "float64",
        "repay_amount_value_eth": "float64",
        "collateral_enabled_count": "float64",
        "collateral_disabled_count": "float64",
        "unique_collateral_enable_users": "float64",
        "unique_collateral_disable_users": "float64",
        "flashloan_amount_value_usd": "float64",
        "flashloan_amount_value_eth": "float64",
        "flashloan_premium_value_usd": "float64",
        "flashloan_premium_value_eth": "float64",
        "flashloan_tx_count": "float64",
        "unique_flashloan_initiators": "float64",
        "no_open_debt_flashloan_tx_count": "float64",
        "variable_flashloan_tx_count": "float64",
    }},
    "borrow_repay": {"columns": {
        "time_bucket": "object",
        "borrow_tx_count": "int64",
        "repay_tx_count": "int64",
        "variable_borrow_tx_count": "int64",
        "unique_borrowers": "int64",
        "unique_repayers": "int64",
        "borrow_amount_value_usd": "float64",
        "borrow_amount_value_eth": "float64",
        "repay_amount_value_usd": "float64",
        "repay_amount_value_eth": "float64",
    }},
    "collateral_toggle": {"columns": {
        "time_bucket": "object",
        "collateral_enabled_count": "int64",
        "collateral_disabled_count": "int64",
        "unique_collateral_enable_users": "int64",
        "unique_collateral_disable_users": "int64",
    }},
    "df_liq_24h": {"columns": {
        "time_bucket": "object",
        "as_collateral_tx_count": "int64",
        "as_debt_tx_count": "int64",
        "liquidation_tx_count": "int64",
        "unique_liquidated_users": "int64",
        "unique_liquidators": "int64",
        "liquidated_collateral_value_usd": "float64",
        "liquidated_collateral_value_eth": "float64",
        "liquidation_debt_covered_value_usd": "float64",
        "liquidation_debt_covered_value_eth": "float64",
    }},
    "df_liq_2h": {"columns": {
        "time_bucket": "object",
        "as_collateral_tx_count": "int64",
        "as_debt_tx_count": "int64",
        "liquidation_tx_count": "int64",
        "unique_liquidated_users": "int64",
        "unique_liquidators": "int64",
        "liquidated_collateral_value_usd": "float64",
        "liquidated_collateral_value_eth": "float64",
        "liquidation_debt_covered_value_usd": "float64",
        "liquidation_debt_covered_value_eth": "float64",
    }},
    "df_liq_7d": {"columns": {
        "time_bucket": "object",
        "as_collateral_tx_count": "int64",
        "as_debt_tx_count": "int64",
        "liquidation_tx_count": "int64",
        "unique_liquidated_users": "int64",
        "unique_liquidators": "int64",
        "liquidated_collateral_value_usd": "float64",
        "liquidated_collateral_value_eth": "float64",
        "liquidation_debt_covered_value_usd": "float64",
        "liquidation_debt_covered_value_eth": "float64",
    }},
    "df_user_account_24h": {"columns": {
        "time_bucket": "object",
        "avg_total_collateral_base": "float64",
        "avg_total_debt_base": "float64",
        "avg_available_borrows_base": "float64",
        "avg_current_liquidation_threshold": "float64",
        "avg_ltv": "float64",
        "sampled_user_count": "int64",
        "account_data_call_count": "int64",
        "user_state_observed": "bool",  # bool audit column added by ffill_state_columns (ERR-17)
        "user_state_age": "float64",  # buckets since the last real observation
    }},
    "df_user_account_2h": {"columns": {
        "time_bucket": "object",
        "avg_total_collateral_base": "float64",
        "avg_total_debt_base": "float64",
        "avg_available_borrows_base": "float64",
        "avg_current_liquidation_threshold": "float64",
        "avg_ltv": "float64",
        "sampled_user_count": "int64",
        "account_data_call_count": "int64",
    }},
    "df_user_account_7d": {"columns": {
        "time_bucket": "object",
        "avg_total_collateral_base": "float64",
        "avg_total_debt_base": "float64",
        "avg_available_borrows_base": "float64",
        "avg_current_liquidation_threshold": "float64",
        "avg_ltv": "float64",
        "sampled_user_count": "int64",
        "account_data_call_count": "int64",
    }},
    "flashloan": {"columns": {
        "time_bucket": "object",
        "flashloan_amount_value_usd": "float64",
        "flashloan_amount_value_eth": "float64",
        "flashloan_premium_value_usd": "float64",
        "flashloan_premium_value_eth": "float64",
        "flashloan_tx_count": "int64",
        "unique_flashloan_initiators": "int64",
        "no_open_debt_flashloan_tx_count": "int64",
        "variable_flashloan_tx_count": "int64",
    }},
    "liquidation": {"columns": {
        "time_bucket": "object",
        "as_collateral_tx_count": "int64",
        "as_debt_tx_count": "int64",
        "liquidation_tx_count": "int64",
        "unique_liquidated_users": "int64",
        "unique_liquidators": "int64",
        "liquidated_collateral_value_usd": "float64",
        "liquidated_collateral_value_eth": "float64",
        "liquidation_debt_covered_value_usd": "float64",
        "liquidation_debt_covered_value_eth": "float64",
    }},
    "reserve_panel": {"columns": {
        "time_bucket": "object",
        "asset": "object",
        "asset_symbol": "object",
        # same frame as DF_common_1, same 10**27 object-dtype story — see the note there.
        "last_borrow_rate": ("float64", "object"),
        "liquidity_rate": "float64",
        "variable_borrow_rate": "float64",
        "liquidity_index": "float64",
        "variable_borrow_index": "float64",
        "update_count_liquidity": "int64",
    }},
    "supply_withdraw": {"columns": {
        "time_bucket": "object",
        "supply_tx_count": "int64",
        "withdrawal_tx_count": "int64",
        "unique_suppliers": "int64",
        "unique_withdraw_users": "int64",
        "supply_amount_value_usd": "float64",
        "supply_amount_value_eth": "float64",
        "withdrawal_amount_value_usd": "float64",
        "withdrawal_amount_value_eth": "float64",
    }},
    "user_account": {"columns": {
        "time_bucket": "object",
        "avg_total_collateral_base": "float64",
        "avg_total_debt_base": "float64",
        "avg_available_borrows_base": "float64",
        "avg_current_liquidation_threshold": "float64",
        "avg_ltv": "float64",
        "sampled_user_count": "int64",
        "account_data_call_count": "int64",
    }},
}

sc.assert_schema_file(SCHEMAS, name="transform_schema")
