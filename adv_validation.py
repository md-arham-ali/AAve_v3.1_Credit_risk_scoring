"""Advanced, finance-aware Great Expectations checks for the Aave V3.1 result tables.

-This module adds *domain* expectations chosen for the kind of financial value each column holds:
amounts, interest rates, liquidity indexes, basis-point configs, oracle prices,
health factors and activity counts.
-Runs under `.venv-ge` (Great Expectations 0.18.x, pandas 2.x) — same kernel as
`data_validation.py`.

The blockchain decimals problem (important)
-------------------------------------------
Raw on-chain values are uint256 / int256 and arrive as plain integer STRINGS far
larger than float64 can hold exactly (e.g. a RAY rate ~1e27, or the no-debt health
factor sentinel 2**256-1, 78 digits). pandas/GE silently upcast such object columns
to float64 during arithmetic, corrupting every digit below ~2**53. So:
  * magnitude / range / shape checks stay in GE (the boundaries sit far from the
    data, so float rounding cannot flip them), but
  * every EXACT cross-column identity on big integers is computed in pure Python
    `int` (`_to_int`), never with pandas object arithmetic.
Those exact checks are reported next to the GE results, tagged `custom_*`.

Single-column diagnostics (values + plots)
------------------------------------------
Besides the per-table suites, this module exposes three standalone, decimals-safe
column checks that each take ``(df, column)`` and return a dict of values plus an
optional matplotlib plot: ``negative_value_check`` (any negatives + how many),
``range_check`` (in-bounds vs out-of-bounds), and ``deviation_score`` (per-row
min-max score in [-1, 1]). matplotlib is imported lazily inside the plot helpers,
so the value computations still work in environments without it.

Statistical pre-EDA validation (univariate, finance-aware)
----------------------------------------------------------
``statistical_validation(df, table_name=None)`` profiles every numeric column with the
descriptive statistics and data-quality flags that is to be settled *before* EDA:
completeness (null %), zero-inflation, cardinality, robust location/scale (median, IQR,
MAD), quantiles, dispersion (CV), distribution shape (skewness / excess kurtosis) and
outlier counts (Tukey IQR fence + MAD modified z-score). It is deliberately univariate —
no correlations, relationships, hypotheses or plots; that is EDA, not validation.
Decimals-safe (uint256/RAY/WAD strings parsed exactly) and the 2**256-1 no-debt
health-factor sentinel is counted then excluded so it can't distort the percentiles.
Returns a tidy one-row-per-column frame, saved to
``validation_results/<table>__stat_validation.csv``.

Transformed-frame validation (model-ready frames, Tiers 0-4)
------------------------------------------------------------
``validate_transformed_final(df)`` and ``validate_reserve_panel(df)`` validate the two
post-transform frames in ``transformed_data/`` against ``context/data_val.md``. Unlike the
suites above (raw uint256 strings, GE), these frames are already scaled floats, so the
checks are plain pandas/Python and return NUMERIC results — pass-rate %, counts, and the
offending columns + ``time_bucket`` keys. Five per-tier functions (``tf_tier0_schema`` …
``tf_tier4_temporal``) each return a one-row-per-check DataFrame so the notebook can show
one tier per cell; the entry points concatenate them and save
``validation_results/<frame>__transform_validation.csv``.

To be scaled as required after addition of other feature tables.
"""

import math
import sys
from decimal import Decimal

import pandas as pd

# Reuse the loaders/keys already defined for the basic suite (no duplication).
from data_validation import load_csv, table_name_from_path, key_columns, ADDRESS_COLS

# --------------------------------------------------------------------------- #
# Constants — on-chain fixed-point scales and shared formats
# --------------------------------------------------------------------------- #
RAY = 10 ** 27               # Aave rate/index fixed point (1.0 == 1e27)
WAD = 10 ** 18               # Aave health-factor fixed point (1.0 == 1e18)
UINT256_MAX = 2 ** 256 - 1   # health-factor sentinel for accounts with no debt
BPS_MAX = 10_000             # 100% in basis points
HF_SANE_CAP = 10 ** 30       # any non-sentinel health factor must sit below this
MAX_BLOCK = 50_000_000       # generous upper bound for an Ethereum block number, though not needed

WINDOW_6H_REGEX = r"^(2025-1[12]|2026-01)-\d{2} ([01]\d|2[0-3]):00:00"  # window + 6h grid
ADDRESS_REGEX = r"^0x[0-9a-f]{40}$"          # 20-byte lower-hex address, not checking hashes here
SYMBOL_REGEX = r"^[A-Za-z0-9._+\-]{1,40}$"   # token tickers incl. PT-style names

# uint256 columns that are non-negative by construction (amounts, rates, indexes, HF), for pre transformation tests
BIGINT_NONNEG = {
    "supply_amount_raw", "withdrawal_amount_raw",
    "borrow_amount_raw", "repay_amount_raw", "last_borrow_rate",
    "liquidity_rate", "variable_borrow_rate", "stable_borrow_rate",
    "liquidity_index", "variable_borrow_index",
    "liquidation_debt_covered_raw", "liquidated_collateral_raw",
    "flashloan_amount_raw", "flashloan_premium_raw",
    "min_health_factor", "max_health_factor",
}
# int256 columns that are signed (net flows) — only shape + identity, never >= 0, 
# to be tested on transformed data not raw oon chain data due to decimal mismatch among assets
BIGINT_SIGNED = {"net_supply_flow_raw", "net_debt_flow_raw"}
BIGINT_COLS = BIGINT_NONNEG | BIGINT_SIGNED

RESULT_COLS = ["table", "expectation", "column", "success",
               "element_count", "unexpected_count", "unexpected_percent", "note"] # to display a structured final result


# --------------------------------------------------------------------------- #
# Big-integer parsing (the decimals-safe primitives)
# --------------------------------------------------------------------------- #
def _to_int(v):
    """Parse one cell to a Python int (arbitrary precision) or None. Never floats."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        return int(v) if v.is_integer() else None      # exact only
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return None
    body = s[1:] if s.startswith("-") else s
    if not body.isdigit():
        return None
    return int(s)


def to_int_series(s):
    """Vectorless parse of a column to object-dtype Python ints (keeps None)."""
    return s.map(_to_int)


def _to_number(v):
    """Parse one cell to a Python number or None — exact int for integer values,
    float for decimals. Big integer STRINGS stay exact ints (decimals-safe), so this
    works for both raw uint256 amount columns and ordinary float columns (prices/bps).
    """
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        return None if pd.isna(v) else v
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return None
    body = s[1:] if s.startswith("-") else s
    if body.isdigit():
        return int(s)                       # exact big int, no float corruption
    try:
        return float(s)                     # genuine decimal value
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# Result collector — wraps GE results and custom checks into one tidy table
# --------------------------------------------------------------------------- #
def _col_label(kw):
    if "column" in kw:
        return kw["column"]
    if "column_A" in kw:
        return f'{kw.get("column_A")},{kw.get("column_B")}'
    if "column_list" in kw:
        return ",".join(kw["column_list"])
    return ""


class Checks:
    """Runs GE expectations (on the prepared frame) and custom int checks; records both."""

    def __init__(self, gdf, table, raw_df):
        self.g = gdf          # GE PandasDataset built from the prepared (parsed) frame
        self.table = table
        self.raw = raw_df     # original frame (strings) for exact big-int checks
        self.cols = set(raw_df.columns)
        self.rows = []

    def ge(self, note, method, **kw):
        """Call a GE 0.18 expectation by name (kwargs only) and record its result."""
        try:
            res = getattr(self.g, method)(**kw)
            d = res.result or {}
            self.rows.append({
                "table": self.table,
                "expectation": res.expectation_config.expectation_type,
                "column": _col_label(kw),
                "success": bool(res.success),
                "element_count": d.get("element_count"),
                "unexpected_count": d.get("unexpected_count"),
                "unexpected_percent": d.get("unexpected_percent"),
                "note": note,
            })
        except Exception as exc:                     # never let one check kill the run
            self.rows.append({
                "table": self.table, "expectation": method, "column": _col_label(kw),
                "success": False, "element_count": None, "unexpected_count": None,
                "unexpected_percent": None, "note": f"{note} [ERROR: {exc}]",
            })

    def custom(self, note, name, column, n_checked, n_bad):
        """Record a pure-Python (non-GE) finance check result."""
        self.rows.append({
            "table": self.table, "expectation": name, "column": column,
            "success": (n_bad == 0), "element_count": n_checked,
            "unexpected_count": n_bad,
            "unexpected_percent": (round(100.0 * n_bad / n_checked, 4) if n_checked else None),
            "note": note,
        })

    def has(self, *names):
        """True only if every named column is present (column-aware gating)."""
        return all(n in self.cols for n in names)


# --------------------------------------------------------------------------- #
# Exact big-int finance checks (pure Python int — decimals-safe)
# --------------------------------------------------------------------------- #
def _net_flow_violations(raw, pos, neg, net):
    """net == coalesce(pos,0) - coalesce(neg,0); exact int (float would corrupt it)."""
    n = bad = 0
    for p, m, t in zip(raw[pos], raw[neg], raw[net]):
        ti = _to_int(t)
        if ti is None:
            continue
        n += 1
        if ti != (_to_int(p) or 0) - (_to_int(m) or 0):
            bad += 1
    return n, bad


def _pair_le_violations(raw, small, big):
    """small <= big over non-null rows (e.g. flashloan premium <= principal)."""
    n = bad = 0
    for a, b in zip(raw[small], raw[big]):
        ai, bi = _to_int(a), _to_int(b)
        if ai is None or bi is None:
            continue
        n += 1
        if ai > bi:
            bad += 1
    return n, bad


def _monotonic_violations(raw, asset_col, time_col, value_col):
    """Per asset, index must never decrease over time (interest only accrues up)."""
    sub = raw[[asset_col, time_col, value_col]].copy()
    sub["_v"] = sub[value_col].map(_to_int)
    sub = sub.sort_values([asset_col, time_col], kind="mergesort")  # time string sorts chronologically
    n = bad = 0
    prev_a = prev_v = None
    for a, v in zip(sub[asset_col], sub["_v"]):
        if v is not None and a == prev_a and prev_v is not None:
            n += 1
            if v < prev_v:
                bad += 1
        prev_a, prev_v = a, v
    return n, bad


def _symbol_consistency_violations(raw, asset_col, symbol_col):
    """Each asset address must map to at most one symbol (no asset with two tickers)."""
    sub = raw[[asset_col, symbol_col]].dropna()
    per_asset = sub.groupby(asset_col)[symbol_col].nunique()
    return int(per_asset.shape[0]), int((per_asset > 1).sum())


# --------------------------------------------------------------------------- #
# prepare() — build the frame GE validates against
# --------------------------------------------------------------------------- #
def prepare(df):
    """Copies df, parse big-int columns to Python ints, and add count-partition helpers."""
    p = df.copy()
    for c in BIGINT_COLS & set(p.columns):
        p[c] = to_int_series(p[c])                       # decimals-safe ints for GE range checks

    # integer count partitions (small ints, float-safe) for GE pair-equality
    if {"stable_borrow_tx_count", "variable_borrow_tx_count", "borrow_tx_count"} <= set(p.columns):
        p["_borrow_mode_sum"] = p["stable_borrow_tx_count"] + p["variable_borrow_tx_count"]
    if {"no_open_debt_flashloan_tx_count", "stable_flashloan_tx_count",
        "variable_flashloan_tx_count", "flashloan_tx_count"} <= set(p.columns):
        p["_fl_mode_sum"] = (p["no_open_debt_flashloan_tx_count"]
                             + p["stable_flashloan_tx_count"]
                             + p["variable_flashloan_tx_count"])
    return p


# --------------------------------------------------------------------------- #
# Common checks — applied to every table, column-aware
# --------------------------------------------------------------------------- #
def common_expectations(chk):
    cols = chk.cols

    chk.ge("table must not be empty", "expect_table_row_count_to_be_between", min_value=1)

    key = key_columns(chk.raw)                                    # composite primary key
    for c in key:                                                 # key parts are always populated
        chk.ge("key column never null", "expect_column_values_to_not_be_null", column=c)
    if len(key) == 1:
        chk.ge("one row per key (no dup PK)", "expect_column_values_to_be_unique", column=key[0])
    elif len(key) > 1:
        chk.ge("one row per composite key", "expect_compound_columns_to_be_unique", column_list=key)

    if "time_bucket" in cols:                                     # inside window + on the 6h grid
        chk.ge("time_bucket in window & on 6h grid",
               "expect_column_values_to_match_regex", column="time_bucket", regex=WINDOW_6H_REGEX)

    for c in ADDRESS_COLS:                                        # 20-byte hex asset addresses
        if c in cols:
            chk.ge("valid 0x-address", "expect_column_values_to_match_regex",
                   column=c, regex=ADDRESS_REGEX)

    for c in [c for c in cols if c.endswith("symbol")]:          # token ticker shape (nulls allowed)
        chk.ge("plausible token symbol", "expect_column_values_to_match_regex",
               column=c, regex=SYMBOL_REGEX)
        # the address paired with this symbol ('asset_symbol'->'asset'; bare 'symbol'->'asset')
        asset_col = c[: -len("_symbol")] if c.endswith("_symbol") else ("asset" if c == "symbol" else None)
        if asset_col in cols:                                    # each asset maps to one symbol
            n, bad = _symbol_consistency_violations(chk.raw, asset_col, c)
            chk.custom("each asset maps to one symbol", "custom_symbol_single_valued",
                       f"{asset_col},{c}", n, bad)

    for c in cols:                                               # activity counts are non-negative
        if c.endswith("_count"):
            chk.ge("count is non-negative", "expect_column_values_to_be_between",
                   column=c, min_value=0)
        elif c.endswith("_block"):                              # block numbers: 0..mainnet height
            chk.ge("block number in sane range", "expect_column_values_to_be_between",
                   column=c, min_value=0, max_value=MAX_BLOCK)

    for c in BIGINT_COLS & cols:                                # raw value parses as an integer
        n = int(chk.raw[c].notna().sum())
        bad = int(sum(1 for v in chk.raw[c] if pd.notna(v) and _to_int(v) is None))
        chk.custom("uint256/int256 raw value is well-formed",
                   "custom_bigint_well_formed", c, n, bad)


# --------------------------------------------------------------------------- #
# Per-table suites (intermediate -> advanced, finance-specific)
# --------------------------------------------------------------------------- #
def adv_supply_withdraw(chk):
    for c in ("supply_amount_raw", "withdrawal_amount_raw"):     # token amounts >= 0
        chk.ge("supplied/withdrawn amount >= 0", "expect_column_values_to_be_between",
               column=c, min_value=0)
    if chk.has("net_supply_flow_raw", "supply_amount_raw", "withdrawal_amount_raw"):
        n, bad = _net_flow_violations(chk.raw, "supply_amount_raw",
                                      "withdrawal_amount_raw", "net_supply_flow_raw")
        chk.custom("net flow == supply - withdraw (signed integrity)",
                   "custom_net_flow_integrity", "net_supply_flow_raw", n, bad)
    # distinct users cannot exceed tx count (HLL approx -> small tolerance)
    for u, t in (("unique_suppliers", "supply_tx_count"),
                 ("unique_withdraw_users", "withdrawal_tx_count")):
        if chk.has(u, t):
            chk.ge("unique users <= tx count", "expect_column_pair_values_A_to_be_greater_than_B",
                   column_A=t, column_B=u, or_equal=True, mostly=0.99)


def adv_borrow_repay(chk):
    for c in ("borrow_amount_raw", "repay_amount_raw"):          # token amounts >= 0
        chk.ge("borrowed/repaid amount >= 0", "expect_column_values_to_be_between",
               column=c, min_value=0)
    if chk.has("net_debt_flow_raw", "borrow_amount_raw", "repay_amount_raw"):
        n, bad = _net_flow_violations(chk.raw, "borrow_amount_raw",
                                      "repay_amount_raw", "net_debt_flow_raw")
        chk.custom("net debt == borrow - repay (signed integrity)",
                   "custom_net_flow_integrity", "net_debt_flow_raw", n, bad)
    if "last_borrow_rate" in chk.cols:                           # RAY rate: >=0, < ~500% APR
        chk.ge("borrow rate in [0, 5e27] (RAY, sane APR ceiling)",
               "expect_column_values_to_be_between",
               column="last_borrow_rate", min_value=0, max_value=5 * RAY)
    if "_borrow_mode_sum" in chk.g.columns:                      # every borrow is stable or variable
        chk.ge("stable + variable == total borrows",
               "expect_column_pair_values_to_be_equal",
               column_A="_borrow_mode_sum", column_B="borrow_tx_count")
    if "stable_borrow_tx_count" in chk.cols:                     # stable mode deprecated on mainnet
        chk.ge("stable-rate borrows == 0 (deprecated)",
               "expect_column_values_to_be_between",
               column="stable_borrow_tx_count", min_value=0, max_value=0)
    for u, t in (("unique_borrowers", "borrow_tx_count"),
                 ("unique_repayers", "repay_tx_count")):
        if chk.has(u, t):
            chk.ge("unique users <= tx count", "expect_column_pair_values_A_to_be_greater_than_B",
                   column_A=t, column_B=u, or_equal=True, mostly=0.99)


def adv_reserve_state(chk):
    for c in ("liquidity_rate", "variable_borrow_rate"):         # RAY rates >= 0
        if c in chk.cols:
            chk.ge("interest rate >= 0 (RAY)", "expect_column_values_to_be_between",
                   column=c, min_value=0)
    if "stable_borrow_rate" in chk.cols:                         # stable rate deprecated -> 0
        chk.ge("stable borrow rate == 0 (deprecated)", "expect_column_values_to_be_between",
               column="stable_borrow_rate", min_value=0, max_value=0)
    for c in ("liquidity_index", "variable_borrow_index"):       # indexes start at 1.0 and only grow
        if c in chk.cols:
            chk.ge("index >= 1.0 (RAY)", "expect_column_values_to_be_between",
                   column=c, min_value=RAY)
            if "asset" in chk.cols and "time_bucket" in chk.cols:  # never decreases per asset
                n, bad = _monotonic_violations(chk.raw, "asset", "time_bucket", c)
                chk.custom("index is monotonic non-decreasing per asset",
                           "custom_index_monotonic", c, n, bad)
    # NOTE: variable_borrow_index >= liquidity_index is deliberately NOT asserted — it is
    # false for borrowing-disabled collateral assets (e.g. sDAI), whose borrow index stays
    # pinned at 1.0 RAY while the liquidity index still grows.


def adv_reserve_config(chk):
    for c in ("supply_cap", "old_supply_cap", "borrow_cap", "old_borrow_cap",
              "debt_ceiling", "old_debt_ceiling"):               # caps/ceilings >= 0
        if c in chk.cols:
            chk.ge("cap / ceiling >= 0", "expect_column_values_to_be_between",
                   column=c, min_value=0)
    for c in ("reserve_factor", "old_reserve_factor", "liquidation_threshold", "ltv"):
        if c in chk.cols:                                        # basis-point fields in [0, 10000]
            chk.ge("bps value in [0, 10000]", "expect_column_values_to_be_between",
                   column=c, min_value=0, max_value=BPS_MAX)
    if "liquidation_bonus" in chk.cols:                          # 0 (non-collateral) OR 100%+extra in [10000, 12000]
        vals = [_to_int(v) for v in chk.raw["liquidation_bonus"]]
        n = sum(v is not None for v in vals)
        bad = sum(1 for v in vals if v is not None and not (v == 0 or BPS_MAX <= v <= 12_000))
        chk.custom("liquidation bonus is 0 or in [10000, 12000] bps",
                   "custom_liquidation_bonus_valid", "liquidation_bonus", n, bad)
    if chk.has("liquidation_threshold", "ltv"):                  # core Aave risk invariant
        chk.ge("ltv <= liquidation_threshold",
               "expect_column_pair_values_A_to_be_greater_than_B",
               column_A="liquidation_threshold", column_B="ltv", or_equal=True)


def adv_liquidation(chk):
    for c in ("liquidation_debt_covered_raw", "liquidated_collateral_raw"):  # amounts >= 0
        if c in chk.cols:
            chk.ge("liquidation amount >= 0", "expect_column_values_to_be_between",
                   column=c, min_value=0)
    if chk.has("receive_atoken_count", "liquidation_tx_count"):  # subset of liquidations
        chk.ge("receive-aToken count <= liquidations",
               "expect_column_pair_values_A_to_be_greater_than_B",
               column_A="liquidation_tx_count", column_B="receive_atoken_count", or_equal=True)
    for u in ("unique_liquidated_users", "unique_liquidators"):  # distinct actors <= events
        if chk.has(u, "liquidation_tx_count"):
            chk.ge("unique actors <= liquidations",
                   "expect_column_pair_values_A_to_be_greater_than_B",
                   column_A="liquidation_tx_count", column_B=u, or_equal=True, mostly=0.99)


def adv_flashloan(chk):
    for c in ("flashloan_amount_raw", "flashloan_premium_raw"):  # principal & fee >= 0
        if c in chk.cols:
            chk.ge("flashloan amount/premium >= 0", "expect_column_values_to_be_between",
                   column=c, min_value=0)
    if chk.has("flashloan_premium_raw", "flashloan_amount_raw"):  # fee is a fraction of principal
        n, bad = _pair_le_violations(chk.raw, "flashloan_premium_raw", "flashloan_amount_raw")
        chk.custom("premium <= principal", "custom_premium_le_amount",
                   "flashloan_premium_raw,flashloan_amount_raw", n, bad)
    if "_fl_mode_sum" in chk.g.columns:                          # every flashloan has mode 0/1/2
        chk.ge("mode 0+1+2 == total flashloans", "expect_column_pair_values_to_be_equal",
               column_A="_fl_mode_sum", column_B="flashloan_tx_count")
    if chk.has("unique_flashloan_initiators", "flashloan_tx_count"):
        chk.ge("unique initiators <= flashloans",
               "expect_column_pair_values_A_to_be_greater_than_B",
               column_A="flashloan_tx_count", column_B="unique_flashloan_initiators",
               or_equal=True, mostly=0.99)


def adv_user_account(chk):
    for c in ("avg_total_collateral_base", "avg_total_debt_base",
              "avg_available_borrows_base"):                     # USD-base (8dp) >= 0
        if c in chk.cols:
            chk.ge("USD-base value >= 0", "expect_column_values_to_be_between",
                   column=c, min_value=0)
    for c in ("avg_current_liquidation_threshold", "avg_ltv"):  # avg bps in [0, 10000]
        if c in chk.cols:
            chk.ge("avg bps in [0, 10000]", "expect_column_values_to_be_between",
                   column=c, min_value=0, max_value=BPS_MAX)
    if chk.has("avg_ltv", "avg_current_liquidation_threshold"):  # market-avg LTV <= liq threshold
        chk.ge("avg_ltv <= avg_current_liquidation_threshold",
               "expect_column_pair_values_A_to_be_greater_than_B",
               column_A="avg_current_liquidation_threshold", column_B="avg_ltv", or_equal=True)
    if chk.has("avg_total_collateral_base", "avg_total_debt_base"):  # overcollateralized (soft)
        chk.ge("avg collateral >= avg debt (aggregate, soft)",
               "expect_column_pair_values_A_to_be_greater_than_B",
               column_A="avg_total_collateral_base", column_B="avg_total_debt_base",
               or_equal=True, mostly=0.90)
    if chk.has("min_health_factor", "max_health_factor"):       # ordering, exact 256-bit
        n = bad = 0
        for lo, hi in zip(chk.raw["min_health_factor"], chk.raw["max_health_factor"]):
            li, hi_ = _to_int(lo), _to_int(hi)
            if li is None or hi_ is None:
                continue
            n += 1
            bad += (li > hi_)
        chk.custom("min health factor <= max health factor",
                   "custom_hf_min_le_max", "min_health_factor,max_health_factor", n, bad)
        # every HF is a real WAD value OR exactly the no-debt sentinel (2**256-1)
        for c in ("min_health_factor", "max_health_factor"):
            vals = [_to_int(v) for v in chk.raw[c]]
            n2 = sum(v is not None for v in vals)
            bad2 = sum(1 for v in vals if v is not None
                       and not (v < HF_SANE_CAP or v == UINT256_MAX))
            chk.custom("HF is sane WAD or the no-debt sentinel",
                       "custom_hf_sentinel_or_sane", c, n2, bad2)
    if chk.has("sampled_user_count", "account_data_call_count"):
        chk.ge("sampled users <= account-data calls",
               "expect_column_pair_values_A_to_be_greater_than_B",
               column_A="account_data_call_count", column_B="sampled_user_count", or_equal=True)


def adv_collateral_toggle(chk):
    for u, t in (("unique_collateral_enable_users", "collateral_enabled_count"),
                 ("unique_collateral_disable_users", "collateral_disabled_count")):
        if chk.has(u, t):                                       # distinct togglers <= toggles
            chk.ge("unique users <= toggle count",
                   "expect_column_pair_values_A_to_be_greater_than_B",
                   column_A=t, column_B=u, or_equal=True, mostly=0.99)


def adv_oracle_price(chk):
    if "decimals" in chk.cols:                                  # token decimals sane (nulls ok)
        chk.ge("decimals in [0, 36]", "expect_column_values_to_be_between",
               column="decimals", min_value=0, max_value=36)
    for c in ("avg_price_usd", "avg_price_eth", "avg_price_weth"):  # prices strictly positive
        if c in chk.cols:
            chk.ge("price > 0", "expect_column_values_to_be_between",
                   column=c, min_value=0, strict_min=True)
    if "avg_price_usd" in chk.cols:                             # guard against absurd USD prices
        chk.ge("avg USD price < 1e7 (sanity)", "expect_column_values_to_be_between",
               column="avg_price_usd", min_value=0, max_value=1e7)
    if chk.has("avg_price_eth", "avg_price_weth"):             # ETH & WETH share one feed -> equal
        chk.ge("price in ETH == price in WETH", "expect_column_pair_values_to_be_equal",
               column_A="avg_price_eth", column_B="avg_price_weth")
    if "price_points" in chk.cols:                             # <= 6 hourly obs per 6h bucket
        chk.ge("price_points in [1, 6]", "expect_column_values_to_be_between",
               column="price_points", min_value=1, max_value=6)


# --------------------------------------------------------------------------- #
# Dispatch — pick the suite by a column unique to each table
# --------------------------------------------------------------------------- #
TABLE_SUITES = {
    "supply_withdraw": adv_supply_withdraw,
    "borrow_repay": adv_borrow_repay,
    "reserve_state_rates": adv_reserve_state,
    "reserve_config": adv_reserve_config,
    "liquidation": adv_liquidation,
    "flashloan": adv_flashloan,
    "user_account": adv_user_account,
    "collateral_toggle": adv_collateral_toggle,
    "oracle_price": adv_oracle_price,
}

# (signature column -> table name); first match wins
_SIGNATURES = [
    ("supply_amount_raw", "supply_withdraw"),
    ("borrow_amount_raw", "borrow_repay"),
    ("liquidity_index", "reserve_state_rates"),
    ("supply_cap", "reserve_config"),
    ("liquidated_collateral_raw", "liquidation"),   # per-asset grain: no collateral_asset col anymore
    ("flashloan_amount_raw", "flashloan"),
    ("min_health_factor", "user_account"),
    ("collateral_enabled_count", "collateral_toggle"),
    ("avg_price_usd", "oracle_price"),
]


def detect_table(df):
    """Identify which table a frame is by a signature column; None if unrecognized."""
    cols = set(df.columns)
    for sig, name in _SIGNATURES:
        if sig in cols:
            return name
    return None


# --------------------------------------------------------------------------- #
# Entry point — run common + table-specific checks; return (and optionally save)
# --------------------------------------------------------------------------- #
def validate_table_advanced(df, table_name=None, results_dir="validation_results", save=True):
    """Run the advanced suite for one table and return a tidy results DataFrame.

    Args:
        df:          the result table as a pandas DataFrame (read with default dtypes).
        table_name:  optional label; auto-detected from the columns when omitted.
        results_dir: where the per-table CSV is written.
        save:        set False to skip writing the CSV.
    """
    name = table_name or detect_table(df) or "unknown"
    prepared = prepare(df)

    import great_expectations as ge                    # lazy: only needed here, gx installed in second venv, venv-ge
    gdf = ge.from_pandas(prepared)

    chk = Checks(gdf, name, df)
    common_expectations(chk)                            # shared checks first
    suite = TABLE_SUITES.get(name)
    if suite:
        suite(chk)                                      # then the table-specific finance checks

    out = pd.DataFrame(chk.rows, columns=RESULT_COLS)
    if save:
        from pathlib import Path
        folder = Path(results_dir)
        folder.mkdir(parents=True, exist_ok=True)
        out.to_csv(folder / f"{name}__adv_ge_results.csv", index=False)
    return out


# --------------------------------------------------------------------------- #
# Single-column diagnostics — values + plots, decimals-safe, take (df, column)
# --------------------------------------------------------------------------- #
def _numeric_pairs(df, column):
    """Return (indices, values) for non-null parseable cells of df[column].

    Big-int strings stay exact Python ints; decimals parse to float (decimals-safe).
    """
    idx, vals = [], []
    for i, v in zip(df.index, df[column]):
        num = _to_number(v)
        if num is not None:
            idx.append(i)
            vals.append(num)
    return idx, vals


def _get_ax(ax):
    """Return a matplotlib Axes (lazy import — only needed when plotting)."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    return ax


def negative_value_check(df, column, plot=True, max_examples=10, ax=None):
    """Report negative values in df[column]: whether any exist and how many.

    Decimals-safe (big-int strings parsed as exact Python ints, so a RAY rate or the
    2**256-1 health-factor sentinel is never misread). Returns a dict of values; with
    plot=True it also draws a non-negative vs negative count bar and adds it as 'ax'.
    """
    idx, vals = _numeric_pairs(df, column)
    n_checked = len(vals)
    negatives = [(i, v) for i, v in zip(idx, vals) if v < 0]
    n_negative = len(negatives)

    result = {
        "column": column,
        "n_checked": n_checked,
        "n_negative": n_negative,
        "negative_pct": round(100.0 * n_negative / n_checked, 4) if n_checked else None,
        "min_value": str(min(vals)) if vals else None,
        "has_negative": n_negative > 0,
        "success": n_negative == 0,
        "examples": [{"index": i, "value": str(v)} for i, v in negatives[:max_examples]],
    }
    if plot:
        ax = _get_ax(ax)
        bars = ax.bar(["non-negative", "negative"],
                      [n_checked - n_negative, n_negative],
                      color=["#4c72b0", "#c44e52"])
        ax.bar_label(bars)
        ax.set_title(f"{column}: negative-value check")
        ax.set_ylabel("row count")
        result["ax"] = ax
    return result


def range_check(df, column, min_value=None, max_value=None, plot=True,
                max_examples=10, ax=None):
    """Check df[column] lies within [min_value, max_value] (each bound optional).

    With no bounds it just describes the observed range. Decimals-safe. Returns a dict
    of values (counts below / above / within, observed min & max, offending examples);
    with plot=True it draws a histogram with the bounds marked and adds it as 'ax'.
    """
    idx, vals = _numeric_pairs(df, column)
    n_checked = len(vals)
    below = [(i, v) for i, v in zip(idx, vals) if min_value is not None and v < min_value]
    above = [(i, v) for i, v in zip(idx, vals) if max_value is not None and v > max_value]
    n_out = len(below) + len(above)

    result = {
        "column": column,
        "n_checked": n_checked,
        "min_value": min_value,
        "max_value": max_value,
        "observed_min": str(min(vals)) if vals else None,
        "observed_max": str(max(vals)) if vals else None,
        "n_below_min": len(below),
        "n_above_max": len(above),
        "n_within": n_checked - n_out,
        "out_of_range_pct": round(100.0 * n_out / n_checked, 4) if n_checked else None,
        "success": n_out == 0,
        "examples": (
            [{"index": i, "value": str(v), "side": "below_min"} for i, v in below[:max_examples]]
            + [{"index": i, "value": str(v), "side": "above_max"} for i, v in above[:max_examples]]
        ),
    }
    if plot:
        ax = _get_ax(ax)
        ax.hist([float(v) for v in vals],
                bins=min(50, max(10, int(n_checked ** 0.5))) if n_checked else 10,
                color="#4c72b0", alpha=0.85)
        if min_value is not None:
            ax.axvline(float(min_value), color="#c44e52", linestyle="--", label=f"min={min_value}")
        if max_value is not None:
            ax.axvline(float(max_value), color="#c44e52", linestyle="--", label=f"max={max_value}")
        ax.set_title(f"{column}: range check")
        ax.set_xlabel(column)
        ax.set_ylabel("frequency")
        if min_value is not None or max_value is not None:
            ax.legend()
        result["ax"] = ax
    return result


def deviation_score(df, column, plot=True, ax=None):
    """Per-row min-max deviation score in [-1, 1]:  2*(x - min)/(max - min) - 1.

    -1 = column minimum, +1 = column maximum, 0 = range midpoint; a constant column
    scores 0 everywhere. Big integers are divided via Decimal then cast to float, so
    the score stays accurate even on uint256 columns. Returns a dict of values with
    the per-row scores under 'scores' (a pandas Series indexed like df); with plot=True
    it draws a histogram of the scores and adds it as 'ax'.
    """
    idx, vals = _numeric_pairs(df, column)
    n_checked = len(vals)

    if n_checked == 0:
        result = {
            "column": column, "n_checked": 0, "observed_min": None, "observed_max": None,
            "score_min": None, "score_max": None, "score_mean": None, "is_constant": None,
            "scores": pd.Series(dtype=float, name=f"{column}__dev_score"),
        }
        if plot:
            result["ax"] = None
        return result

    vmin, vmax = min(vals), max(vals)
    span = Decimal(str(vmax)) - Decimal(str(vmin))
    if span == 0:                                    # constant column -> midpoint score 0
        score_vals = [0.0] * n_checked
    else:
        score_vals = [float(2 * (Decimal(str(v)) - Decimal(str(vmin))) / span - 1) for v in vals]
    scores = pd.Series(score_vals, index=idx, name=f"{column}__dev_score")

    result = {
        "column": column,
        "n_checked": n_checked,
        "observed_min": str(vmin),
        "observed_max": str(vmax),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_mean": float(scores.mean()),
        "is_constant": span == 0,
        "scores": scores,
    }
    if plot:
        ax = _get_ax(ax)
        ax.hist(scores.to_numpy(), bins=40, range=(-1, 1), color="#55a868", alpha=0.85)
        ax.axvline(0.0, color="#333333", linestyle=":", linewidth=1)
        ax.set_title(f"{column}: min-max deviation score [-1, 1]")
        ax.set_xlabel("deviation score   (-1 = min, +1 = max)")
        ax.set_ylabel("frequency")
        ax.set_xlim(-1.05, 1.05)
        result["ax"] = ax
    return result


# --------------------------------------------------------------------------- #
# Statistical pre-EDA validation — univariate profile + data-quality flags
# --------------------------------------------------------------------------- #
# Advisory thresholds (tunable). These raise FLAGS, not failures: heavy tails and
# zero-inflation are normal for DeFi data, they just want noting before EDA.
NULL_FRAC_WARN = 0.20         # > 20% missing -> completeness concern
ZERO_FRAC_WARN = 0.50         # > 50% zeros -> zero-inflated / sparse activity
QUASI_CONSTANT_FRAC = 0.95    # one value covers > 95% of rows -> near-constant
HEAVY_TAIL_KURT = 3.0         # excess kurtosis above this -> fat-tailed
HEAVY_TAIL_SKEW = 2.0         # |skewness| above this -> strongly one-sided
OUTLIER_IQR_K = 1.5           # Tukey fence multiplier (q25 - k*IQR, q75 + k*IQR)
OUTLIER_MODZ = 3.5            # modified (MAD) z-score cutoff (Iglewicz-Hoaglin)
NUMERIC_PARSE_MIN = 0.80      # treat a column as numeric if >= 80% of present cells parse

STAT_COLS = [
    "table", "column",
    "n", "n_checked", "n_null", "null_pct",
    "n_zero", "zero_pct", "n_negative", "negative_pct",
    "n_unique", "unique_pct", "n_sentinel",
    "is_constant", "is_quasi_constant",
    "mean", "std", "cv",
    "min", "p01", "p05", "q25", "median", "q75", "p95", "p99", "max",
    "iqr", "mad", "skewness", "excess_kurtosis",
    "n_outliers_iqr", "outlier_iqr_pct", "n_outliers_mad", "outlier_mad_pct",
    "heavy_tailed", "flags", "success",
]


def _pct(part, whole):
    """Percentage with a safe zero-denominator (None when nothing was counted)."""
    return round(100.0 * part / whole, 4) if whole else None


def _finite(x):
    """Pass a float through, or None if it is NaN/inf (e.g. an overflowed moment)."""
    return x if (x is not None and math.isfinite(x)) else None


def _series_stats(s):
    """Robust + classical univariate stats for one sentinel-free float Series.

    Robust order statistics (median, IQR, MAD, quantiles) lead because DeFi amount and
    rate columns are heavy-tailed, where a single whale row distorts mean/std. Higher
    moments are float64 and reported only when finite. Outliers are counted two ways:
    the Tukey IQR fence and the MAD-based modified z-score (both robust to fat tails).
    """
    n = int(len(s))
    qs = s.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    med = float(qs.loc[0.50])
    q25, q75 = float(qs.loc[0.25]), float(qs.loc[0.75])
    iqr = q75 - q25
    mad = float((s - med).abs().median())            # median absolute deviation (robust scale)
    mean = float(s.mean())
    std = float(s.std(ddof=1)) if n > 1 else 0.0
    skew = _finite(float(s.skew())) if n > 2 else None
    kurt = _finite(float(s.kurt())) if n > 3 else None   # pandas .kurt() is excess (Fisher)

    lo, hi = q25 - OUTLIER_IQR_K * iqr, q75 + OUTLIER_IQR_K * iqr
    n_out_iqr = int(((s < lo) | (s > hi)).sum()) if iqr > 0 else 0
    n_out_mad = (int((0.6745 * (s - med).abs() / mad > OUTLIER_MODZ).sum())
                 if mad > 0 else 0)

    heavy = bool((kurt is not None and kurt > HEAVY_TAIL_KURT)
                 or (skew is not None and abs(skew) > HEAVY_TAIL_SKEW))
    return {
        "mean": mean, "std": std,
        "cv": (std / abs(mean)) if mean != 0 else None,
        "min": float(s.min()), "p01": float(qs.loc[0.01]), "p05": float(qs.loc[0.05]),
        "q25": q25, "median": med, "q75": q75,
        "p95": float(qs.loc[0.95]), "p99": float(qs.loc[0.99]), "max": float(s.max()),
        "iqr": iqr, "mad": mad, "skewness": skew, "excess_kurtosis": kurt,
        "n_outliers_iqr": n_out_iqr, "outlier_iqr_pct": _pct(n_out_iqr, n),
        "n_outliers_mad": n_out_mad, "outlier_mad_pct": _pct(n_out_mad, n),
        "heavy_tailed": heavy,
    }


def _column_stats(df, column, table):
    """Profile one column; None for non-numeric (address/symbol/time/key) columns."""
    from collections import Counter

    series = df[column]
    n = int(len(series))
    n_null = int(series.isna().sum())
    n_present = n - n_null
    if n_present == 0:
        return None                                  # all-null: nothing to profile

    parsed = [x for x in (_to_number(v) for v in series) if x is not None]
    if not parsed or (len(parsed) / n_present) < NUMERIC_PARSE_MIN:
        return None                                  # categorical / address / symbol / time_bucket

    n_sentinel = sum(1 for x in parsed if x == UINT256_MAX)
    core = [x for x in parsed if x != UINT256_MAX]   # exact ints/floats, no-debt sentinel removed
    n_checked = len(core)

    n_zero = sum(1 for x in core if x == 0)
    n_neg = sum(1 for x in core if x < 0)
    n_unique = len(set(core))
    top_freq = max(Counter(core).values()) if core else 0
    is_constant = n_unique <= 1
    is_quasi = (not is_constant) and n_checked > 0 and (top_freq / n_checked) > QUASI_CONSTANT_FRAC

    row = {c: None for c in STAT_COLS}
    row.update({
        "table": table, "column": column,
        "n": n, "n_checked": n_checked, "n_null": n_null, "null_pct": _pct(n_null, n),
        "n_zero": n_zero, "zero_pct": _pct(n_zero, n_checked),
        "n_negative": n_neg, "negative_pct": _pct(n_neg, n_checked),
        "n_unique": n_unique, "unique_pct": _pct(n_unique, n_checked),
        "n_sentinel": n_sentinel,
        "is_constant": is_constant, "is_quasi_constant": is_quasi,
    })

    flags = []
    if (row["null_pct"] or 0) > NULL_FRAC_WARN * 100:
        flags.append("high_null")
    if n_checked and n_zero / n_checked > ZERO_FRAC_WARN:
        flags.append("zero_inflated")
    if is_constant:
        flags.append("constant")
    elif is_quasi:
        flags.append("quasi_constant")
    if n_sentinel:
        flags.append("no_debt_sentinel")

    if n_checked > 0 and not is_constant:            # distribution stats need a varying value
        stats = _series_stats(pd.Series(core, dtype="float64"))
        row.update(stats)
        if stats["heavy_tailed"]:
            flags.append("heavy_tailed")
        if stats["n_outliers_iqr"]:
            flags.append("iqr_outliers")
        row["success"] = True
    else:
        row["success"] = False                       # empty or zero-variance -> not EDA-ready

    row["flags"] = ";".join(flags)
    return row


def statistical_validation(df, table_name=None, columns=None,
                           results_dir="validation_results", save=True):
    """Pre-EDA statistical profile + data-quality flags for every numeric column.

    A validation gate, not EDA: it answers "is each column fit to explore?" with
    univariate descriptive statistics and flags only — no correlations, bivariate
    relationships, hypotheses or plots. Tuned for financial / DeFi data:

      * completeness (null %), zero-inflation (sparse activity) and cardinality,
      * robust location/scale (median, IQR, MAD) plus mean/std/CV,
      * quantiles p01/p05/q25/median/q75/p95/p99 and min/max,
      * distribution shape (skewness, excess kurtosis) with a heavy-tail flag,
      * outlier counts via the Tukey IQR fence and the MAD modified z-score.

    Decimals-safe: uint256/RAY/WAD strings are parsed exactly via ``_to_number`` and the
    2**256-1 no-debt health-factor sentinel is counted (``n_sentinel``) then excluded so
    it can't distort the percentiles. The distribution moments themselves run in float64
    (magnitude/shape only — never an exact identity), consistent with the rest of this
    module. Non-numeric columns (addresses, symbols, time_bucket, keys) are skipped.

    Args:
        df:          the result table as a pandas DataFrame.
        table_name:  optional label; auto-detected from the columns when omitted.
        columns:     optional subset of columns to profile (default: all columns).
        results_dir: where the per-table CSV is written.
        save:        set False to skip writing the CSV.

    Returns:
        a tidy DataFrame, one row per numeric column (``STAT_COLS``); ``success`` is
        False only for empty or zero-variance columns (everything else is advisory).
    """
    name = table_name or detect_table(df) or "unknown"
    cols = list(df.columns) if columns is None else list(columns)
    rows = [r for r in (_column_stats(df, c, name) for c in cols) if r is not None]

    out = pd.DataFrame(rows, columns=STAT_COLS)
    if save and len(out):
        from pathlib import Path
        folder = Path(results_dir)
        folder.mkdir(parents=True, exist_ok=True)
        out.to_csv(folder / f"{name}__stat_validation.csv", index=False)
    return out


# --------------------------------------------------------------------------- #
# Transformed-frame validation (the model-ready frames in transformed_data/)
# --------------------------------------------------------------------------- #
# The two post-transform frames have a different grain/schema from the raw
# query_result_data tables validated above, and are already scaled (no uint256
# strings), so these checks use plain pandas/Python int — exact here — and return
# tidy NUMERIC results (pass-rate %, counts, offending columns + time_buckets)
# rather than GE objects. They implement context/data_val.md, Tiers 0-4:
#   * DF_common_final — protocol-level 6h feature matrix; time_bucket is the PK.
#   * DF_common_1     — asset-level reserve panel keyed on (time_bucket, asset).
# One function per tier returns a DataFrame (one row per check); the per-tier
# results carry tier / check / columns / severity / n_checked / n_pass / n_fail /
# pass_rate_pct / anomaly_columns / anomaly_buckets / detail.

TRANSFORM_RESULT_COLS = [
    "tier", "check", "columns", "severity",
    "n_checked", "n_pass", "n_fail", "pass_rate_pct",
    "anomaly_columns", "anomaly_buckets", "detail",
]

TIME_TZ_REGEX = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)? UTC$"  # tz-explicit ts
GRID_6H_REGEX = r" (?:[01]\d|2[0-3]):00:00(?:\.0+)? UTC$"                 # 6h boundary
WEI_LOOKING_MIN = 1e15        # a USD/ETH value this large => decimals never applied
SENTINELS = (-1, 1e18, 999999999)   # magic values that must not leak into value fields
MAGNITUDE_JUMP_DEX = 1.0      # |Δlog10| >= this between consecutive buckets = 10x break
PRICE_SPREAD_TOL = 0.05       # cross-family implied-price spread tolerance (5%)


# --------------------------------------------------------------------------- #
# Tidy-result primitives (shared by every transformed-frame check)
# --------------------------------------------------------------------------- #
def _col_numbers(df, col):
    """Row-aligned list of parsed numbers (None for null / non-numeric), decimals-safe."""
    return [_to_number(v) for v in df[col]]


def _key_buckets(df, mask, key="time_bucket", cap=10):
    """(count, capped sample) of the key (or row index) where boolean mask is True."""
    src = df[key] if key in df.columns else df.index
    hits = [k for k, m in zip(src, mask) if m]
    return len(hits), hits[:cap]


def _t_record(tier, check, columns, n_checked, n_fail, severity="hard",
              anomaly_columns="", anomaly_buckets=None, detail=""):
    """Build one tidy result row (computes n_pass + pass_rate_pct)."""
    n_checked, n_fail = int(n_checked), int(n_fail)
    n_pass = n_checked - n_fail
    buckets = anomaly_buckets or []
    return {
        "tier": tier, "check": check, "columns": columns, "severity": severity,
        "n_checked": n_checked, "n_pass": n_pass, "n_fail": n_fail,
        "pass_rate_pct": _pct(n_pass, n_checked),
        "anomaly_columns": anomaly_columns,
        "anomaly_buckets": "; ".join(str(b) for b in buckets),
        "detail": detail,
    }


def _count_like_cols(df):
    """Activity-count columns: *_count, unique_*, sampled_user_count."""
    return [c for c in df.columns
            if c.endswith("_count") or c.startswith("unique_") or c == "sampled_user_count"]


def _value_cols(df):
    """Valued amount columns: *_value_usd / *_value_eth."""
    return [c for c in df.columns if c.endswith("_value_usd") or c.endswith("_value_eth")]


def _avg_base_cols(df):
    """User-account USD-base averages: avg_*_base."""
    return [c for c in df.columns if c.startswith("avg_") and c.endswith("_base")]


def _pair_le_num(df, a_col, b_col, cap=10):
    """a <= b over rows where both present -> (n_checked, n_fail, sample buckets)."""
    a, b = _col_numbers(df, a_col), _col_numbers(df, b_col)
    mask = [(av is not None and bv is not None and av > bv) for av, bv in zip(a, b)]
    n_checked = sum(1 for av, bv in zip(a, b) if av is not None and bv is not None)
    _, sample = _key_buckets(df, mask, cap=cap)
    return n_checked, sum(mask), sample


def _monotonic_violations_num(df, asset_col, time_col, value_col):
    """Per asset, value never decreases over time — decimals-aware (post RAY->decimal)."""
    sub = df[[asset_col, time_col, value_col]].copy()
    sub["_v"] = sub[value_col].map(_to_number)
    sub = sub.sort_values([asset_col, time_col], kind="mergesort")  # time str sorts chronologically
    n = bad = 0
    prev_a = prev_v = None
    for a, v in zip(sub[asset_col], sub["_v"]):
        if v is not None and a == prev_a and prev_v is not None:
            n += 1
            if v < prev_v:
                bad += 1
        prev_a, prev_v = a, v
    return n, bad


# --------------------------------------------------------------------------- #
# Consolidated scanners — one tidy row per logical check across many columns
# (failures surface via anomaly_columns + per-offender counts in detail, so the
# suite stays compact and a sub-100% pass rate is never buried in green noise).
# --------------------------------------------------------------------------- #
def _scan_values(df, cols, is_bad):
    """Across cols, over non-null parsed values, count those failing is_bad(v).
    Returns (n_checked, n_bad, {col: bad_count} for offending cols only)."""
    n = bad = 0
    off = {}
    for c in cols:
        cb = 0
        for v in _col_numbers(df, c):
            if v is None:
                continue
            n += 1
            if is_bad(v):
                bad += 1; cb += 1
        if cb:
            off[c] = cb
    return n, bad, off


def _scan_unparseable(df, cols):
    """Across cols, count PRESENT cells that do not parse as a number."""
    n = bad = 0
    off = {}
    for c in cols:
        cb = 0
        for v in df[c]:
            if pd.isna(v):
                continue
            n += 1
            if _to_number(v) is None:
                bad += 1; cb += 1
        if cb:
            off[c] = cb
    return n, bad, off


def _scan_nulls(df, cols):
    """Across cols, count null cells (a count/flow null = a quiet bucket left unfilled)."""
    n = null = 0
    off = {}
    for c in cols:
        cn = int(df[c].isna().sum())
        n += len(df); null += cn
        if cn:
            off[c] = cn
    return n, null, off


def _scan_pairs_le(df, pairs):
    """Each present (a, b): require a <= b. Returns (n_checked, n_bad, {label: bad})."""
    n = bad = 0
    off = {}
    for a, b in pairs:
        if a in df.columns and b in df.columns:
            nn, bd, _ = _pair_le_num(df, a, b)
            n += nn; bad += bd
            if bd:
                off[f"{a} <= {b}"] = bd
    return n, bad, off


def _offenders_str(off, unit=""):
    """Human-readable 'col=count; col2=count2' for the detail field."""
    return "; ".join(f"{k}={v}{unit}" for k, v in off.items())


def _consolidated(tier, check, columns, n_checked, n_bad, off,
                  severity="hard", detail=None):
    """One tidy row for a multi-column check: anomaly_columns = the offenders."""
    return _t_record(tier, check, columns, n_checked, n_bad, severity=severity,
                     anomaly_columns=";".join(off),
                     detail=detail if detail is not None else _offenders_str(off))


# --------------------------------------------------------------------------- #
# Tier 0 — Schema & type
# --------------------------------------------------------------------------- #
def tf_tier0_schema(df):
    rows = []
    if "time_bucket" in df.columns:                              # tz-explicit timestamp (UTC)
        ok = df["time_bucket"].astype("string").str.match(TIME_TZ_REGEX, na=False)
        _, sample = _key_buckets(df, (~ok).tolist())
        rows.append(_t_record("T0", "time_bucket is a tz-explicit UTC timestamp",
                              "time_bucket", len(df), int((~ok).sum()), anomaly_buckets=sample))

    counts = _count_like_cols(df)
    n, bad, off = _scan_values(df, counts, lambda v: float(v) != int(v))  # integer-valued
    rows.append(_consolidated("T0", "counts are integer-valued (no fractional aggregation)",
                              f"{len(counts)} count columns", n, bad, off))

    # HARD: a quiet bucket must be a 0, not a missing row left over from a left-join.
    # null count/flow cells are 'gaps' the transform never 0-filled (data_val.md Tier 4.5).
    n, nul, off = _scan_nulls(df, counts)
    rows.append(_consolidated("T0", "activity counts complete (quiet bucket = 0, not null)",
                              f"{len(counts)} count columns", n, nul, off,
                              detail="null cells per column (should be 0-filled): "
                                     + (_offenders_str(off) or "none")))

    vcols = (_value_cols(df) + _avg_base_cols(df)
             + [c for c in ("avg_ltv", "avg_current_liquidation_threshold") if c in df.columns])
    n, bad, off = _scan_unparseable(df, vcols)                  # numeric float, not junk string
    rows.append(_consolidated("T0", "value / avg columns parse as a numeric float",
                              f"{len(vcols)} value/avg columns", n, bad, off))
    return pd.DataFrame(rows, columns=TRANSFORM_RESULT_COLS)


# --------------------------------------------------------------------------- #
# Tier 1 — Univariate domain / range
# --------------------------------------------------------------------------- #
def tf_tier1_domain(df):
    rows = []
    nn = _count_like_cols(df) + _value_cols(df) + _avg_base_cols(df)
    n, bad, off = _scan_values(df, nn, lambda v: v < 0)         # non-negativity (one scan)
    rows.append(_consolidated("T1", "non-negative (>= 0): counts, *_value_*, avg_*_base",
                              f"{len(nn)} columns", n, bad, off))

    bps = [c for c in ("avg_ltv", "avg_current_liquidation_threshold") if c in df.columns]
    n, bad, off = _scan_values(df, bps, lambda v: v < 0 or v > BPS_MAX)   # numeric bound
    rows.append(_consolidated("T1", "bps numeric bound [0, 10000]",
                              "avg_ltv, avg_current_liquidation_threshold", n, bad, off))

    # HARD: bps fields must be on the bps SCALE. A populated value in (0, 1] is the silent
    # bps->fraction conversion data_val.md says to catch — it slips past the [0, 10000] bound
    # only because 0.79 < 10000. Count every fractional value as a unit violation.
    n, bad, off = _scan_values(df, bps, lambda v: 0 < v <= 1.0)
    rows.append(_consolidated("T1", "bps fields on bps scale (not silently rescaled to fraction)",
                              "avg_ltv, avg_current_liquidation_threshold", n, bad, off,
                              detail="fractional (0,1] values per column — spec expects bps 0-10000: "
                                     + (_offenders_str(off) or "none")))

    sv = _value_cols(df) + _avg_base_cols(df)
    n, bad, off = _scan_values(df, sv, lambda v: v in SENTINELS or abs(v) >= WEI_LOOKING_MIN)
    rows.append(_consolidated("T1", "no sentinel / un-scaled wei magnitude in value fields",
                              f"{len(sv)} value/base columns", n, bad, off,
                              detail=f"flags {SENTINELS} or |v|>={WEI_LOOKING_MIN:g}; offenders: "
                                     + (_offenders_str(off) or "none")))
    return pd.DataFrame(rows, columns=TRANSFORM_RESULT_COLS)


# --------------------------------------------------------------------------- #
# Tier 2 — Intra-row logical invariants
# --------------------------------------------------------------------------- #
def _zero_coupling(df, anchor, amount_members, count_members, cap=10):
    """count <=> amount <=> uniques per family.  Per row with the anchor present:
         anchor == 0  =>  every member == 0
         anchor  > 0  =>  every amount & unique member > 0
    All-null rows (anchor and members all null) are NOT counted — they are the
    'quiet bucket vs pipeline gap' case and are reported separately so a structural
    zero is never confused with a missing row.
    """
    members = amount_members + count_members
    a = _col_numbers(df, anchor)
    M = {m: _col_numbers(df, m) for m in members}
    n_checked = n_fail = n_null = n_zero = n_active = 0
    mask = [False] * len(df)
    for i in range(len(df)):
        av = a[i]
        mv = [M[m][i] for m in members]
        if av is None and all(v is None for v in mv):
            n_null += 1                                          # all-null: quiet or gap
            continue
        n_checked += 1                                           # any non-all-null row is checked
        if av is None:                                           # anchor null but members present
            n_fail += 1; mask[i] = True
            continue
        if av == 0:
            if all(v == 0 for v in mv):
                n_zero += 1
            else:
                n_fail += 1; mask[i] = True
        else:
            n_active += 1
            ok = (all(M[m][i] is not None and M[m][i] > 0 for m in amount_members)
                  and all(M[m][i] is not None and M[m][i] > 0 for m in count_members))
            if not ok:
                n_fail += 1; mask[i] = True
    _, sample = _key_buckets(df, mask, cap=cap)
    return n_checked, n_fail, sample, n_null, n_zero, n_active


def _sampling_consistency(df, cap=10):
    """sampled_user_count == 0  =>  the avg_* aggregates must be null (not a 0 from /0).
    One consolidated row; offenders = avg_* columns that held a value at a zero-sample row."""
    if "sampled_user_count" not in df.columns:
        return []
    s = _col_numbers(df, "sampled_user_count")
    zero_idx = [i for i, v in enumerate(s) if v is not None and v == 0]
    targets = [c for c in ("avg_total_collateral_base", "avg_total_debt_base",
                           "avg_available_borrows_base", "avg_ltv",
                           "avg_current_liquidation_threshold") if c in df.columns]
    mask = [False] * len(df)
    off = {}
    for c in targets:
        cn = _col_numbers(df, c)
        cb = 0
        for i in zero_idx:
            if cn[i] is not None:                                # present where it should be null
                mask[i] = True; cb += 1
        if cb:
            off[c] = cb
    _, sample = _key_buckets(df, mask, cap=cap)
    note = f"rows with sampled_user_count==0: {len(zero_idx)}"
    if not zero_idx:
        note += " (invariant not exercised on this data)"
    return [_t_record("T2", "sampled_user_count==0 => avgs are null (no /0 masquerading as 0)",
                      f"{len(targets)} avg_* columns", len(zero_idx) * len(targets),
                      sum(off.values()), anomaly_columns=";".join(off),
                      anomaly_buckets=sample, detail=note)]


def tf_tier2_invariants(df):
    rows = []
    uniq_pairs = [("unique_suppliers", "supply_tx_count"),
                  ("unique_withdraw_users", "withdrawal_tx_count"),
                  ("unique_borrowers", "borrow_tx_count"),
                  ("unique_repayers", "repay_tx_count"),
                  ("unique_liquidated_users", "liquidation_tx_count"),
                  ("unique_liquidators", "liquidation_tx_count"),
                  ("unique_flashloan_initiators", "flashloan_tx_count"),
                  ("unique_collateral_enable_users", "collateral_enabled_count"),
                  ("unique_collateral_disable_users", "collateral_disabled_count")]
    n, bad, off = _scan_pairs_le(df, uniq_pairs)               # uniques <= tx (all families)
    rows.append(_consolidated("T2", "unique users <= tx count (all families)",
                              "9 unique/tx family pairs", n, bad, off))
    subset_pairs = [("variable_borrow_tx_count", "borrow_tx_count"),
                    ("variable_flashloan_tx_count", "flashloan_tx_count"),
                    ("no_open_debt_flashloan_tx_count", "flashloan_tx_count")]
    n, bad, off = _scan_pairs_le(df, subset_pairs)             # mode partitions <= total
    rows.append(_consolidated("T2", "subset count <= total (mode partitions)",
                              "3 subset pairs", n, bad, off))
    families = [
        ("supply", "supply_tx_count",
         ["supply_amount_value_usd", "supply_amount_value_eth"], ["unique_suppliers"]),
        ("withdrawal", "withdrawal_tx_count",
         ["withdrawal_amount_value_usd", "withdrawal_amount_value_eth"], ["unique_withdraw_users"]),
        ("borrow", "borrow_tx_count",
         ["borrow_amount_value_usd", "borrow_amount_value_eth"], ["unique_borrowers"]),
        ("repay", "repay_tx_count",
         ["repay_amount_value_usd", "repay_amount_value_eth"], ["unique_repayers"]),
        ("liquidation", "liquidation_tx_count",
         ["liquidated_collateral_value_usd", "liquidated_collateral_value_eth",
          "liquidation_debt_covered_value_usd", "liquidation_debt_covered_value_eth"],
         ["unique_liquidated_users", "unique_liquidators"]),
        ("flashloan", "flashloan_tx_count",
         ["flashloan_amount_value_usd", "flashloan_amount_value_eth"],
         ["unique_flashloan_initiators"]),
    ]
    for name, anchor, amts, cnts in families:
        if anchor in df.columns and all(c in df.columns for c in amts + cnts):
            n, bad, sample, n_null, n_zero, n_active = _zero_coupling(df, anchor, amts, cnts)
            rows.append(_t_record("T2", f"{name}: zero-coupling (count<=>amount<=>uniques)",
                                  anchor, n, bad,
                                  anomaly_columns=(",".join([anchor] + amts + cnts) if bad else ""),
                                  anomaly_buckets=sample,
                                  detail=f"all_null(quiet/gap)={n_null} structural_zero={n_zero} "
                                         f"active={n_active}"))
    if {"liquidation_debt_covered_value_usd", "liquidated_collateral_value_usd"} <= set(df.columns):
        n, bad, sample = _pair_le_num(df, "liquidation_debt_covered_value_usd",
                                      "liquidated_collateral_value_usd")
        rows.append(_t_record("T2", "liquidated collateral >= debt covered (USD; bonus)",
                              "debt_covered <= collateral (usd)", n, bad, severity="soft",
                              anomaly_buckets=sample))
    if {"avg_ltv", "avg_current_liquidation_threshold"} <= set(df.columns):
        n, bad, sample = _pair_le_num(df, "avg_ltv", "avg_current_liquidation_threshold")
        rows.append(_t_record("T2", "avg_ltv <= avg_current_liquidation_threshold",
                              "avg_ltv <= avg_current_liquidation_threshold", n, bad,
                              anomaly_columns=("avg_ltv,avg_current_liquidation_threshold" if bad else ""),
                              anomaly_buckets=sample))
    if {"avg_total_debt_base", "avg_total_collateral_base"} <= set(df.columns):
        n, bad, sample = _pair_le_num(df, "avg_total_debt_base", "avg_total_collateral_base")
        rows.append(_t_record("T2", "avg_total_debt_base <= avg_total_collateral_base (overcollat)",
                              "debt_base <= collateral_base", n, bad, severity="soft",
                              anomaly_buckets=sample))
    if "sampled_user_count" in df.columns:                     # SOFT: avg_* reliability
        n, bad, off = _scan_values(df, ["sampled_user_count"], lambda v: v < 2)
        rows.append(_t_record("T2", "avg_* backed by >= 2 sampled users (not a single-user proxy)",
                              "sampled_user_count", n, bad, severity="soft",
                              anomaly_columns=";".join(off),
                              detail="buckets where sampled_user_count==1 make every avg_* a single "
                                     "user's value, not a market average — unreliable as a feature"))
    rows.extend(_sampling_consistency(df))
    return pd.DataFrame(rows, columns=TRANSFORM_RESULT_COLS)


# --------------------------------------------------------------------------- #
# Tier 3 — Unit & cross-asset consistency
# --------------------------------------------------------------------------- #
def _usd_eth_pairing(df, usd, eth, cap=10):
    """usd == 0 (or null) <=> eth == 0 (or null), over rows where either side is present."""
    u, e = _col_numbers(df, usd), _col_numbers(df, eth)
    mask, n_checked = [], 0
    for uv, ev in zip(u, e):
        if uv is None and ev is None:
            mask.append(False); continue
        n_checked += 1
        mask.append(((uv is None or uv == 0)) != ((ev is None or ev == 0)))
    _, sample = _key_buckets(df, mask, cap=cap)
    return n_checked, sum(mask), sample


def _implied_price_coherence(df, families, tol=PRICE_SPREAD_TOL, cap=10):
    """Per bucket, usd/eth across families should imply ~one ETH price (spread <= tol)."""
    cols = {f: (_col_numbers(df, f[0]), _col_numbers(df, f[1])) for f in families}
    mask = [False] * len(df)
    spreads = []
    for i in range(len(df)):
        prices = []
        for f in families:
            uv, ev = cols[f][0][i], cols[f][1][i]
            if uv is not None and ev is not None and uv > 0 and ev > 0:
                prices.append(uv / ev)
        if len(prices) >= 2:
            med = sorted(prices)[len(prices) // 2]
            spread = (max(prices) - min(prices)) / med if med else 0.0
            spreads.append(spread)
            if spread > tol:
                mask[i] = True
    n_checked = len(spreads)
    _, sample = _key_buckets(df, mask, cap=cap)
    ss = sorted(spreads)
    detail = (f"median_spread={ss[len(ss)//2]:.4f} p95={ss[int(0.95*(len(ss)-1))]:.4f} "
              f"max={max(spreads):.4f}" if spreads else "no comparable buckets")
    return n_checked, sum(mask), sample, detail


def _magnitude_stability(df, cols, dex=MAGNITUDE_JUMP_DEX, cap=10):
    """avg_*_base base-unit stability. A decimals / base-unit change is a *persistent*
    mid-series level shift, not bucket-to-bucket noise — and these averages legitimately
    swing orders of magnitude on their own because the per-bucket sample is tiny. So this is
    a series-level check: compare the median log10 of the first vs second half of the
    time-ordered series and flag only a sustained shift of >= dex orders (one soft check per
    column). The old per-bucket jump count fired on ordinary sampling variation (~50% FPs)."""
    import statistics
    recs = []
    for c in cols:
        logs = [math.log10(x) for x in _col_numbers(df, c) if x is not None and x > 0]
        if len(logs) < 4:                                       # too few points to judge a break
            recs.append(_t_record("T3", "avg_*_base magnitude stable (no 10^x base-unit break)",
                                  c, 0, 0, severity="soft", detail="insufficient positive values"))
            continue
        half = len(logs) // 2
        m1, m2 = statistics.median(logs[:half]), statistics.median(logs[half:])
        broke = abs(m2 - m1) >= dex
        recs.append(_t_record("T3", "avg_*_base magnitude stable (no 10^x base-unit break)",
                              c, 1, 1 if broke else 0, severity="soft",
                              anomaly_columns=(c if broke else ""),
                              detail=f"median log10 1st-half={m1:.2f} 2nd-half={m2:.2f} "
                                     f"(Δ={m2 - m1:+.2f} dex); range=[{min(logs):.2f}, {max(logs):.2f}]"))
    return recs


def tf_tier3_consistency(df):
    rows = []
    pairs = [("supply_amount_value_usd", "supply_amount_value_eth"),
             ("withdrawal_amount_value_usd", "withdrawal_amount_value_eth"),
             ("borrow_amount_value_usd", "borrow_amount_value_eth"),
             ("repay_amount_value_usd", "repay_amount_value_eth"),
             ("liquidated_collateral_value_usd", "liquidated_collateral_value_eth"),
             ("liquidation_debt_covered_value_usd", "liquidation_debt_covered_value_eth"),
             ("flashloan_amount_value_usd", "flashloan_amount_value_eth"),
             ("flashloan_premium_value_usd", "flashloan_premium_value_eth")]
    n = bad = 0
    off = {}
    for u, e in pairs:                                         # USD<=>ETH nullity (one scan)
        if u in df.columns and e in df.columns:
            nn, bd, _ = _usd_eth_pairing(df, u, e)
            n += nn; bad += bd
            if bd:
                off[f"{u}~{e}"] = bd
    rows.append(_consolidated("T3", "USD<=>ETH paired nullity (0/empty together)",
                              f"{len(pairs)} value pairs", n, bad, off))
    fams = [(u, e) for u, e in pairs[:7]                       # amount pairs (skip premium)
            if u in df.columns and e in df.columns and "premium" not in u]
    if len(fams) >= 2:
        n, bad, sample, detail = _implied_price_coherence(df, fams)
        rows.append(_t_record("T3", "cross-family implied ETH price coherent (<= 5% spread)",
                              "usd/eth across families", n, bad, severity="soft",
                              anomaly_buckets=sample, detail=detail))
    rows.extend(_magnitude_stability(df, _avg_base_cols(df)))
    return pd.DataFrame(rows, columns=TRANSFORM_RESULT_COLS)


# --------------------------------------------------------------------------- #
# Tier 4 — Temporal integrity (time_bucket as the protocol-level primary key)
# --------------------------------------------------------------------------- #
def _full_6h_grid(start, end):
    """Every 6h boundary from start to end inclusive (as pandas Timestamps)."""
    from datetime import timedelta
    step, t, grid = timedelta(hours=1), start, []
    while t <= end:
        grid.append(t); t += step
    return grid


def tf_tier4_temporal(df):
    rows = []
    if "time_bucket" not in df.columns:
        return pd.DataFrame(rows, columns=TRANSFORM_RESULT_COLS)
    s = df["time_bucket"].astype("string")
    parsed = pd.to_datetime(s.str.replace(" UTC", "", regex=False), errors="coerce", utc=True)

    dup = df["time_bucket"].duplicated(keep=False).tolist()    # PK uniqueness
    _, dup_sample = _key_buckets(df, dup)
    rows.append(_t_record("T4", "time_bucket unique (no duplicate PK)", "time_bucket",
                          len(df), sum(dup), anomaly_buckets=dup_sample,
                          detail=f"distinct={df['time_bucket'].nunique()} of {len(df)} rows"))

    grid_bad = [bool(pd.isna(p)) or (p.minute or p.second)
                for p in parsed]                                # 6h grid alignment
    _, g_sample = _key_buckets(df, grid_bad)
    rows.append(_t_record("T4", "time_bucket lands on a 6h boundary (00/06/12/18 UTC)",
                          "time_bucket", len(df), sum(grid_bad), anomaly_buckets=g_sample))

    valid = sorted(p for p in parsed if not pd.isna(p))
    if valid:
        grid = _full_6h_grid(valid[0], valid[-1])               # gaps / coverage
        present = set(valid)
        missing = [g for g in grid if g not in present]
        rows.append(_t_record("T4", "no gaps: full 6h coverage over min..max range",
                              "time_bucket", len(grid), len(missing),
                              anomaly_buckets=[g.strftime("%Y-%m-%d %H:%M UTC") for g in missing[:10]],
                              detail=f"expected={len(grid)} present={len(present)} "
                                     f"missing={len(missing)} "
                                     f"range=[{valid[0]:%Y-%m-%d %H:%M} .. {valid[-1]:%Y-%m-%d %H:%M}]"))
        pv = parsed.tolist()                                    # strictly increasing in row order
        n_tr = sum(1 for i in range(len(pv) - 1)
                   if not pd.isna(pv[i]) and not pd.isna(pv[i + 1]))
        n_bad = sum(1 for i in range(len(pv) - 1)
                    if not pd.isna(pv[i]) and not pd.isna(pv[i + 1]) and pv[i + 1] <= pv[i])
        rows.append(_t_record("T4", "time_bucket strictly increasing in row order",
                              "time_bucket", n_tr, n_bad))
    return pd.DataFrame(rows, columns=TRANSFORM_RESULT_COLS)


# --------------------------------------------------------------------------- #
# Entry points — run all tiers and (optionally) save one tidy CSV per frame
# --------------------------------------------------------------------------- #
def validate_transformed_final(df, table_name="DF_common_final",
                               results_dir="validation_results", save=True):
    """Run Tiers 0-4 on the protocol-level feature matrix; return one tidy DataFrame."""
    out = pd.concat([tf_tier0_schema(df), tf_tier1_domain(df), tf_tier2_invariants(df),
                     tf_tier3_consistency(df), tf_tier4_temporal(df)], ignore_index=True)
    if save:
        from pathlib import Path
        folder = Path(results_dir); folder.mkdir(parents=True, exist_ok=True)
        out.to_csv(folder / f"{table_name}__transform_validation.csv", index=False)
    return out


def validate_reserve_panel(df, table_name="DF_common_1",
                           results_dir="validation_results", save=True):
    """Applicable subset of data_val.md for the asset-level reserve panel
    (composite-PK uniqueness, time schema/grid, rate/index non-negativity, index>=1.0
    and per-asset index monotonicity). Returns one tidy DataFrame."""
    rows = []
    if {"time_bucket", "asset"} <= set(df.columns):
        dup = df.duplicated(subset=["time_bucket", "asset"], keep=False).tolist()
        _, sample = _key_buckets(df, dup, key="asset")
        n_keys = df.drop_duplicates(["time_bucket", "asset"]).shape[0]
        rows.append(_t_record("T0/T4", "composite PK (time_bucket, asset) unique",
                              "time_bucket,asset", len(df), sum(dup), anomaly_buckets=sample,
                              detail=f"distinct keys={n_keys} of {len(df)} rows"))
    if "time_bucket" in df.columns:
        s = df["time_bucket"].astype("string")
        ok = s.str.match(TIME_TZ_REGEX, na=False)
        rows.append(_t_record("T0", "time_bucket is a tz-explicit UTC timestamp",
                              "time_bucket", len(df), int((~ok).sum())))
        grid = s.str.contains(GRID_6H_REGEX, regex=True, na=False)
        rows.append(_t_record("T4", "time_bucket lands on a 6h boundary",
                              "time_bucket", len(df), int((~grid).sum())))
    for c in ("last_borrow_rate", "liquidity_rate", "variable_borrow_rate"):
        if c in df.columns:
            r = negative_value_check(df, c, plot=False)
            rows.append(_t_record("T1", "rate >= 0 (RAY->decimal)", c,
                                  r["n_checked"], r["n_negative"],
                                  anomaly_columns=(c if r["n_negative"] else ""),
                                  detail=f"observed_min={r['min_value']}"))
    for c in ("liquidity_index", "variable_borrow_index"):
        if c in df.columns:
            r = range_check(df, c, min_value=1.0, plot=False)
            rows.append(_t_record("T1", "index >= 1.0 (RAY->decimal)", c,
                                  r["n_checked"], r["n_below_min"],
                                  anomaly_columns=(c if r["n_below_min"] else ""),
                                  detail=f"observed_min={r['observed_min']}"))
            if {"asset", "time_bucket"} <= set(df.columns):
                n, bad = _monotonic_violations_num(df, "asset", "time_bucket", c)
                rows.append(_t_record("T2", "index non-decreasing per asset over time", c,
                                      n, bad, anomaly_columns=(c if bad else "")))
    out = pd.DataFrame(rows, columns=TRANSFORM_RESULT_COLS)
    if save:
        from pathlib import Path
        folder = Path(results_dir); folder.mkdir(parents=True, exist_ok=True)
        out.to_csv(folder / f"{table_name}__transform_validation.csv", index=False)
    return out


if __name__ == "__main__":
    # Convenience smoke runner:  python adv_validation.py [csv ...]
    # (the notebook calls validate_table_advanced() directly — not written here)
    from pathlib import Path

    paths = sys.argv[1:] or sorted(str(p) for p in Path("query_result_data").glob("*.csv"))
    for path in paths:
        frame = load_csv(path)
        name = detect_table(frame) or table_name_from_path(path)
        if name not in TABLE_SUITES:
            print(f"- skip {name:20s} (no advanced suite) :: {Path(path).name}")
            continue
        res = validate_table_advanced(frame, name)
        n_fail = int((~res["success"]).sum())
        flag = "OK " if n_fail == 0 else f"{n_fail} FAIL"
        print(f"- {name:20s} {len(res):2d} checks  {flag}")
