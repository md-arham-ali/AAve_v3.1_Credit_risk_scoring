"""Normalization stage for the Aave V3.1 raw extracts.

The ``optimized_queries`` now emit RAW values (no in-SQL division), so this module
is the SINGLE place where raw columns are scaled to real units. The pipeline is:

    optimized_queries (raw)  ->  API fetch  ->  normalize.ipynb (this module)  ->  transform.ipynb

so ``transform.ipynb`` always consumes already-normalized frames (its own per-table
normalization is removed). Every function returns a NEW DataFrame (input untouched).

Normalization rules (raw -> real units):
  * reserve_config : per-column ``/10**decimals`` from the decimal reference
                     (caps are decimals 0 -> unchanged; debt_ceiling /1e2; bps /1e4).
  * reserve_state  : RAY rates / indexes ``/1e27``.
  * flashloan / borrow amounts : ``/10**token_decimals`` per asset.

Libraries imported:
    pandas          -> dataframes
    decimal.Decimal -> exact big-integer division before the float cast
"""

from decimal import Decimal

import pandas as pd

RAY_DECIMALS = 27  # RAY-encoded rates / indexes are scaled by 10**27


def _scale(value, decimals):
    """Divide one value by ``10**decimals`` exactly, or NaN if either is missing."""
    if pd.isna(value) or pd.isna(decimals):
        return float("nan")
    return float(Decimal(str(value)) / (Decimal(10) ** int(decimals)))


def decimals_from_reference(reference, metric_col="metric", decimals_col="decimals"):
    """Build a ``{column_name: decimals}`` map from a ``decimal_reference_partN`` frame."""
    ref = reference.dropna(subset=[metric_col, decimals_col]).drop_duplicates(metric_col)
    return dict(zip(ref[metric_col], ref[decimals_col].astype(int)))


def normalize_columns_by_decimals(df, decimals_map, columns):
    """Divide each named column by ``10**decimals`` (a fixed decimals PER column)."""
    out = df.copy()
    for col in columns:
        if col in out.columns and col in decimals_map:
            dec = decimals_map[col]
            out[col] = [_scale(v, dec) for v in df[col]]
    return out


# --------------------------------------------------------------------------- #
# Table-specific wrappers
# --------------------------------------------------------------------------- #
# reserve_config: only these carry a non-zero scale. supply_cap / borrow_cap are
# whole tokens (decimals 0) so they stay raw; the reference still maps them, but
# dividing by 10**0 is a no-op, so listing them is harmless.
RESERVE_CONFIG_COLS = [
    "debt_ceiling",            # usd, 2dp   -> /1e2
    "reserve_factor",          # basis pts  -> /1e4
    "liquidation_threshold",   # basis pts  -> /1e4
    "liquidation_bonus",       # basis pts  -> /1e4
    "ltv",                     # basis pts  -> /1e4
]

RESERVE_STATE_COLS = [
    "liquidity_rate", "variable_borrow_rate", "stable_borrow_rate",
    "liquidity_index", "variable_borrow_index",
]


def normalize_reserve_config(df, decimals_reference, columns=RESERVE_CONFIG_COLS):
    """Scale reserve_config raw bps / 2dp columns to real units via the decimal reference."""
    dmap = decimals_from_reference(decimals_reference)
    return normalize_columns_by_decimals(df, dmap, columns)


def normalize_reserve_state(df, columns=RESERVE_STATE_COLS):
    """Scale RAY-encoded (1e27) rates / indexes to plain decimals."""
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = [_scale(v, RAY_DECIMALS) for v in df[col]]
    return out


def normalize_amounts_by_token_decimals(df, token_decimals, columns, asset_col="asset"):
    """Divide amount columns by ``10**token_decimals`` per asset (flashloan / borrow)."""
    out = df.copy()
    decs = df[asset_col].map(token_decimals)
    for col in columns:
        if col in out.columns:
            out[col] = [_scale(v, d) for v, d in zip(df[col], decs)]
    return out


# --------------------------------------------------------------------------- #
# Already-normalized guard (so re-running on already-scaled data is a no-op)
# --------------------------------------------------------------------------- #
def looks_raw_bps(series, raw_floor=100):
    """True if a bps column still looks raw (values >> 1), i.e. not yet /1e4 scaled.

    Raw ltv / liquidation_threshold are basis points (e.g. 6700); once normalized
    they are fractions (e.g. 0.67). A max above ``raw_floor`` means still raw.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    return bool(len(s)) and s.max() > raw_floor


# Raw token amounts are integers on the token's own decimals scale (18-decimal
# assets routinely exceed 1e18); a normalized amount is a real token count, and no
# Aave reserve moves 1e15 tokens in one bucket. Same floor adv_validation uses for
# its "un-scaled wei magnitude" check, so the two stages agree on what 'raw' means.
WEI_LOOKING_MIN = 1e15


def looks_raw_token_amounts(df, columns, raw_floor=WEI_LOOKING_MIN):
    """True if any amount column still looks raw (un-scaled by token decimals).

    Guard for :func:`normalize_amounts_by_token_decimals` so re-running on an
    already-normalized extract is a no-op. Missing columns are ignored.
    """
    for col in columns:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna().abs()
        if len(s) and s.max() >= raw_floor:
            return True
    return False


def looks_raw_ray(series, raw_floor=1e3):
    """True if a RAY rate / index column is still raw (not yet /1e27 scaled).

    Normalized rates are fractions and indexes sit just above 1.0, so anything
    above ``raw_floor`` is still RAY-encoded.
    """
    s = pd.to_numeric(series, errors="coerce").dropna().abs()
    return bool(len(s)) and s.max() > raw_floor
