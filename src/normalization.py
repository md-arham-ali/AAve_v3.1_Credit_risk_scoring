"""Normalization stage — the single place raw Aave extracts are scaled to real units.

Rules: reserve_config /10**decimals, reserve_state RAY /1e27, amounts /10**token_decimals.
Every function returns a NEW DataFrame; transform.ipynb assumes it ran first.
"""

from decimal import Decimal

import pandas as pd

RAY_DECIMALS = 27  # RAY-encoded rates / indexes are scaled by 10**27


def _scale(value, decimals):
    """Divide one value by ``10**decimals`` exactly, or NaN if either is missing."""
    # Decimal division keeps the big integer exact until the final float cast
    if pd.isna(value) or pd.isna(decimals):
        return float("nan")
    return float(Decimal(str(value)) / (Decimal(10) ** int(decimals)))


def decimals_from_reference(reference, metric_col="metric", decimals_col="decimals"):
    """Build a ``{column_name: decimals}`` map from a ``decimal_reference_partN`` frame."""
    # only raw_token_amount rows — block-number rows are decimals 0 and would override
    ref = reference.dropna(subset=[metric_col, decimals_col]).drop_duplicates(metric_col)
    return dict(zip(ref[metric_col], ref[decimals_col].astype(int)))


def normalize_columns_by_decimals(df, decimals_map, columns):
    """Divide each named column by ``10**decimals`` (a fixed decimals PER column)."""
    # returns a new frame; nothing here mutates the input
    out = df.copy()
    for col in columns:
        if col in out.columns and col in decimals_map:
            dec = decimals_map[col]
            out[col] = [_scale(v, dec) for v in df[col]]
    return out


# --- table-specific wrappers ---
# supply_cap / borrow_cap are decimals 0, so their /10**0 is a harmless no-op.
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
    # guard with looks_raw_bps first if you might be re-running
    dmap = decimals_from_reference(decimals_reference)
    return normalize_columns_by_decimals(df, dmap, columns)


def normalize_reserve_state(df, columns=RESERVE_STATE_COLS):
    """Scale RAY-encoded (1e27) rates / indexes to plain decimals."""
    # RAY /1e27: rates become fractions, indexes sit just above 1.0
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = [_scale(v, RAY_DECIMALS) for v in df[col]]
    return out


def normalize_amounts_by_token_decimals(df, token_decimals, columns, asset_col="asset"):
    """Divide amount columns by ``10**token_decimals`` per asset (flashloan / borrow)."""
    # per-asset decimals, so an 18-decimal asset and USDC scale differently
    out = df.copy()
    decs = df[asset_col].map(token_decimals)
    for col in columns:
        if col in out.columns:
            out[col] = [_scale(v, d) for v, d in zip(df[col], decs)]
    return out


# --- already-normalized guards, so a re-run is a no-op ---
def looks_raw_bps(series, raw_floor=100):
    """True if a bps column still looks raw (6700 rather than 0.67).
    df, columns, raw_floor -> max-value check.
    Returns: bool."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    return bool(len(s)) and s.max() > raw_floor


    # Same floor adv_validation uses, so both stages agree on what 'raw' means.
WEI_LOOKING_MIN = 1e15


def looks_raw_token_amounts(df, columns, raw_floor=WEI_LOOKING_MIN):
    """True if any amount column is still un-scaled by token decimals.
    df, columns, raw_floor -> max-value check; missing columns ignored.
    Returns: bool."""
    for col in columns:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna().abs()
        if len(s) and s.max() >= raw_floor:
            return True
    return False


def looks_raw_ray(series, raw_floor=1e3):
    """True if a RAY rate / index column is still raw (not yet /1e27).
    df, columns, raw_floor -> max-value check.
    Returns: bool."""
    s = pd.to_numeric(series, errors="coerce").dropna().abs()
    return bool(len(s)) and s.max() > raw_floor
