"""Transformation helpers for the Aave V3.1 Dune result tables.

READS the versioned CSVs in query_result_data/ and never writes back there. Each
fetch adds a new timestamped file, so the loaders resolve the latest per table.
"""

import re
from decimal import Decimal
from pathlib import Path

import pandas as pd

# Make sibling src/ modules importable however this file is loaded (notebook, script, or import).
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_validation import TABLE_LABELS, table_name_from_path

# Folder holding the raw, versioned extracts. Read-only from this stage.
SOURCE_DIR = "query_result_data"

# Matches the versioned suffix _<query_id>_<YYYYMMDDTHHMMSSZ> at the end of the stem —
# works for both query_result_data_<id>_<ts>.csv and <table_name>_<id>_<ts>.csv.
_NAME_RE = re.compile(r"_(\d+)_(\d{8}T\d{6}Z)$")


def _parse_name(path):
    """Return (query_id, timestamp) from a versioned CSV path, or (None, None)."""
    match = _NAME_RE.search(Path(path).stem)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def latest_paths(source_dir=SOURCE_DIR):
    """Map each table label -> its newest versioned CSV path.
    data_dir -> filenames parsed for query id + fetch timestamp.
    Returns: {label: Path}. The timestamp suffix sorts lexicographically, so max = newest."""
    newest = {}  # query_id -> (timestamp, path)
    for path in Path(source_dir).glob("*.csv"):
        qid, stamp = _parse_name(path)
        if qid is None:
            continue
        if qid not in newest or stamp > newest[qid][0]:
            newest[qid] = (stamp, path)
    return {
        TABLE_LABELS.get(qid, f"query_{qid}"): path
        for qid, (_, path) in newest.items()
    }


def list_tables(source_dir=SOURCE_DIR):
    """Return a {label: filename} view of the latest version of every table."""
    return {label: path.name for label, path in latest_paths(source_dir).items()}


def load_table(table, source_dir=SOURCE_DIR):
    """Load one result table as a DataFrame (read-only).
    table: a label whose latest version is resolved, or a direct CSV path.
    Returns: DataFrame as-loaded — time_bucket / asset stay strings."""
    path = Path(table)
    if path.suffix == ".csv" and path.exists():
        return pd.read_csv(path)

    paths = latest_paths(source_dir)
    if table not in paths:
        raise KeyError(
            f"Unknown table '{table}'. Available labels: {sorted(paths)}"
        )
    return pd.read_csv(paths[table])


def load_all(source_dir=SOURCE_DIR):
    """Load the latest version of every table into {label: DataFrame} (read-only)."""
    return {
        label: pd.read_csv(path)
        for label, path in latest_paths(source_dir).items()
    }


# --- per-ASSET scaling: raw integer amounts -> real token units ---
# Division goes through Decimal so the big integers stay exact before the float cast.
RAW_SUFFIX = "_raw"


def raw_amount_columns(df, suffix=RAW_SUFFIX):
    """Return the columns holding raw integer amounts (those ending in ``_raw``)."""
    return [c for c in df.columns if c.endswith(suffix)]


def decimals_map(decimals, asset_col="asset", decimals_col="decimals", unit_col="unit",
                 token_unit="raw_token_amount"):
    """Normalize a `decimals` argument into an {asset_address: int} lookup.
    Accepts int (uniform), dict/Series (used directly), or DataFrame (asset + decimals cols).
    Returns: the lookup. Reference tables use only raw_token_amount rows; dupes keep the first."""
    if isinstance(decimals, int):
        return decimals
    if isinstance(decimals, pd.Series):
        return decimals.dropna().astype(int).to_dict()
    if isinstance(decimals, pd.DataFrame):
        ref = decimals
        if unit_col in ref.columns:
            ref = ref[ref[unit_col] == token_unit]
        ref = ref.dropna(subset=[asset_col, decimals_col]).drop_duplicates(asset_col)
        return dict(zip(ref[asset_col], ref[decimals_col].astype(int)))
    return dict(decimals)


def _scaled_name(col, suffix):
    """New column name for a scaled raw column: strip ``_raw``, else append suffix."""
    if col.endswith(RAW_SUFFIX):
        return col[: -len(RAW_SUFFIX)]
    return f"{col}{suffix}"


def _scale_value(value, dec):
    """Divide one value by 10**dec exactly (Decimal -> float).
    value, dec.
    Returns: float, or NaN when either side is missing."""
    if pd.isna(value) or pd.isna(dec):
        return float("nan")
    return float(Decimal(str(value)) / (Decimal(10) ** int(dec)))


def scale_by_decimals(df, decimals, columns=None, asset_col="asset",
                      decimals_col="decimals", drop_raw=False, suffix="_scaled"):
    """Divide raw integer-amount columns by 10**decimals -> real token units.
    decimals: int/dict/Series/DataFrame (see decimals_map); columns defaults to every *_raw;
    asset_col, decimals_col, drop_raw, suffix. Returns: a NEW DataFrame."""

    # NOTE: *_raw columns lose the suffix rather than gain one (supply_amount_raw -> supply_amount)
    cols = columns if columns is not None else raw_amount_columns(df)
    if not cols:
        return df.copy()

    dmap = decimals_map(decimals, asset_col=asset_col, decimals_col=decimals_col)
    out = df.copy()

    if isinstance(dmap, int):
        dec_per_row = pd.Series(dmap, index=df.index)
    else:
        dec_per_row = df[asset_col].map(dmap)

    decs = list(dec_per_row)
    for col in cols:
        out[_scaled_name(col, suffix)] = [
            _scale_value(raw, dec) for raw, dec in zip(df[col], decs)
        ]

    if drop_raw:
        out = out.drop(columns=[c for c in cols if c in out.columns])
    return out


# --- per-COLUMN scaling: fixed decimals from a metric map ---
# Companion to scale_by_decimals. Aave config metrics carry one decimals for the whole
# column (caps -> 0, debt_ceiling -> 2, bps fields -> 4), keyed by `metric` in the reference.

def column_decimals_map(decimals, metric_col="metric", decimals_col="decimals"):
    """Normalize a per-column decimals argument into a {column_name: int} lookup.
    Accepts dict/Series (keys are column names) or DataFrame (metric + decimals cols).
    Returns: the lookup; null rows dropped, duplicate metrics keep the first."""
    if isinstance(decimals, pd.Series):
        return decimals.dropna().astype(int).to_dict()
    if isinstance(decimals, pd.DataFrame):
        ref = decimals.dropna(subset=[metric_col, decimals_col]).drop_duplicates(metric_col)
        return dict(zip(ref[metric_col], ref[decimals_col].astype(int)))
    return dict(decimals)


def scale_columns_by_decimals(df, decimals, columns=None, metric_col="metric",
                              decimals_col="decimals", overwrite=False,
                              drop_original=False, suffix="_scaled"):
    """Divide whole columns by 10**decimals, with a FIXED decimals PER COLUMN.
    decimals (see column_decimals_map), columns, overwrite, drop_original, suffix.
    Returns: a NEW DataFrame; NaN wherever value or decimals is missing."""
    dmap = column_decimals_map(decimals, metric_col=metric_col, decimals_col=decimals_col)
    cols = [c for c in (columns if columns is not None else df.columns)
            if c in dmap and c in df.columns]
    if not cols:
        return df.copy()

    out = df.copy()
    for col in cols:
        dec = dmap[col]
        out[col if overwrite else f"{col}{suffix}"] = [
            _scale_value(v, dec) for v in df[col]
        ]

    if drop_original and not overwrite:
        out = out.drop(columns=[c for c in cols if c in out.columns])
    return out





# --- price multiplication: value each amount in USD and in ETH ---
# Matched on (time_bucket, asset); the two frames store time_bucket in different string
# formats, so both are normalized to a UTC datetime key first. Missing price or amount
# yields NaN, never a silent 0.

def multiply_by_price(df, prices, amount_columns, time_col="time_bucket",
                      asset_col="asset", usd_col="avg_price_usd",
                      eth_col="avg_price_eth", usd_suffix="_value_usd",
                      eth_suffix="_value_eth"):
    """Value each amount column in USD and ETH against an oracle-price table.
    df, prices, amount_columns, time_col, asset_col, usd_suffix, eth_suffix.
    Returns: a NEW DataFrame with two value columns per amount; NaN when either side is absent."""
    out = df.copy()

    # normalized match key on both sides (the tables differ in time_bucket format)
    left_key = pd.to_datetime(df[time_col], utc=True, format="mixed")
    right_key = pd.to_datetime(prices[time_col], utc=True, format="mixed")

    # one price row per (time, asset): usd + eth, null prices stay NaN
    price = pd.DataFrame({
        "_t": right_key,
        "_a": prices[asset_col],
        "_usd": pd.to_numeric(prices[usd_col], errors="coerce"),
        "_eth": pd.to_numeric(prices[eth_col], errors="coerce"),
    }).drop_duplicates(["_t", "_a"])

    # left-join keeps df's row order; an unmatched (time, asset) -> NaN price
    merged = pd.DataFrame({"_t": left_key, "_a": df[asset_col]}).merge(
        price, on=["_t", "_a"], how="left")
    usd = merged["_usd"].to_numpy()
    eth = merged["_eth"].to_numpy()

    for col in amount_columns:
        amount = pd.to_numeric(df[col], errors="coerce").to_numpy()
        out[f"{col}{usd_suffix}"] = amount * usd
        out[f"{col}{eth_suffix}"] = amount * eth
    return out

# --- null repair after the protocol-panel merge (left joins on time_bucket) ---
# Two kinds of null come out of that merge: EVENT columns, where an absent bucket truly
# means zero events (left null they read >70% missing and poison every ratio), and STATE
# columns sampled from getUserAccountData, which land in ~half the buckets and must be
# carried forward under a staleness cap.

def fill_event_zeros(df, columns):
    """Fill nulls with 0 in EVENT columns (structural zeros from left joins).
    df, columns -> counts, unique-user counts, valued sums only.
    Returns: a NEW DataFrame. Never use on state or ratio columns."""
    out = df.copy()
    cols = [c for c in columns if c in out.columns]
    out[cols] = out[cols].fillna(0)
    return out


def ffill_state_columns(df, columns, time_col="time_bucket", limit=None,
                        observed_col=None, age_col=None):
    """Forward-fill sampled STATE columns along the time axis.
    columns (filled together), limit (staleness cap), observed_col, age_col.
    Returns: a NEW DataFrame in the original row order."""
    order = pd.to_datetime(df[time_col], utc=True, format="mixed").sort_values().index
    out = df.copy()
    cols = [c for c in columns if c in out.columns]

    block = out.loc[order, cols]           # time-ordered view of the state columns
    observed = block.notna().any(axis=1)   # bucket had its own observation
    out.loc[order, cols] = block.ffill(limit=limit)

    # metadata columns assign back by index, so they land on the right rows
    if observed_col:
        out[observed_col] = observed
    if age_col:
        # buckets since the last observed row: position - position of last observation
        pos = pd.Series(range(len(order)), index=order, dtype="float64")
        out[age_col] = pos - pos.where(observed).ffill()
    return out


def aggregate_by_time_bucket(df, time_col, group_cols, agg_func='sum', freq=None):
    if isinstance(agg_func, str):
        agg_dict = {col: agg_func for col in group_cols}
    else:
        agg_dict = agg_func

    if freq is None:
        result = df.groupby(time_col, as_index=False)[group_cols].agg(agg_dict)
    else:
        df = df.copy()
        # Remove " UTC" suffix if present, then parse
        df[time_col] = pd.to_datetime(
            df[time_col].astype(str).str.replace(' UTC', '', regex=False)
        )
        result = (
            df.set_index(time_col)
              .resample(freq)[group_cols]
              .agg(agg_dict)
              .reset_index()
        )

    return result
