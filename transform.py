"""Transformation helpers for the Aave V3.1 Dune result tables.

This stage READS the versioned CSVs in ``query_result_data/`` and never writes
back to that folder — the raw extracts stay untouched. Because each fetch saves
a new ``query_result_data_{query_id}_{fetch_time}.csv``, the loaders here resolve
the *latest* version per table by default.

Step 1 (this module) is just loading: turn a table label into a DataFrame.
Later transformation logic (column ordering, derived columns, ...) builds on top.

Libraries imported by this module:
    pandas                -> read CSVs into DataFrames
    re, pathlib           -> standard library (filename parsing)
    data_validation       -> reuse the query-id -> table-label map (single source
                             of truth) so labels never drift between stages
"""

import re
from decimal import Decimal
from pathlib import Path

import pandas as pd

from data_validation import TABLE_LABELS, table_name_from_path

# Folder holding the raw, versioned extracts. Read-only from this stage.
SOURCE_DIR = "query_result_data"

# Matches query_result_data_<query_id>_<YYYYMMDDTHHMMSSZ>.csv
_NAME_RE = re.compile(r"query_result_data_(\d+)_(\d{8}T\d{6}Z)$")


def _parse_name(path):
    """Return (query_id, timestamp) from a versioned CSV path, or (None, None)."""
    match = _NAME_RE.search(Path(path).stem)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def latest_paths(source_dir=SOURCE_DIR):
    """Map each table label -> its newest versioned CSV path.

    The timestamp suffix sorts lexicographically in chronological order, so the
    max string per query id is the most recent fetch.
    """
    newest = {}  # query_id -> (timestamp, path)
    for path in Path(source_dir).glob("query_result_data_*.csv"):
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

    ``table`` is either a table label (e.g. "supply_withdraw"), whose latest
    version is resolved automatically, or a direct path to a specific CSV.
    Values are left as-loaded (time_bucket / asset stay strings).
    """
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


# --------------------------------------------------------------------------- #
# Column division: scale raw integer amounts to real token units
# --------------------------------------------------------------------------- #
# The extracts keep token amounts as raw integers (object/strings) in columns
# ending in ``_raw``. Real token units = raw / 10**decimals, where ``decimals``
# is per-asset. Division uses ``Decimal`` so the big integers stay exact before
# the result is cast to float.
RAW_SUFFIX = "_raw"


def raw_amount_columns(df, suffix=RAW_SUFFIX):
    """Return the columns holding raw integer amounts (those ending in ``_raw``)."""
    return [c for c in df.columns if c.endswith(suffix)]


def decimals_map(decimals, asset_col="asset", decimals_col="decimals", unit_col="unit",
                 token_unit="raw_token_amount"):
    """Normalize a ``decimals`` argument into an ``{asset_address: int}`` lookup.

    Accepts:
      * ``int``                  -> same decimals for every asset (returned as-is),
      * ``dict`` / ``pd.Series`` -> used directly,
      * ``pd.DataFrame``         -> built from its asset + decimals columns. If the
        frame has a ``unit`` column (the decimals-reference table), only the
        ``raw_token_amount`` rows are used so block-number rows (decimals 0)
        don't override token decimals. Duplicate assets keep the first row.
    """
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


def scale_by_decimals(df, decimals, columns=None, asset_col="asset",
                      decimals_col="decimals", drop_raw=False, suffix="_scaled"):
    """Divide raw integer-amount columns by ``10**decimals`` -> real token units.

    Returns a NEW DataFrame (the input is not mutated).

    Parameters
    ----------
    decimals : int | dict | pd.Series | pd.DataFrame
        Per-asset token decimals; see :func:`decimals_map`. A DataFrame may be a
        decimals-reference table OR any frame that carries per-asset decimals (e.g.
        the oracle-price table, whose ``decimals`` column is read the same way).
    columns : list[str] | None
        Raw columns to scale. Defaults to every column ending in ``_raw``.
    asset_col : str
        Asset column — used both to look up per-asset decimals on ``df`` and to read
        the asset key from a ``decimals`` DataFrame (ignored when ``decimals`` is int).
    decimals_col : str
        Name of the decimals column to read from a ``decimals`` DataFrame.
    drop_raw : bool
        Drop the original raw columns after scaling.
    suffix : str
        Suffix for the scaled column when the source name doesn't end in ``_raw``.
        Columns ending in ``_raw`` become the same name without the suffix
        (e.g. ``supply_amount_raw`` -> ``supply_amount``).
    """
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
        raws = list(df[col])
        scaled = [
            float(Decimal(str(raw)) / (Decimal(10) ** int(dec)))
            if not (pd.isna(raw) or pd.isna(dec)) else float("nan")
            for raw, dec in zip(raws, decs)
        ]
        out[_scaled_name(col, suffix)] = scaled

    if drop_raw:
        out = out.drop(columns=[c for c in cols if c in out.columns])
    return out





# --------------------------------------------------------------------------- #
# Price multiplication: value each amount in USD and in ETH
# --------------------------------------------------------------------------- #
# The amount frame and the oracle-price frame are matched on (time_bucket, asset).
# The two frames store time_bucket in different string formats
# ("2025-11-01 00:00:00.000 UTC" vs "2025-11-01 00:00:00"), so both are normalized
# to a UTC datetime key before matching. Each amount column gets two new value
# columns: amount * USD price and amount * ETH price. A missing price (the asset/time
# is absent from the price table, or the price is null) or a missing amount counts
# as 0, so the resulting value is 0 rather than NaN.

def multiply_by_price(df, prices, amount_columns, time_col="time_bucket",
                      asset_col="asset", usd_col="avg_price_usd",
                      eth_col="avg_price_eth", usd_suffix="_value_usd",
                      eth_suffix="_value_eth"):
    """Value each amount column in USD and ETH by matching ``df`` to a price table.

    Returns a NEW DataFrame (the input is not mutated). For every column in
    ``amount_columns`` two columns are added::

        <col><usd_suffix> = amount * USD price
        <col><eth_suffix> = amount * ETH price

    Rows are matched on (``time_col``, ``asset_col``). The two frames may store
    ``time_col`` in different string formats, so it is normalized to a UTC datetime
    key on both sides. Any missing price or amount is treated as 0.

    Parameters
    ----------
    df : pd.DataFrame
        The amount frame (e.g. the scaled supply/withdraw amounts).
    prices : pd.DataFrame
        The oracle-price frame holding the USD and ETH price columns.
    amount_columns : list[str]
        The (real-unit) amount columns to value.
    """
    out = df.copy()

    # normalized match key on both sides (the tables differ in time_bucket format)
    left_key = pd.to_datetime(df[time_col], utc=True, format="mixed")
    right_key = pd.to_datetime(prices[time_col], utc=True, format="mixed")

    # one price row per (time, asset): usd + eth, with null prices -> 0
    price = pd.DataFrame({
        "_t": right_key,
        "_a": prices[asset_col],
        "_usd": pd.to_numeric(prices[usd_col], errors="coerce").fillna(0.0),
        "_eth": pd.to_numeric(prices[eth_col], errors="coerce").fillna(0.0),
    }).drop_duplicates(["_t", "_a"])

    # left-join keeps df's row order; an unmatched (time, asset) -> 0 price
    merged = pd.DataFrame({"_t": left_key, "_a": df[asset_col]}).merge(
        price, on=["_t", "_a"], how="left")
    usd = merged["_usd"].fillna(0.0).to_numpy()
    eth = merged["_eth"].fillna(0.0).to_numpy()

    for col in amount_columns:
        amount = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
        out[f"{col}{usd_suffix}"] = amount * usd
        out[f"{col}{eth_suffix}"] = amount * eth
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