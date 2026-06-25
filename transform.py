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

    The timestamp suffix sorts lexicographically in chronological order, so the
    max string per query id is the most recent fetch.
    """
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


def _scale_value(value, dec):
    """Divide one value by ``10**dec`` exactly (Decimal -> float).

    Returns NaN when either the value or its decimals is missing, so scaling only
    happens when both are present. The big integer stays exact until the final cast.
    """
    if pd.isna(value) or pd.isna(dec):
        return float("nan")
    return float(Decimal(str(value)) / (Decimal(10) ** int(dec)))


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
        out[_scaled_name(col, suffix)] = [
            _scale_value(raw, dec) for raw, dec in zip(df[col], decs)
        ]

    if drop_raw:
        out = out.drop(columns=[c for c in cols if c in out.columns])
    return out


# --------------------------------------------------------------------------- #
# Per-COLUMN scaling: scale whole columns by a fixed decimals from a metric map
# --------------------------------------------------------------------------- #
# Companion to scale_by_decimals (per-ASSET, ``_raw`` columns). Some metrics carry a
# FIXED decimals that is the same for every row of the column rather than per-asset
# (e.g. Aave config metrics: supply_cap/borrow_cap -> 0, debt_ceiling -> 2, the bps
# fields -> 4). Their decimals live in a reference table keyed by ``metric`` (the
# column name) + ``decimals``; this scales each matched column by ``10**decimals``.

def column_decimals_map(decimals, metric_col="metric", decimals_col="decimals"):
    """Normalize a per-column decimals argument into a ``{column_name: int}`` lookup.

    Accepts:
      * ``dict`` / ``pd.Series`` -> used directly (index/keys are column names),
      * ``pd.DataFrame``         -> built from its ``metric`` + ``decimals`` columns
        (``metric`` holds the column name). Rows with a null metric or null decimals
        are dropped; duplicate metrics keep the first row.
    """
    if isinstance(decimals, pd.Series):
        return decimals.dropna().astype(int).to_dict()
    if isinstance(decimals, pd.DataFrame):
        ref = decimals.dropna(subset=[metric_col, decimals_col]).drop_duplicates(metric_col)
        return dict(zip(ref[metric_col], ref[decimals_col].astype(int)))
    return dict(decimals)


def scale_columns_by_decimals(df, decimals, columns=None, metric_col="metric",
                              decimals_col="decimals", overwrite=False,
                              drop_original=False, suffix="_scaled"):
    """Divide whole columns by ``10**decimals``, with a FIXED decimals PER COLUMN.

    Returns a NEW DataFrame (the input is not mutated). For each scaled column the
    value is divided by ``10**decimals`` only when both the value and its decimals are
    present; otherwise the result is NaN (same rule as :func:`scale_by_decimals`).

    Parameters
    ----------
    decimals : dict | pd.Series | pd.DataFrame
        Per-column decimals; see :func:`column_decimals_map`. A DataFrame is read from
        its ``metric_col`` (column name) and ``decimals_col``.
    columns : list[str] | None
        Columns to scale. Defaults to every ``df`` column found in the decimals map.
    overwrite : bool
        Write the scaled values back onto the source column. Otherwise a new
        ``<col><suffix>`` column is added.
    drop_original : bool
        Drop the source columns after scaling (ignored when ``overwrite`` is True).
    suffix : str
        Suffix for the scaled column when ``overwrite`` is False.
    """
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





# --------------------------------------------------------------------------- #
# Price multiplication: value each amount in USD and in ETH
# --------------------------------------------------------------------------- #
# The amount frame and the oracle-price frame are matched on (time_bucket, asset).
# The two frames store time_bucket in different string formats
# ("2025-11-01 00:00:00.000 UTC" vs "2025-11-01 00:00:00"), so both are normalized
# to a UTC datetime key before matching. Each amount column gets two new value
# columns: amount * USD price and amount * ETH price. The multiplication happens
# only when BOTH operands are present; a missing price (the asset/time is absent
# from the price table, or the price is null) or a missing amount yields NaN, so
# rows with no oracle price (e.g. dates the price table doesn't cover) are left
# blank rather than silently valued at 0.

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
    key on both sides. The product is computed only when both the amount and the
    matched price are present; if either is missing the result is NaN (not 0).

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