"""Modeling stage — daily credit-risk dataset assembly.

2h panel -> daily grain, joined with the 24h liq/user risk frames, restricted to
the credit-risk feature set, plus the forward-looking stress/volume targets.
"""

# TODO: the 7d variant waits on more bulk data.

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import transform as tf
import EDA as eda
from data_validation import canonicalize_keys
from domains.credit_risk.model_config import (TIME_COL, HORIZONS, TARGET_BASE_COL, REG_TARGET,
                          LIQ_CONDITIONAL_COLS, CREDIT_RISK_COLS, CREDIT_RISK_WEIGHTS,
                          CREDIT_RISK_PRIORITY, AGG_OVERRIDES, AGG_RULES,
                          RATIO_RECIPES, SHARE_RECIPES)


def daily_agg_map(columns):
    """Aggregation function per column for the 2h->daily rollup.
    columns -> AGG_OVERRIDES first, then the ordered AGG_RULES matchers.
    Returns: {column: agg}; unmatched columns fall back to mean."""
    out = {}
    for col in columns:
        if col in AGG_OVERRIDES:
            out[col] = AGG_OVERRIDES[col]
            continue
        for matcher, agg in AGG_RULES:
            if matcher(col):
                out[col] = agg
                break
        else:
            out[col] = "mean"
    return out


def build_daily_panel(df, time_col=TIME_COL, freq="24h"):
    """Aggregate the 2h protocol panel to daily grain with type-appropriate functions.
    df -> counts/flows summed, ratios and state averaged (via transform.aggregate_by_time_bucket).
    Returns: daily DataFrame with canonical string keys."""
    value_cols = [c for c in df.columns if c != time_col]
    amap = daily_agg_map(value_cols)
    daily = tf.aggregate_by_time_bucket(df, time_col, value_cols, agg_func=amap, freq=freq)
    return recompute_daily_ratios(canonicalize_keys(daily))


def recompute_daily_ratios(daily, ratios=RATIO_RECIPES, shares=SHARE_RECIPES):
    """Rebuild ratio columns from their daily-summed components (ratio-of-sums).
    df -> overwrites the mean-of-2h-ratios values left by the rollup.
    Returns: the frame with consistent daily ratios."""
    out = daily.copy()
    n = 0
    for col, (num, den) in ratios.items():
        if col in out.columns and num in out.columns and den in out.columns:
            out[col] = out[num] / out[den].replace(0, np.nan)
            n += 1
    for col, (num, dens) in shares.items():
        if col in out.columns and all(d in out.columns for d in dens):
            denom = sum(out[d] for d in dens)
            out[col] = out[num] / denom.replace(0, np.nan)
            n += 1
    print(f" recomputed {n} daily ratios as ratio-of-sums")
    return out


def join_model_frames(daily, liq, user, time_col=TIME_COL):
    """Inner-join the daily panel with the liquidation and user-account risk frames.
    Duplicated borrow-context columns are allclose-verified, then dropped.
    Returns: the joined frame; prints how many rows the inner joins drop."""
    daily = canonicalize_keys(daily)
    liq = canonicalize_keys(liq)
    user = canonicalize_keys(user)

    dup_liq = [c for c in ("borrow_tx_count", "unique_borrowers",
                           "borrow_amount_value_usd", "borrow_amount_value_eth")
               if c in liq.columns]
    dup_user = [c for c in ("borrow_amount_value_usd",) if c in user.columns]

    check = daily[[time_col] + dup_liq].merge(
        liq[[time_col] + dup_liq], on=time_col, suffixes=("_panel", "_liq"))
    for c in dup_liq:
        a = check[f"{c}_panel"].astype(float).to_numpy()
        b = check[f"{c}_liq"].astype(float).to_numpy()
        if not np.allclose(a, b, rtol=1e-6, equal_nan=True):
            raise AssertionError(f"borrow-context mismatch between panel and liq frame: {c}")

    merged = daily.merge(liq.drop(columns=dup_liq), on=time_col, how="inner")
    merged = merged.merge(user.drop(columns=dup_user), on=time_col, how="inner")
    dropped = max(len(daily), len(liq), len(user)) - len(merged)
    print(f" inner join: {len(merged)} rows kept, {dropped} dropped "
          f"(daily {len(daily)} / liq {len(liq)} / user {len(user)})")
    return merged


def fill_conditional_zeros(df, columns=LIQ_CONDITIONAL_COLS, indicator_col="has_liquidation",
                           count_col="liquidation_tx_count"):
    """Zero-fill conditional liquidation ratios; add the has-liquidation indicator.
    df -> NA ratios on zero-liquidation days become 0 (the honest value).
    Returns: the frame plus the indicator, keeping "no event" separable from "small event"."""
    out = df.copy()
    cols = [c for c in columns if c in out.columns]
    out[cols] = out[cols].fillna(0.0)
    if count_col in out.columns:
        out[indicator_col] = (out[count_col].fillna(0) > 0).astype(int)
    return out


# Columns that SUM across grains (counts, USD/ETH totals). unique_* and avg_* do not: a user
# active in two 2h buckets is one daily user, and an average of averages is not the average.
LIQ_ADDITIVE = ("as_collateral_tx_count", "as_debt_tx_count", "liquidation_tx_count",
                "liquidated_collateral_value_usd", "liquidated_collateral_value_eth",
                "liquidation_debt_covered_value_usd", "liquidation_debt_covered_value_eth")
USER_ADDITIVE = ("sampled_user_count", "account_data_call_count")


def _to_utc(series):
    """time_bucket strings ('2025-04-01 06:00:00.000 UTC') -> tz-aware timestamps."""
    return pd.to_datetime(series.astype(str).str.replace(" UTC", "", regex=False), utc=True)


def reconcile_grains(fine, coarse, columns, width, time_col=TIME_COL, rtol=1e-9, atol=1e-6,
                     label=""):
    """Check that a finer-grain frame sums to a coarser one (2h->24h, 24h->7d).
    fine/coarse: frames keyed by time_col (bucket START); columns: additive columns only;
    width: the coarse bucket length (pd.Timedelta). Fine rows are assigned to the coarse bucket
    that contains them; a coarse bucket with no fine rows must read 0 (event-only frames).
    Fine rows past the last coarse bucket are ignored (the 7d frame ends 6 days early).
    Returns: DataFrame [check, column, buckets, max_abs_diff, ok]; raises AssertionError on a
    failure, since a disagreement means the two frames describe different data."""
    cols = [c for c in columns if c in fine.columns and c in coarse.columns]
    starts = coarse[[time_col]].assign(_start=_to_utc(coarse[time_col])).sort_values("_start")
    fine = fine.assign(_t=_to_utc(fine[time_col])).sort_values("_t")
    fine = fine[fine["_t"] < starts["_start"].max() + width]
    assigned = pd.merge_asof(fine, starts[["_start"]], left_on="_t", right_on="_start")
    summed = assigned.groupby("_start")[cols].sum()
    want = coarse.assign(_start=_to_utc(coarse[time_col])).set_index("_start")[cols]
    got = summed.reindex(want.index).fillna(0.0)
    rows = []
    for c in cols:
        diff = (got[c].astype(float) - want[c].astype(float)).abs()
        ok = bool((diff <= atol + rtol * want[c].astype(float).abs()).all())
        rows.append({"check": label, "column": c, "buckets": len(want),
                     "max_abs_diff": float(diff.max()), "ok": ok})
    out = pd.DataFrame(rows)
    assert out["ok"].all(), (f"{label}: fine frame does not sum to the coarse frame in "
                             f"{out.loc[~out['ok'], 'column'].tolist()}")
    return out


def align_to_grid(df, grid, fill=None, time_col=TIME_COL):
    """Put a sparse frame on a dense grid of stamps, one row per grid stamp, in grid order.
    df: the sparse frame (df_liq_2h holds only buckets WITH a liquidation, df_user_account_2h only
    SAMPLED buckets); grid: the dense frame whose time_col defines the rows (e.g. the 2h panel),
    its other columns are kept; fill: what a missing bucket means — a scalar or {column: value}
    for an event frame (no row = no event = 0), None to leave NaN for a state frame (no row =
    unobserved). Stamps are matched as timestamps, not strings.
    Returns: grid + df's columns; raises if df has a stamp off the grid (it would be dropped)
    or a column the grid already has (the join would suffix it)."""
    clash = [c for c in df.columns if c != time_col and c in grid.columns]
    if clash:
        raise ValueError(f"column(s) {clash} in both frames — select them from one side only")
    g = grid.assign(_t=_to_utc(grid[time_col]))
    d = df.assign(_t=_to_utc(df[time_col])).drop(columns=[time_col])
    off = ~d["_t"].isin(g["_t"])
    if off.any():
        raise AssertionError(f"{int(off.sum())} stamp(s) not on the grid, e.g. "
                             f"{d.loc[off, '_t'].head(3).tolist()} — different grain or convention")
    out = g.merge(d, on="_t", how="left").drop(columns=["_t"])
    if fill is not None:
        out = out.fillna(fill if isinstance(fill, dict) else
                         {c: fill for c in df.columns if c != time_col})
    print(f" aligned {len(df)} rows onto a {len(grid)}-row grid "
          f"({len(grid) - len(df)} bucket(s) "
          + ("filled" if fill is not None else "left NaN") + ")")
    return out


def sync_priorities(weights=None):
    """Push the tiered CREDIT_RISK_WEIGHTS into column_priority.json (the EDA map).
    Replaces the old flat 2.0 sync — one weighting scheme for the whole project.
    Returns: the refreshed map DataFrame."""
    weights = CREDIT_RISK_WEIGHTS if weights is None else weights
    eda.register_columns(list(weights))
    return eda.update_priorities(dict(weights))


def select_credit_risk_columns(df, source="constant", threshold=CREDIT_RISK_PRIORITY):
    """Ordered training-feature list — credit-risk metrics only.
    source="constant" intersects CREDIT_RISK_COLS; "priority" takes score >= threshold.
    Returns: list of column names; missing expected columns are printed, not raised."""
    if source == "priority":
        pri = eda.get_priorities()
        wanted = pri.loc[pri["score"] >= threshold, "column"].tolist()
    else:
        wanted = list(CREDIT_RISK_COLS)
    present = [c for c in wanted if c in df.columns]
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        print(f" ⚠️ {len(missing)} expected credit-risk columns missing: {missing}")
    print(f" selected {len(present)} credit-risk feature columns")
    return present


def add_forward_targets(df, target_col=TARGET_BASE_COL, horizons=HORIZONS, time_col=TIME_COL):
    """Add forward-looking targets and trim the incomplete trailing window.
    target_next_1d = t+1; target_fwd_max_{h}d = max over t+1..t+h; y_reg_log1p = log1p of the former.
    Returns: the frame minus the last max(horizons) rows (count printed)."""

    # binarization into y_stress_{h}d waits until after the split — keeps the threshold train-only
    out = df.sort_values(time_col, kind="stable").reset_index(drop=True).copy()
    s = out[target_col].astype(float)
    out["target_next_1d"] = s.shift(-1)
    for h in horizons:
        fwd = s[::-1].rolling(h, min_periods=h).max()[::-1].shift(-1)
        out[f"target_fwd_max_{h}d"] = fwd
    out[REG_TARGET] = np.log1p(out["target_next_1d"])
    trim = max(horizons)
    print(f" dropped {trim} trailing rows with incomplete forward windows")
    return out.iloc[:-trim].reset_index(drop=True)


def load_pinned_frames(run, tables, expect_cols=None, time_col=TIME_COL):
    """Load labels from a pinned BatchRun, asserting every frame covers the run's window.
    run: tf.use_output_batch(BATCH); tables: labels to load; expect_cols: {label: n_cols}
    width guard — DF_common_final_2 is 61 columns, and the 30-col _1 would pass every other
    check while silently dropping 15 of the 53 features.
    Returns: {label: DataFrame}; raises on a frame from another window or the wrong width."""
    span = tuple(run.span)
    expect_cols = {} if expect_cols is None else expect_cols
    frames = {}
    for label in tables:
        df = run.load(label)
        got = tf.month_span(df, column=time_col)
        assert got == span, f"{label}: covers {got}, not the pinned batch {span}"
        want = expect_cols.get(label)
        assert want is None or df.shape[1] == want, (
            f"{label}: {df.shape[1]} columns, expected {want} — wrong frame version")
        frames[label] = df
        print(f" {label:<22} {len(df):>5} rows x {df.shape[1]:>3} cols")
    return frames


# Grain reconciliation: each finer frame must sum to the coarser one on its additive columns
# before anything is joined. These frames are never features (a 7d frame is stamped with the
# week's START but sums the whole week, so joining one to a day-t row would show the future) —
# they are the evidence that the 24h frames the model DOES use describe the same events.
RECONCILE_CHECKS = (
    ("liq 2h -> 24h",           "df_liq_2h",            "df_liq_24h",           LIQ_ADDITIVE,  "24h"),
    ("liq 24h -> 7d",           "df_liq_24h",           "df_liq_7d",            LIQ_ADDITIVE,  "7d"),
    ("liq features 24h -> 7d",  "DF_liq_features_24h",  "DF_liq_features_7d",   LIQ_ADDITIVE,  "7d"),
    ("user 2h -> 24h",          "df_user_account_2h",   "df_user_account_24h",  USER_ADDITIVE, "24h"),
    ("user 24h -> 7d",          "df_user_account_24h",  "df_user_account_7d",   USER_ADDITIVE, "7d"),
    ("user features 24h -> 7d", "DF_user_features_24h", "DF_user_features_7d",  USER_ADDITIVE, "7d"),
)


def reconcile_model_sources(frames, checks=RECONCILE_CHECKS, time_col=TIME_COL):
    """Run every RECONCILE_CHECKS pair that is present in `frames` through reconcile_grains.
    frames: {label: DataFrame} from load_pinned_frames; checks: (label, fine, coarse, cols, width).
    Returns: one report [check, column, buckets, max_abs_diff, ok]; raises on any mismatch.
    A pair whose frames were not loaded is skipped and named, never passed silently."""
    reports, skipped = [], []
    for label, fine, coarse, columns, width in checks:
        if fine in frames and coarse in frames:
            reports.append(reconcile_grains(frames[fine], frames[coarse], columns,
                                            pd.Timedelta(width), time_col=time_col, label=label))
        else:
            skipped.append(label)
    print(f" reconciled {len(reports)} grain pair(s)"
          + (f"; skipped, frames not loaded: {skipped}" if skipped else ""))
    return pd.concat(reports, ignore_index=True) if reports else pd.DataFrame()


# --- trailing (backward-looking) windows closed at t -------------------------------- #
# A trailing window is defined by TIME, not by a row count: for a row stamped t it sums the
# rows whose stamp falls in (t - window, t]. `window` is ANY fixed pandas span — '12h',
# '24h' (the default), '34h', '3d', '7d', '15d', '30d' — and is chosen at the call site; the
# code never assumes one. How many rows that covers follows from the frame's own grain
# (24h over a 2h panel is at most 12 rows, 7d is at most 84), and is FEWER wherever the
# panel has a gap — which a fixed `rolling(12)` would paper over by reaching further back
# than the window. The window is CLOSED AT t: it holds the row itself and nothing after it,
# so a trailing feature is leak-free by construction and can sit beside a forward-looking
# target. Sum ADDITIVE columns only (counts, USD/ETH totals): a trailing sum of a ratio or
# of an avg_* column has no meaning — see LIQ_ADDITIVE / USER_ADDITIVE.
#
# SEVERAL WINDOWS IN ONE CALL. Every function below takes `window` as either a single span or
# a sequence of them (TRAILING_WINDOWS), and writes one column per (column, window) pair named
# by trailing_name — borrow_tx_count_trail_12h / _trail_3d / _trail_7d. One pass over the
# sorted frame serves all of them. To choose the names, put WINDOW_FIELD in the suffix:
# suffix='rolled_{window}' gives _rolled_24h for one window and _rolled_12h / _rolled_3d /
# _rolled_7d for three, so a notebook switches window sets by changing ONE constant, and
# trailing_name(column, w, suffix) rebuilds the same names for a hand-made column. Refused,
# because each would silently lose a column: two spans of the same LENGTH, a plain suffix
# with no '{window}' across several windows, and a suffix mapping that sends two windows to
# one name.
#
# These windows NEST: 12h sits inside 3d sits inside 7d, so the three sums are correlated by
# construction (measured on this batch: log corr 0.70 between the nested 7d and 15d totals).
# That is fine for prediction and misleading for interpretation. If the point is to separate
# "what just happened" from "what happened earlier in the week", difference the bands
# (3d minus 12h, 7d minus 3d) rather than feeding the nested totals in as they are.
TRAILING_WINDOW = "24h"

# The vetted multi-window set — short / medium / week-long trailing context on one frame.
# Not a default: every function still defaults to the single TRAILING_WINDOW, so existing
# call sites are unchanged. Pass window=TRAILING_WINDOWS to get all three.
TRAILING_WINDOWS = ("12h", "3d", "7d")

# Placeholder a suffix (or a trailing_rows name) may carry; filled with each window's label.
WINDOW_FIELD = "{window}"


def as_window(window=TRAILING_WINDOW):
    """Validate a window argument and return it as a pd.Timedelta.
    Accepts anything pandas reads as a FIXED span ('90min', '12h', '34h', '3d', '15d').
    Rejects a calendar offset ('1M', 'W', 'Q'): its length varies, so neither .rolling nor
    a leak argument about "the last 30 days" would mean one thing. Raises ValueError."""
    try:
        delta = pd.Timedelta(window)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"window {window!r} is not a fixed time span — use '12h', '24h', "
                         f"'3d', '7d', '15d'; calendar offsets ('1M', 'W') vary in length "
                         f"and are not supported") from exc
    if pd.isna(delta):
        raise ValueError(f"window {window!r} has no length (pandas reads it as NaT) — pass a "
                         f"span such as '24h' or '7d'")
    if delta <= pd.Timedelta(0):
        raise ValueError(f"window {window!r} must be positive, got {delta}")
    return delta


def _window_label(delta):
    """Compact column-name label for a window passed as a Timedelta rather than a string.
    str(pd.Timedelta('3d')) is '3 days 00:00:00', which is not a column name; this gives '3d'."""
    total = delta.total_seconds()
    for unit, size in (("d", 86400), ("h", 3600), ("min", 60)):
        if total >= size and total % size == 0:
            return f"{int(total // size)}{unit}"
    return f"{total:g}s"


def _is_sequence(window):
    """True for a SET of windows (tuple, list, …); False for one span, a string included."""
    return not isinstance(window, str) and hasattr(window, "__iter__")


def _label_of(window):
    """(label, Timedelta) for ONE window. A string keeps its spelling, minus stray spaces, since
    it names the column ('3d' and '72h' are one length but read differently); anything else
    is labelled compactly (pd.Timedelta('3d') -> '3d')."""
    delta = as_window(window)
    return (window.strip() if isinstance(window, str) else _window_label(delta)), delta


def as_windows(window=TRAILING_WINDOW):
    """One window or several -> an ordered tuple of (label, pd.Timedelta), each via as_window.
    window: a single fixed span ('7d', pd.Timedelta('7d')) or a sequence of them
    (TRAILING_WINDOWS = ('12h', '3d', '7d')). Order is kept as given, so pass a list or tuple
    rather than a set — it is the order the output columns are written in.
    Two entries of the same LENGTH are refused — they would compute identical numbers twice.
    Returns: ((label, Timedelta), …); raises ValueError on an empty or duplicated set."""
    items = list(window) if _is_sequence(window) else [window]
    if not items:
        raise ValueError("no window given — pass a span such as '24h', or a sequence "
                         "such as TRAILING_WINDOWS ('12h', '3d', '7d')")
    pairs, seen = [], {}
    for item in items:
        label, delta = _label_of(item)
        if delta in seen:
            raise ValueError(f"windows {seen[delta]!r} and {label!r} are the same length "
                             f"({delta}) — they would compute the same numbers twice")
        seen[delta] = label
        pairs.append((label, delta))
    return tuple(pairs)


def _per_window(value, label, delta, n_windows, what="suffix"):
    """Resolve a per-window naming argument for ONE window of a call.
    value: None or '' (the default name), a string — which may carry WINDOW_FIELD, filled in
    later by the caller — or a {window: value} mapping, matched by LENGTH so a '72h' key
    serves a '3d' window. A plain string with no WINDOW_FIELD is refused across several
    windows: each would write the SAME column and only the last would survive."""
    if not value:
        return None
    if isinstance(value, str):
        if n_windows > 1 and WINDOW_FIELD not in value:
            raise ValueError(f"{what}={value!r} is one name but {n_windows} windows were asked "
                             f"for — every window would overwrite the same column. Put "
                             f"'{WINDOW_FIELD}' in it ({what}='rolled_{WINDOW_FIELD}'), pass None "
                             f"for the default name, or a mapping {{window: {what}}}")
        return value
    matches = [v for key, v in value.items() if as_window(key) == delta]
    if len(matches) != 1:
        raise ValueError(f"{what} mapping has {len(matches)} entries for window {label!r} "
                         f"(keys: {list(value)}) — it needs exactly one per window")
    return matches[0]


def _unique_names(named, what="suffix"):
    """[(label, column name), …] -> the names, refusing two windows that land on ONE column —
    the later would silently replace the earlier (e.g. a mapping {'12h': 'a', '3d': 'a'})."""
    seen = {}
    for label, name in named:
        if name in seen:
            raise ValueError(f"windows {seen[name]!r} and {label!r} both resolve to column "
                             f"{name!r} — give each window its own {what}")
        seen[name] = label
    return [name for _, name in named]


def _output_names(column, windows, suffix):
    """One trailing column name per window for `column` (trailing_name + _per_window), unique."""
    return _unique_names([(label, trailing_name(column, label,
                                                _per_window(suffix, label, delta, len(windows))))
                          for label, delta in windows])


def _shown(windows):
    """How a window set reads in a progress line: '24h' for one, ['12h', '3d', '7d'] for several."""
    labels = [label for label, _ in windows]
    return labels[0] if len(labels) == 1 else labels


def _time_rank(stamps):
    """Each row's position in time order — the inverse of a sort.
    Lets a value computed on the time-sorted frame be written back in the frame's OWN row
    order positionally, without relying on index labels (which may be duplicated or reset)."""
    values = stamps.to_numpy()
    return np.argsort(np.argsort(values, kind="stable"), kind="stable")


def _as_time_index(df, time_col=TIME_COL):
    """Time-sorted copy of df indexed by a tz-aware DatetimeIndex — the form .rolling needs.
    Raises on a duplicated stamp: a panel keyed by time_col has one row per bucket, and a
    duplicate would be counted twice inside every window that covers it.
    Returns: (sorted copy, rank array that maps a sorted result back to df's row order)."""
    if time_col not in df.columns:
        raise KeyError(f"'{time_col}' not in the frame; columns: {list(df.columns)[:8]}")
    stamps = _to_utc(df[time_col])
    if stamps.duplicated().any():
        n = int(stamps.duplicated().sum())
        raise AssertionError(f"{time_col} is not unique ({n} duplicate stamp(s), e.g. "
                             f"{stamps[stamps.duplicated()].head(3).tolist()}) — a trailing "
                             f"sum would double-count those buckets")
    indexed = df.copy()
    indexed.index = pd.DatetimeIndex(stamps.to_numpy(), name="_t")
    return indexed.sort_index(), _time_rank(stamps)


def frame_grain(df, time_col=TIME_COL):
    """The frame's own bucket spacing, as the MODAL gap between consecutive stamps.
    Modal, not minimum or mean: a sparse event frame (liquidations land in 28% of 2h buckets)
    has gaps of 2h, 4h, 6h…, and the smallest one is the real grain.
    Returns: pd.Timedelta, or pd.NaT for a frame with fewer than two rows."""
    stamps = _to_utc(df[time_col]).sort_values()
    gaps = stamps.diff().dropna()
    return pd.NaT if gaps.empty else gaps.mode().iloc[0]


def window_rows(df, window=TRAILING_WINDOW, time_col=TIME_COL):
    """How many rows a FULL `window` covers on this frame — the number to sanity-check against.
    window / grain rounded UP, because the window is half-open: 24h over a 2h panel is 12,
    7d is 84, 34h over a DAILY frame is 2 (t and t-24h both sit inside (t-34h, t]), and a
    window shorter than the grain is 1 (the row itself).
    Returns: int for ONE span, {label: int} whenever a sequence is passed (even of one, so a
    loop over the result never meets an int); None in place of a count when the grain cannot
    be read."""
    windows = as_windows(window)
    grain = frame_grain(df, time_col=time_col)
    if pd.isna(grain) or grain <= pd.Timedelta(0):
        counts = {label: None for label, _ in windows}
    else:                                                      # exact: both are integer ns
        counts = {label: max(1, -(-delta.value // grain.value)) for label, delta in windows}
    return counts if _is_sequence(window) else next(iter(counts.values()))


def trailing_name(column, window=TRAILING_WINDOW, suffix=None):
    """Where a trailing feature lands — the naming rule, so call sites don't hardcode strings.
    ('liquidation_tx_count', '7d') -> 'liquidation_tx_count_trail_7d'; `suffix` overrides, and a
    WINDOW_FIELD inside it is filled with the window label: suffix='rolled_{window}' ->
    'liquidation_tx_count_rolled_7d'. Use it to name a hand-built column (a product of rolled
    sums) exactly as the trailing_* functions would.
    window: ONE span — for a set, loop: [trailing_name(column, w, suffix) for w in windows]."""
    if _is_sequence(window):
        raise ValueError(f"trailing_name takes ONE window, got {window!r} — loop over the set: "
                         f"[trailing_name(column, w, suffix) for w in windows]")
    label, _ = _label_of(window)
    if not suffix:
        return f"{column}_trail_{label}"
    return f"{column}_{suffix.replace(WINDOW_FIELD, label)}"


def trailing_sum(df, column, window=TRAILING_WINDOW, time_col=TIME_COL, suffix=None,
                 min_periods=1):
    """Add the trailing sum of ONE column over (t - window, t] to the frame, per window.
    df: one row per time_col bucket, any row order; column: an additive column;
    window: any fixed pandas span, default TRAILING_WINDOW ('24h') — '12h', '34h', '3d',
    '7d' and '15d' all behave the same way, only the span changes — or a SEQUENCE of them
    (TRAILING_WINDOWS), which adds one column per window from a single sort of the frame;
    suffix: name override — 'rolled_{window}' (works for any number of windows), a plain
    string (one window only), or a {window: suffix} mapping;
    min_periods: how many rows a window needs before it reports a number rather than NaN.
    NaNs are skipped, not read as 0. df is not mutated and its row order is preserved.
    Returns: a copy of df with one new column per window (names from trailing_name)."""
    if column not in df.columns:
        raise KeyError(f"'{column}' not in the frame; columns: {list(df.columns)[:8]}")
    windows = as_windows(window)
    names = _output_names(column, windows, suffix)
    indexed, rank = _as_time_index(df, time_col=time_col)
    series = indexed[column].astype(float)
    out = df.copy()
    for (_, delta), name in zip(windows, names):
        rolled = series.rolling(delta, closed="right", min_periods=min_periods).sum()
        out[name] = rolled.to_numpy()[rank]
    return out


def trailing_rows(df, window=TRAILING_WINDOW, time_col=TIME_COL, name=None):
    """Add how many rows each trailing window actually covered — the QA column for a sum.
    Compare it with window_rows(df, window): a lower number means the panel has a gap there,
    or the row sits in the frame's first window, NOT that activity was low.
    window: one span or a sequence of them, as trailing_sum — one count column each, since a
    12h window being short says nothing about whether the 7d one was full.
    name: full column-name override — 'rows_{window}', a plain string (one window only), or a
    {window: name} mapping.
    Returns: a copy of df with an integer count column per window ('rows_trail_24h')."""
    windows = as_windows(window)
    named = []
    for label, delta in windows:
        chosen = _per_window(name, label, delta, len(windows), what="name")
        named.append((label, chosen.replace(WINDOW_FIELD, label) if chosen
                      else f"rows_trail_{label}"))
    names = _unique_names(named, what="name")
    indexed, rank = _as_time_index(df, time_col=time_col)
    ones = pd.Series(1.0, index=indexed.index)
    out = df.copy()
    for (_, delta), col in zip(windows, names):
        counts = ones.rolling(delta, closed="right", min_periods=1).sum()
        out[col] = counts.to_numpy()[rank].astype("int64")
    return out


def _ratio_operands(recipe):
    """(num, den) or (num, (den_a, den_b, …)) -> (num, dens tuple) — the RATIO_RECIPES and
    SHARE_RECIPES shapes read as one: a share is a ratio whose denominator is a sum."""
    num, den = recipe
    return num, ((den,) if isinstance(den, str) else tuple(den))


def _divide(parts, num, dens):
    """sum(num) / sum(dens) on an already-summed (or per-row) operand frame; 0 denominator -> NaN."""
    return parts[num] / parts[list(dens)].sum(axis=1, min_count=len(dens)).replace(0, np.nan)


def ratio_of_sums(df, recipes, window=TRAILING_WINDOW, time_col=TIME_COL, suffix=None,
                  min_periods=1, weight=None):
    """Ratio columns rebuilt as sum(num) / sum(dens) over (t - window, t] — never the mean of
    the bucket ratios, which weights a $10 bucket like a $10M one.
    recipes: {name: (num, den)} or {name: (num, (den_a, den_b))} — cfg.RATIO_RECIPES and
    cfg.SHARE_RECIPES both fit, merged or alone; window: any fixed span (as trailing_sum), a
    SEQUENCE of them (TRAILING_WINDOWS) for one ratio column per window, or None for the
    per-row ratio (overwrites `name`, like recompute_daily_ratios).
    suffix: name override, as trailing_sum ('rolled_{window}' for any number of windows).
    weight: optional column multiplied into every operand before the sum — for recipes whose
    operands are per-row AVERAGES (avg_ltv / avg_current_liquidation_threshold), so a day with
    40 sampled users counts 40x a day with 1. Cancels out when window is None.
    A bucket enters the window only if EVERY operand of that recipe (and the weight) was
    observed there, so numerator and denominator always sum over the same rows.
    Zero denominator -> NaN.
    Returns: a copy of df with one column per recipe PER WINDOW (name from trailing_name when
    windowed). The operands are masked once per recipe and re-summed per window."""
    if window is None:
        windows, frame, rank = (), df, None
    else:
        windows = as_windows(window)
        frame, rank = _as_time_index(df, time_col=time_col)
    if weight is not None and weight not in df.columns:
        raise KeyError(f"weight '{weight}' not in the frame")
    targets = {name: _output_names(name, windows, suffix) for name in recipes} if windows else {}
    out = df.copy()
    for name, recipe in recipes.items():
        num, dens = _ratio_operands(recipe)
        cols = list(dict.fromkeys((num, *dens)))      # a share's num is often one of its dens
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"{name}: operand(s) {missing} not in the frame")
        parts = frame[cols].astype(float)
        seen = parts.notna().all(axis=1)
        if weight is not None:
            w = frame[weight].astype(float)
            seen &= w.notna()
            parts = parts.mul(w, axis=0)
        parts = parts.where(seen)
        if not windows:
            out[name] = _divide(parts, num, dens).to_numpy()
            continue
        for (_, delta), col in zip(windows, targets[name]):
            summed = parts.rolling(delta, closed="right", min_periods=min_periods).sum()
            out[col] = _divide(summed, num, dens).to_numpy()[rank]
    print(f" {len(recipes)} ratio(s) rebuilt as ratio-of-sums"
          + (" per row" if window is None else f" over trailing {_shown(windows)}")
          + (f" ({len(recipes) * len(windows)} columns)" if len(windows) > 1 else "")
          + (f", weighted by {weight}" if weight else ""))
    return out


def trailing_means(df, columns, window=TRAILING_WINDOW, weight=None, time_col=TIME_COL,
                   suffix=None, min_periods=1):
    """Trailing mean of state columns over (t - window, t] — the rollup for columns that do
    NOT sum across rows (avg_*, risk_buffer): a summed average has no meaning.
    window: one span or a sequence of them (TRAILING_WINDOWS), one column per pair.
    weight: optional column (e.g. sampled_user_count) giving sum(w * x) / sum(w), the mean of
    the users rather than of the days; None gives the plain mean of the rows.
    A row enters only where the column (and the weight) is observed.
    Returns: a copy of df with one column per input column PER WINDOW (name from trailing_name)."""
    windows = as_windows(window)
    columns = list(columns)
    missing = [c for c in (*columns, *([weight] if weight else [])) if c not in df.columns]
    if missing:
        raise KeyError(f"column(s) {missing} not in the frame")
    targets = {column: _output_names(column, windows, suffix) for column in columns}
    indexed, rank = _as_time_index(df, time_col=time_col)
    w = (indexed[weight].astype(float) if weight
         else pd.Series(1.0, index=indexed.index))
    out = df.copy()
    for column in columns:
        x = indexed[column].astype(float)
        seen = x.notna() & w.notna()
        weighted, mass = (x * w).where(seen), w.where(seen)
        for (_, delta), name in zip(windows, targets[column]):
            total = weighted.rolling(delta, closed="right", min_periods=min_periods).sum()
            denom = mass.rolling(delta, closed="right", min_periods=min_periods).sum()
            out[name] = (total / denom.replace(0, np.nan)).to_numpy()[rank]
    print(f" trailing {_shown(windows)} means added for {len(columns)} column(s)"
          + (f" ({len(columns) * len(windows)} columns)" if len(windows) > 1 else "")
          + (f", weighted by {weight}" if weight else ""))
    return out


def trailing_sums(df, columns, window=TRAILING_WINDOW, time_col=TIME_COL, suffix=None,
                  min_periods=1, with_rows=False):
    """trailing_sum over several columns, plus optionally the window-size QA column.
    columns: additive columns only (LIQ_ADDITIVE / USER_ADDITIVE are the vetted lists);
    window: as trailing_sum — one span, or a sequence of them (TRAILING_WINDOWS) to get
    every column x every window in one pass; '24h' and '7d' sets can also be built by two
    separate calls, which is the same thing.
    Returns: a copy of df with one trailing column per (column, window) pair."""
    windows = as_windows(window)
    columns = list(columns)
    out = df
    for column in columns:
        out = trailing_sum(out, column, window=window, time_col=time_col, suffix=suffix,
                           min_periods=min_periods)
    if with_rows:
        out = trailing_rows(out, window=window, time_col=time_col)
    full = window_rows(df, window, time_col=time_col)
    print(f" trailing {_shown(windows)} sums added for {len(columns)} column(s)"
          + (f" ({len(columns) * len(windows)} columns)" if len(windows) > 1 else "")
          + f" · grain {frame_grain(df, time_col=time_col)}, a full window covers {full} row(s)")
    return out
