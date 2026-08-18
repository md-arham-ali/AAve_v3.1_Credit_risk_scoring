"""Statistics-driven splits of the protocol-level feature panel (DF_common_final_1).

Each split is justified by statistics from adv_validation / EDA rather than by eye,
and every row split is equal-count balanced — no piece dominates by sheer count.
"""

import numpy as np
import pandas as pd

# Make sibling src/ modules importable however this file is loaded (notebook, script, or import).
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import EDA
import adv_validation as adv

TIME_COL = "time_bucket"
STAT_COLS = adv.STAT_COLS

# Default signal columns (present in DF_common_final_1).
FLOW_COL = "net_liquidity_flow_usd"      # umbrella 5 — directional throughput
ACTIVITY_COL = "user_activity"           # umbrella 6 — per-bucket active-user count
VOLUME_COL = "protocol_turnover_usd"     # umbrella 7 — $ throughput (numerator)


def numeric_columns(df, exclude=(TIME_COL,)):
    """Numeric feature columns, excluding the time key."""
    # time key excluded — it is a string in these frames anyway
    return [c for c in df.select_dtypes("number").columns if c not in exclude]


# --- per-column statistic resolver: the bridge that makes column splits generic ---
def column_stat_series(df, stat, stats=None, columns=None, stat_col="column"):
    """Resolve ANY per-column statistic into a Series indexed by column name.
    stat: Series/dict, callable(df, col), a `stats` profile column, or an EDA function name.
    Returns: pd.Series[column -> float]. A str is looked up in `stats` first, then in EDA."""
    if isinstance(stat, pd.Series):
        return stat.astype("float64")
    if isinstance(stat, dict):
        return pd.Series(stat, dtype="float64")

    cols = list(columns) if columns is not None else numeric_columns(df)
    if callable(stat):
        return pd.Series({c: stat(df, c) for c in cols}, dtype="float64")
    if isinstance(stat, str):
        if stats is not None and stat in getattr(stats, "columns", ()):
            s = stats.set_index(stat_col)[stat]
            return (s.reindex(cols) if columns is not None else s).astype("float64")
        fn = getattr(EDA, stat, None)
        if callable(fn):
            return pd.Series({c: fn(df, c) for c in cols}, dtype="float64")
        raise ValueError(
            f"stat '{stat}' is not a column of `stats` nor a function in EDA")
    raise TypeError("stat must be a str, callable, dict, or pd.Series")


def _band_series(s, thresholds, labels):
    """Label each value of `s` with a band.
    thresholds=None -> equal-count qcut; otherwise left-closed cut on [-inf, *thresholds, inf].
    Returns: Categorical Series. len(thresholds) must be len(labels) - 1."""
    if thresholds is None:
        return pd.qcut(s, len(labels), labels=list(labels), duplicates="drop")
    if len(thresholds) != len(labels) - 1:
        raise ValueError("len(thresholds) must equal len(labels) - 1")
    edges = [-np.inf, *thresholds, np.inf]
    return pd.cut(s, bins=edges, labels=list(labels), right=False)


# --- shared row-split engine: one balanced tercile cut, reused by umbrellas 5 / 6 / 7 ---
def _tercile_split(df, score, labels, signal_name):
    """Cut `df` into 3 equal-count row groups by `score`.
    df, score Series, signal_name, labels -> qcut, NaN scores left unlabeled and excluded.
    Returns: {signal_name, score, labels, cutpoints, frames, n_unlabeled}."""
    score = pd.Series(np.asarray(score, dtype="float64"), index=df.index)
    cats, edges = pd.qcut(score, 3, labels=list(labels), duplicates="drop",
                          retbins=True)
    frames = {lab: df.loc[cats[cats == lab].index] for lab in cats.cat.categories}
    return {
        "signal_name": signal_name,
        "score": score,
        "labels": cats,
        "cutpoints": list(edges),
        "frames": frames,
        "n_unlabeled": int(cats.isna().sum()),
    }


# --- umbrella 5: temporal volatility regime (ROW axis) ---
def split_by_volatility_regime(df, flow_col=FLOW_COL, window=12,
                               labels=("calm", "normal", "turbulent")):
    """Split rows by local volatility of flow_col (calm / normal / turbulent).
    Signal = rolling std over `window` buckets (12 = one day on the 2h grid), via EDA.rolling_volatility.
    Returns: the _tercile_split dict."""

    # dense signal, so its terciles balance — unlike the 72%-zero market_stress_index,
    # which this split validates against rather than builds from
    score = EDA.rolling_volatility(df, flow_col, window=window)
    return _tercile_split(df, score, labels,
                          signal_name=f"rolling_vol({flow_col}, w={window})")


# --- umbrella 6: activity intensity (ROW axis) — transaction COUNT ---
def split_by_activity_intensity(df, activity_col=ACTIVITY_COL,
                                labels=("quiet", "active", "peak")):
    """Split rows by transaction count per bucket (quiet / active / peak).
    df, activity_col -> counts only, regardless of $ size.
    Returns: the _tercile_split dict. A quiet bucket may still hold one huge whale trade."""
    score = pd.to_numeric(df[activity_col], errors="coerce")
    return _tercile_split(df, score, labels, signal_name=activity_col)


# --- umbrella 7: whale dominance (ROW axis) — average transaction SIZE ---
def split_by_whale_dominance(df, volume_col=VOLUME_COL, count_col=ACTIVITY_COL,
                             labels=("retail", "mixed", "whale")):
    """Split rows by average transaction size = volume / count (retail / mixed / whale).
    df, volume_col, activity_col.
    Returns: the _tercile_split dict — concentration, not raw volume."""
    vol = pd.to_numeric(df[volume_col], errors="coerce")
    cnt = pd.to_numeric(df[count_col], errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        score = vol / cnt.replace(0, np.nan)
    return _tercile_split(df, score, labels,
                          signal_name=f"avg_tx_size({volume_col} / {count_col})")


# --- umbrella 2: co-movement clustering (COLUMN axis) ---
def split_columns_by_correlation(df, columns=None, n_clusters=6, absolute=False,
                                 method="pearson", linkage="complete"):
    """Group columns by co-movement via EDA.correlation_clusters.
    Signed r by default (absolute=False), so anti-correlated columns are pushed apart; n_clusters tunes granularity.
    Returns: {groups: {cluster: [columns]}, n_clusters} — sizes are reported, not forced."""
    cols = columns if columns is not None else numeric_columns(df)
    groups = EDA.correlation_clusters(df, cols, n_clusters=n_clusters,
                                      absolute=absolute, method=method,
                                      linkage=linkage)
    return {"groups": groups, "n_clusters": len(groups)}


# --- umbrella 3: tail-risk / volatility tier (COLUMN axis) ---
def column_tail_scores(df, columns=None):
    """Per-column heavy-tail / dispersion metrics + a composite tail_score.
    Excess kurtosis, robust CV (IQR), Hill heaviness — each percentile-ranked, then averaged.
    Returns: DataFrame with tail_score in [0, 1]; NaN metrics skipped in the average."""
    cols = columns if columns is not None else numeric_columns(df)
    t = pd.DataFrame({
        "excess_kurtosis": column_stat_series(df, "excess_kurtosis", columns=cols),
        "hill_tail_index": column_stat_series(df, "hill_tail_index", columns=cols),
        "robust_cv_iqr": column_stat_series(
            df, lambda d, c: EDA.robust_cv(d, c, method="iqr"), columns=cols),
    })
    t.index.name = "column"
    ranks = pd.concat([
        t["excess_kurtosis"].rank(pct=True),
        t["robust_cv_iqr"].rank(pct=True),
        (-t["hill_tail_index"]).rank(pct=True),     # heavier tail (lower α) -> higher
    ], axis=1)
    t["tail_score"] = ranks.mean(axis=1, skipna=True)
    return t.sort_values("tail_score", ascending=False)


def split_columns_by_tail_risk(df, columns=None,
                               labels=("stable", "moderate", "wild")):
    """Tier columns into stable / moderate / wild by composite tail_score terciles.
    df, optional precomputed scores -> equal-count terciles over the columns.
    Returns: {scores, labels, groups}."""
    t = column_tail_scores(df, columns)
    cats = pd.qcut(t["tail_score"], 3, labels=list(labels), duplicates="drop")
    groups = {lab: t.index[cats == lab].tolist() for lab in cats.cat.categories}
    return {"scores": t, "labels": cats, "groups": groups}


# --- generic column split by ANY single per-column statistic ---
def split_columns_by_stat(df, stat, threshold, stats=None, stat_col="column",
                          keep=(TIME_COL,)):
    """Split df's numeric feature columns into TWO frames by one per-column statistic.
    stat: a STAT_COLS name; threshold: the cut (a value exactly on it goes high); stats: optional profile.
    Returns: (low_df, high_df), each led by the `keep` key columns; NaN-stat columns dropped from both."""

    # NOTE: all statistic calculation lives in adv_validation — this only reads and partitions
    if stat not in STAT_COLS:
        raise ValueError(f"stat must be one of STAT_COLS: {list(STAT_COLS)}")
    if stats is None:
        stats = adv.statistical_validation(df, save=False)
    s = column_stat_series(df, stat, stats=stats, stat_col=stat_col)

    low_cols = s.index[s < threshold].tolist()
    high_cols = s.index[s >= threshold].tolist()
    keep_cols = [c for c in keep if c in df.columns]
    low_df = df[keep_cols + [c for c in low_cols if c in df.columns]]
    high_df = df[keep_cols + [c for c in high_cols if c in df.columns]]
    return low_df, high_df


def column_band_matrix(df, stats=None, stat_names=None, thresholds=None,
                       labels=("low", "moderate", "high"), columns=None,
                       stat_col="column"):
    """Band every column under MANY statistics at once.
    stat_names defaults to every numeric stat in the profile; same banding as split_columns_by_stat.
    Returns: a (columns x stats) label grid; an uncuttable stat comes back all-NaN, never raising."""
    if stat_names is None:
        if stats is None:
            raise ValueError("pass stat_names, or a `stats` table to default them from")
        stat_names = [c for c in stats.select_dtypes("number").columns if c != stat_col]

    out = {}
    for name in stat_names:
        s = column_stat_series(df, name, stats=stats, columns=columns, stat_col=stat_col)
        try:
            out[name] = _band_series(s, thresholds, labels)
        except ValueError:                       # not enough distinct values to band
            out[name] = pd.Series(np.nan, index=s.index)
    return pd.DataFrame(out)


# --- proof / evidence helpers (before vs after a split) ---
def split_balance(labels):
    """Count + percentage of rows per group — the balance check for requirement 5."""
    # sanity check that the terciles really came out ~1/3 each
    vc = pd.Series(labels).value_counts(dropna=False)
    vc = vc.reindex(pd.Series(labels).cat.categories) if hasattr(labels, "cat") else vc
    out = pd.DataFrame({"n": vc, "pct": (vc / vc.sum() * 100).round(2)})
    return out


def group_stat_table(frames, columns, before=None, agg="median"):
    """One aggregate (median/mean/std/sum) per `column` for each row-group frame.
    frames, column, agg, before (the full frame) as the leading column.
    Returns: the before/after evidence table."""
    funcs = {"median": np.nanmedian, "mean": np.nanmean,
             "std": lambda a: np.nanstd(a, ddof=1), "sum": np.nansum}
    f = funcs[agg]

    def col_agg(frame, col):
        return float(f(pd.to_numeric(frame[col], errors="coerce").to_numpy()))

    data = {}
    if before is not None:
        data[f"ALL ({agg})"] = {c: col_agg(before, c) for c in columns}
    for name, fr in frames.items():
        data[str(name)] = {c: col_agg(fr, c) for c in columns}
    return pd.DataFrame(data).loc[columns]


def cluster_coherence(df, groups, method="pearson"):
    """Within-cluster vs overall mean r — proof the clusters are internally coherent.
    df, groups -> signed r in (-1, 1).
    Returns: per-cluster table (size, mean within-r) plus an `overall` baseline row."""
    cols = [c for g in groups.values() for c in g]
    corr = df[cols].apply(pd.to_numeric, errors="coerce").corr(method=method)

    def mean_within(cs):
        if len(cs) < 2:
            return np.nan
        sub = corr.loc[cs, cs].to_numpy()
        iu = np.triu_indices(len(cs), k=1)
        return float(sub[iu].mean())

    rows = [{"group": name, "n_cols": len(cs), "mean_within_r": mean_within(cs)}
            for name, cs in groups.items()]
    overall_iu = np.triu_indices(len(cols), k=1)
    overall = float(corr.to_numpy()[overall_iu].mean())
    out = pd.DataFrame(rows).set_index("group")
    out.loc["overall"] = [len(cols), overall]
    return out


def robust_profile(frames, columns, stats=("mean", "std", "cv", "skewness",
                                           "excess_kurtosis", "null_pct", "zero_pct"),
                   before=None):
    """Rich robust per-column profile per row-group, via adv.statistical_validation(save=False).
    frames, stats to keep, before (added as the "ALL" group).
    Returns: one tidy table indexed by (column, group) — the deeper before/after evidence."""
    pieces = {}
    if before is not None:
        pieces["ALL"] = before
    pieces.update({str(k): v for k, v in frames.items()})

    blocks = []
    for name, fr in pieces.items():
        prof = adv.statistical_validation(fr, table_name=name, columns=list(columns),
                                          save=False).set_index("column")
        block = prof[list(stats)].copy()
        block.insert(0, "group", name)
        blocks.append(block)
    return (pd.concat(blocks).reset_index()
              .set_index(["column", "group"]).sort_index())


def column_group_profile(df, groups, metrics=("excess_kurtosis", "hill_tail_index",
                                              "robust_cv_iqr")):
    """Mean per-column tail metric for each column tier — proof the tiers really differ.
    scores table, labels.
    Returns: table indexed by tier with each metric's mean + n_cols."""

    # sanity check: 'wild' should show higher kurtosis and lower Hill alpha than 'stable'
    scores = column_tail_scores(df, [c for g in groups.values() for c in g])
    rows = []
    for name, cs in groups.items():
        sub = scores.loc[cs]
        row = {"n_cols": len(cs)}
        for m in metrics:
            row[f"mean_{m}"] = float(sub[m].mean(skipna=True))
        rows.append((name, row))
    return pd.DataFrame({name: r for name, r in rows}).T
