"""Controlled, statistics-driven splits of the protocol-level feature panel.

This module turns the "split umbrellas" brainstormed for ``DF_common_final_1`` into
concrete, *decision-based* split functions. Each split is justified by statistics
(computed with :mod:`adv_validation` and :mod:`EDA`) rather than by eyeballing, and
the row splits are built to be **balanced** — every piece holds ~1/3 of the rows, so
no group dominates by sheer count.

Two axes, kept deliberately distinct (no two umbrellas use the same signal):

ROW-axis (segment the 4 380 time buckets into 3 balanced sub-frames)
    * :func:`split_by_volatility_regime`   — umbrella 5: rolling volatility of net flow
      (calm / normal / turbulent). A *dense* regime signal; the sparse, 72%-zero
      ``market_stress_index`` can't be balanced-terciled, so it is used to VALIDATE
      this split, not to build it.
    * :func:`split_by_activity_intensity`  — umbrella 6: transaction COUNT per bucket
      (quiet / active / peak). "How busy", independent of $ size.
    * :func:`split_by_whale_dominance`     — umbrella 7: average transaction SIZE
      (turnover / count) (retail / mixed / whale). "How concentrated", independent
      of how busy.

COLUMN-axis (group the 95 feature columns)
    * :func:`split_columns_by_correlation` — umbrella 2: co-movement clustering
      (which columns are redundant / move together).
    * :func:`split_columns_by_tail_risk`   — umbrella 3: heavy-tail / volatility tier
      (stable / moderate / wild), from per-column kurtosis, Hill index, robust CV.
    * :func:`split_columns_by_stat`        — split columns into TWO frames by one
      ``STAT_COLS`` statistic about a user threshold (the stat is read from the
      ``adv_validation`` profile, computed there); :func:`column_band_matrix` bands
      across all stats at once.

Each function returns a small result dict so the notebook can show the signal,
the labels, the balance, and the resulting pieces. Proof helpers
(:func:`split_balance`, :func:`group_stat_table`, :func:`cluster_coherence`,
:func:`column_group_profile`) produce the before/after evidence tables.

Libraries imported by this module:
    pandas, numpy          -> framing, quantile binning
    EDA                    -> rolling_volatility, correlation_clusters, per-column
                             tail metrics (excess kurtosis / Hill / robust CV)
    adv_validation         -> full robust per-column profile (statistical_validation)
                             for the deeper before/after evidence table
"""

import numpy as np
import pandas as pd

import  adv_validation as adv

import EDA
import adv_validation as adv

TIME_COL = "time_bucket"
STAT_COLS = adv.STAT_COLS

# Default signal columns (present in DF_common_final_1).
FLOW_COL = "net_liquidity_flow_usd"      # umbrella 5 — directional throughput
ACTIVITY_COL = "total_activity"          # umbrella 6 — transaction count
VOLUME_COL = "protocol_turnover_usd"     # umbrella 7 — $ throughput (numerator)


def numeric_columns(df, exclude=(TIME_COL,)):
    """Numeric feature columns, excluding the time key."""
    return [c for c in df.select_dtypes("number").columns if c not in exclude]


# --------------------------------------------------------------------------- #
# Per-column statistic resolver — the bridge that makes column splits generic
# --------------------------------------------------------------------------- #
def column_stat_series(df, stat, stats=None, columns=None, stat_col="column"):
    """Resolve ANY per-column statistic into a Series indexed by column name.

    ``stat`` may be any of:
      * a ``pd.Series`` / ``dict``  -> ``{column: value}`` used as-is;
      * a callable ``f(df, column) -> float``  -> evaluated per column;
      * a ``str`` naming a column of the ``stats`` profile table (e.g. ``"cv"``,
        ``"skewness"``, ``"p95"``) -> read straight from that table;
      * a ``str`` naming a function in :mod:`EDA` (e.g. ``"gini_coefficient"``,
        ``"hill_tail_index"``, ``"sharpe_ratio"``) -> ``EDA.<stat>(df, column)`` per
        column.

    This is what makes the column splits work for ANY statistic — whether it already
    lives in an ``adv_validation.statistical_validation`` profile or is computed on the
    fly from :mod:`EDA`. A ``str`` is looked up in ``stats`` first (cheaper, already
    computed), then in :mod:`EDA`.
    """
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
    """Label each value of ``s`` with a band: quantile (thresholds=None) or fixed cuts.

    ``thresholds=None`` -> equal-count ``pd.qcut`` into ``len(labels)`` bands. Otherwise
    fixed, left-closed ``pd.cut`` edges ``[-inf, *thresholds, inf]`` (value on an edge ->
    higher band); ``len(thresholds)`` must be ``len(labels) - 1``.
    """
    if thresholds is None:
        return pd.qcut(s, len(labels), labels=list(labels), duplicates="drop")
    if len(thresholds) != len(labels) - 1:
        raise ValueError("len(thresholds) must equal len(labels) - 1")
    edges = [-np.inf, *thresholds, np.inf]
    return pd.cut(s, bins=edges, labels=list(labels), right=False)


# --------------------------------------------------------------------------- #
# Shared row-split engine — one balanced tercile cut, reused by 5 / 6 / 7
# --------------------------------------------------------------------------- #
def _tercile_split(df, score, labels, signal_name):
    """Cut ``df`` into 3 balanced row groups by ``score`` (equal-count terciles).

    ``pd.qcut`` makes the three groups equal-count by construction (requirement:
    no weight difference between pieces). Rows whose score is NaN (e.g. the leading
    buckets of a rolling signal) are left unlabeled and excluded from every piece;
    the count is reported in the result as ``n_unlabeled``.

    Returns a dict: ``signal_name``, ``score`` (Series), ``labels`` (Categorical
    Series aligned to ``df.index``), ``cutpoints`` (the tercile edges), ``frames``
    ({label: sub-DataFrame}), and ``n_unlabeled``.
    """
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


# --------------------------------------------------------------------------- #
# Umbrella 5 — Temporal volatility regime (ROW axis)
# --------------------------------------------------------------------------- #
def split_by_volatility_regime(df, flow_col=FLOW_COL, window=12,
                               labels=("calm", "normal", "turbulent")):
    """Split rows by local volatility of ``flow_col`` (calm / normal / turbulent).

    Signal = rolling std of net liquidity flow over ``window`` buckets
    (12 = one day on the 2h grid) via :func:`EDA.rolling_volatility`. Dense, so its
    terciles are balanced — unlike the 72%-zero ``market_stress_index``, which this
    split is meant to be validated against rather than built from.
    """
    score = EDA.rolling_volatility(df, flow_col, window=window)
    return _tercile_split(df, score, labels,
                          signal_name=f"rolling_vol({flow_col}, w={window})")


# --------------------------------------------------------------------------- #
# Umbrella 6 — Activity intensity (ROW axis) — transaction COUNT
# --------------------------------------------------------------------------- #
def split_by_activity_intensity(df, activity_col=ACTIVITY_COL,
                                labels=("quiet", "active", "peak")):
    """Split rows by transaction count per bucket (quiet / active / peak).

    Distinct from the whale split: this is *how many* transactions, regardless of
    their $ size — a quiet bucket may still hold one huge whale trade.
    """
    score = pd.to_numeric(df[activity_col], errors="coerce")
    return _tercile_split(df, score, labels, signal_name=activity_col)


# --------------------------------------------------------------------------- #
# Umbrella 7 — Whale dominance (ROW axis) — average transaction SIZE
# --------------------------------------------------------------------------- #
def split_by_whale_dominance(df, volume_col=VOLUME_COL, count_col=ACTIVITY_COL,
                             labels=("retail", "mixed", "whale")):
    """Split rows by average transaction size = volume / count (retail / mixed / whale).

    The whale signal is *volume concentrated into few transactions* (high average
    size), not raw volume — so this is distinct from both the activity split (count)
    and a plain volume tiering.
    """
    vol = pd.to_numeric(df[volume_col], errors="coerce")
    cnt = pd.to_numeric(df[count_col], errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        score = vol / cnt.replace(0, np.nan)
    return _tercile_split(df, score, labels,
                          signal_name=f"avg_tx_size({volume_col} / {count_col})")


# --------------------------------------------------------------------------- #
# Umbrella 2 — Co-movement clustering (COLUMN axis)
# --------------------------------------------------------------------------- #
def split_columns_by_correlation(df, columns=None, n_clusters=6, absolute=False,
                                 method="pearson", linkage="complete"):
    """Group columns by co-movement via :func:`EDA.correlation_clusters`.

    Uses the SIGNED correlation ``r`` in (-1, 1) (``absolute=False``): positively
    co-moving columns (r -> +1) cluster together while anti-correlated columns
    (r -> -1) are pushed apart. Surfaces redundant feature families. Cluster sizes are
    structure-driven, so they are reported (not forced) — balance here means "no single
    mega-cluster swallowing everything", which ``linkage='complete'`` (the default)
    enforces; tune granularity via ``n_clusters``. Pass ``absolute=True`` to group by
    magnitude only (co-moving OR exactly opposite).

    Returns a dict: ``groups`` ({cluster: [columns]}) and ``n_clusters``.
    """
    cols = columns if columns is not None else numeric_columns(df)
    groups = EDA.correlation_clusters(df, cols, n_clusters=n_clusters,
                                      absolute=absolute, method=method,
                                      linkage=linkage)
    return {"groups": groups, "n_clusters": len(groups)}


# --------------------------------------------------------------------------- #
# Umbrella 3 — Tail-risk / volatility tier (COLUMN axis)
# --------------------------------------------------------------------------- #
def column_tail_scores(df, columns=None):
    """Per-column heavy-tail / dispersion metrics + a composite ``tail_score``.

    Three EDA metrics (all "bigger = wilder"): excess kurtosis, robust CV (IQR), and
    Hill heaviness (``-hill_tail_index``, so a heavier tail = larger). Each is turned
    into a percentile rank across columns (scale-free) and averaged into
    ``tail_score`` in [0, 1]. NaN metrics (e.g. Hill needs >=10 positive values) are
    skipped in the average.
    """
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
    """Tier columns into stable / moderate / wild by composite ``tail_score`` terciles.

    Equal-count terciles over the columns, so the three tiers hold ~the same number
    of columns. Returns a dict: ``scores`` (the per-column table), ``labels``
    (Categorical aligned to columns), and ``groups`` ({tier: [columns]}).
    """
    t = column_tail_scores(df, columns)
    cats = pd.qcut(t["tail_score"], 3, labels=list(labels), duplicates="drop")
    groups = {lab: t.index[cats == lab].tolist() for lab in cats.cat.categories}
    return {"scores": t, "labels": cats, "groups": groups}


# --------------------------------------------------------------------------- #
# Column split by ANY single per-column statistic (generic)
# --------------------------------------------------------------------------- #
def split_columns_by_stat(df, stat, threshold, stats=None, stat_col="column",
                          keep=(TIME_COL,)):
    """Split ``df``'s feature columns into TWO frames by one per-column statistic.

    Binary column split: every numeric feature of ``df`` is placed by whether its
    ``stat`` value is below ``threshold`` (low frame) or at/above it (high frame — a
    value exactly on the threshold goes high).

    ``stat`` — one of :data:`STAT_COLS` (an ``adv_validation.statistical_validation``
    profile column, e.g. ``"cv"``, ``"skewness"``, ``"p95"``, ``"excess_kurtosis"``).
    ``threshold`` — the user-defined cut value.
    ``stats`` — the precomputed profile table; if ``None`` it is computed with
    ``adv.statistical_validation(df, save=False)``. ALL statistic calculation lives in
    :mod:`adv_validation`; this module only reads the value and partitions the columns.

    The columns to split are taken from ``df`` itself (every numeric feature) — there is
    no column argument. Returns ``(low_df, high_df)``: the ``keep`` key column(s) followed
    by the ``< threshold`` and ``>= threshold`` columns respectively. Columns whose
    statistic is NaN are excluded from both.
    """
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
    """Band every column under MANY statistics at once -> a (columns x stats) label grid.

    Applies the same banding as :func:`split_columns_by_stat` for each name in
    ``stat_names`` and assembles the labels into one frame: rows = columns, columns =
    statistics, cells = band label. ``stat_names`` defaults to every numeric statistic in
    the ``stats`` profile table. A stat whose values can't be cut into ``len(labels)``
    bands (e.g. too many ties) is returned as an all-NaN column rather than raising.

    A compact view of how each feature ranks across all stats simultaneously — its CV
    band, skewness band, kurtosis band, ... side by side.
    """
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


# --------------------------------------------------------------------------- #
# Proof / evidence helpers (before vs after a split)
# --------------------------------------------------------------------------- #
def split_balance(labels):
    """Count + percentage of rows per group — the balance check for requirement 5."""
    vc = pd.Series(labels).value_counts(dropna=False)
    vc = vc.reindex(pd.Series(labels).cat.categories) if hasattr(labels, "cat") else vc
    out = pd.DataFrame({"n": vc, "pct": (vc / vc.sum() * 100).round(2)})
    return out


def group_stat_table(frames, columns, before=None, agg="median"):
    """One aggregate (median/mean/std/sum) per ``column`` for each row-group frame.

    With ``before`` (the full frame) included as the first column, this is the literal
    before/after evidence: how each metric shifts from the whole panel to each piece.
    """
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
    """Within-cluster vs overall mean ``r`` — proof clusters are internally coherent.

    A good clustering has within-cluster mean correlation ``r`` (signed, in (-1, 1))
    well above the overall average pairwise ``r``. Returns a per-cluster table (size,
    mean within-``r``) plus an ``overall`` row for the baseline.
    """
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
    """Rich robust per-column profile for each row-group via ``adv.statistical_validation``.

    Runs the full pre-EDA profile (``save=False`` — no files written) on each frame and
    stacks the chosen ``stats`` into one tidy table indexed by (column, group). With
    ``before`` (the full frame) added as a "ALL" group, this is the deeper before/after
    statistical evidence that a split actually changed the distribution, not just counts.
    """
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
    """Mean per-column tail metric for each column tier — proof tiers really differ.

    Confirms the 'wild' tier has higher kurtosis / dispersion and lower Hill α than
    'stable'. Returns a table indexed by tier with the mean of each metric + n_cols.
    """
    scores = column_tail_scores(df, [c for g in groups.values() for c in g])
    rows = []
    for name, cs in groups.items():
        sub = scores.loc[cs]
        row = {"n_cols": len(cs)}
        for m in metrics:
            row[f"mean_{m}"] = float(sub[m].mean(skipna=True))
        rows.append((name, row))
    return pd.DataFrame({name: r for name, r in rows}).T
