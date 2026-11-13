"""Modeling stage — the split, defined by TIME, for the 2h DF_model (features) + DF_target.

The two frames stay apart: this stage DEFINES the split and every train-fitted step and writes
them to split_meta.json. Nothing joined is written; training joins one target column at a time
(model_dataset.join_for_training) and keeps the rows assign_split labels for it.

- Holdout: the last `holdout` (15d) — prediction + later validation, in no split.
- 70/15/15 by time over the rest; boundaries are timestamps, not row counts.
- Embargo = the model's horizon: a row is dropped when its label window (t, t+h] would reach
  into the next split. Warm-up: the first 7d of train (the trailing fill has no past there).
- Train-fit: column selection (near-constant, mostly blank, |Spearman| duplicates) — no
  reshaping, the features are already scaled — and mean/std for distance/gradient models only.
- Labels: regression on every target; binary stress per spec from train-only thresholds.

fit_scaler / apply_scaler at the end are what training reads back through split_meta.json.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import re

import numpy as np
import pandas as pd

from model_config import TIME_COL, STRESS_QUANTILE, SPLIT_FRACTIONS, NEAR_CONSTANT_FRAC

SPLITS = ("train", "val", "test")
HOLDOUT = "15d"
WARMUP = "7d"


# --- inputs ------------------------------------------------------------------------- #

def time_of(df, time_col=TIME_COL):
    """time_bucket strings ('2025-04-01 06:00:00.000 UTC') -> tz-aware timestamps."""
    return pd.to_datetime(df[time_col].astype(str).str.replace(" UTC", "", regex=False), utc=True)


def horizon_of(column):
    """The forward window of a target column: 'x_fwd_3d' -> Timedelta('3d')."""
    head, sep, label = column.rpartition("_fwd_")
    if not sep:
        raise ValueError(f"'{column}' is not a forward target (no '_fwd_<window>' suffix)")
    return pd.Timedelta(label)


def _is_several(src):
    """True for a LIST of sources (several frames to join), False for one source.
    One source is a CSV path or a (folder, table) pair, and JSON turns that pair into a 2-item
    list, so a list counts as several only when every item is itself a source: a .csv path or a pair."""
    return (isinstance(src, (list, tuple)) and len(src) > 0
            and all(isinstance(s, (list, tuple)) or str(s).endswith(".csv") for s in src))


def join_inputs(frames, names=None, time_col=TIME_COL):
    """Several frames on ONE time grid -> one frame, columns side by side (panel + liq + user ...).
    Every frame must hold exactly the first frame's stamps. A column name that repeats is dropped from
    the later frame when its values are identical (a shared operand), and raises when they differ:
    nothing is overwritten silently.
    Returns: the joined frame, first frame's column order first."""
    names = names or [f"frame {i}" for i in range(len(frames))]
    out, stamps = frames[0].copy(), time_of(frames[0], time_col).to_numpy()
    for frame, name in zip(frames[1:], names[1:]):
        if len(frame) != len(out) or (time_of(frame, time_col).to_numpy() != stamps).any():
            raise AssertionError(f"{name} is not on the first frame's time grid: join only frames of one batch")
        for c in frame.columns:
            if c == time_col:
                continue
            if c in out.columns:
                a, b = pd.to_numeric(out[c], errors="coerce"), pd.to_numeric(frame[c], errors="coerce")
                if not np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True):
                    raise AssertionError(f"column {c!r} is in two frames with different values ({name})")
                print(f" {c}: same column in {name}, kept once")
                continue
            out[c] = frame[c].to_numpy()
    return out


def load_input(src, batch=None):
    """Load DF_model / DF_target. src: a CSV path, or (folder, table) for a frame saved with
    run.save (resolved in `batch`); or a LIST of such sources, joined on time (join_inputs).
    Returns the frame sorted by time, one row per stamp."""
    import transform as tf
    if src is None:
        raise ValueError("no source set — fill DF_MODEL_SRC / DF_TARGET_SRC in the config cell")
    if _is_several(src):
        return join_inputs([load_input(s, batch) for s in src], [str(s) for s in src])
    df = (tf.load_table(str(src)) if isinstance(src, (str, os.PathLike))
          else tf.load_output(src[1], batch=batch, out_dir=src[0]))
    t = time_of(df)
    if t.duplicated().any():
        raise AssertionError(f"{int(t.duplicated().sum())} duplicate {TIME_COL} in {src}")
    return df.iloc[np.argsort(t.to_numpy(), kind="stable")].reset_index(drop=True)


# --- boundaries and row labels ------------------------------------------------------ #

def split_boundaries(stamps, fractions=SPLIT_FRACTIONS, holdout=HOLDOUT):
    """Time boundaries of the split. stamps: every time_bucket of the grid.
    Holdout = rows with t > last - holdout (the same rows whose 15d target is pending).
    The rest [start, holdout_after] is cut by TIME into fractions, each edge snapped to the
    first stamp at or after it. Returns: {start, val_start, test_start, holdout_after, end}."""
    if abs(sum(fractions) - 1) > 1e-9:
        raise ValueError(f"fractions {fractions} do not sum to 1")
    t = pd.Series(pd.DatetimeIndex(stamps)).sort_values().reset_index(drop=True)
    start, end = t.iloc[0], t.iloc[-1]
    holdout_after = end - pd.Timedelta(holdout)
    span = holdout_after - start
    snap = lambda edge: t[t >= edge].iloc[0]
    return {"start": start, "val_start": snap(start + span * fractions[0]),
            "test_start": snap(start + span * (fractions[0] + fractions[1])),
            "holdout_after": holdout_after, "end": end}


def boundary_table(bounds, stamps):
    """train / val / test / holdout as rows: first and last stamp, days and rows."""
    t = pd.Series(pd.DatetimeIndex(stamps))
    masks = {"train": t < bounds["val_start"],
             "val": (t >= bounds["val_start"]) & (t < bounds["test_start"]),
             "test": (t >= bounds["test_start"]) & (t <= bounds["holdout_after"]),
             "holdout": t > bounds["holdout_after"]}
    rows = [{"split": k, "first": t[m].min(), "last": t[m].max(), "rows": int(m.sum()),
             "days": round((t[m].max() - t[m].min()) / pd.Timedelta("1d"), 1)}
            for k, m in masks.items()]
    return pd.DataFrame(rows)


def assign_split(df, bounds, horizon, warmup=WARMUP, time_col=TIME_COL):
    """Per-row split label for a model of this horizon: train / val / test, or why the row
    is out — 'embargo' (its label window (t, t + h] reaches the next split), 'warmup' (first
    `warmup` of train) or 'holdout'. Test rows need no embargo: the holdout is not trained on.
    Returns: a Series of labels on df's index."""
    t, h = time_of(df, time_col), pd.Timedelta(horizon)
    lab = pd.Series("train", index=df.index, dtype=object)
    lab[t >= bounds["val_start"]] = "val"
    lab[t >= bounds["test_start"]] = "test"
    lab[t > bounds["holdout_after"]] = "holdout"
    lab[(lab == "train") & (t + h >= bounds["val_start"])] = "embargo"
    lab[(lab == "val") & (t + h >= bounds["test_start"])] = "embargo"
    lab[(lab == "train") & (t < bounds["start"] + pd.Timedelta(warmup))] = "warmup"
    return lab


def assignment_table(assignments):
    """{horizon: labels} -> rows per label per horizon (the embargo cost at a glance)."""
    order = [*SPLITS, "embargo", "warmup", "holdout"]
    return (pd.DataFrame({h: lab.value_counts() for h, lab in assignments.items()})
            .reindex(order).fillna(0).astype(int).T)


def split_summary(targets, assignments):
    """Labelled (non-blank) rows per target column per split, with its horizon's labels.
    holdout = rows of the holdout whose target is already known (short horizons)."""
    rows = []
    for col in [c for c in targets.columns if "_fwd_" in c]:
        lab = assignments[col.rpartition("_fwd_")[2]]
        known = targets[col].notna()
        rows.append({"column": col, **{s: int((known & (lab == s)).sum())
                                       for s in (*SPLITS, "holdout")},
                     "pending": int((~known & (lab == "holdout")).sum())})
    return pd.DataFrame(rows)


def assert_no_leakage(df, labels, bounds, horizon, time_col=TIME_COL):
    """Hard checks for ONE horizon's labels: splits in time order, every train label window
    ends before val starts and every val one before test starts, no holdout row in a split.
    Raises AssertionError; returns None."""
    t, h = time_of(df, time_col), pd.Timedelta(horizon)
    part = {s: t[labels == s] for s in SPLITS}
    assert part["train"].max() < part["val"].min() <= part["val"].max() < part["test"].min(), \
        f"{horizon}: split time ranges overlap"
    assert (part["train"] + h < bounds["val_start"]).all(), \
        f"{horizon}: a train label window reaches into val"
    assert (part["val"] + h < bounds["test_start"]).all(), \
        f"{horizon}: a val label window reaches into test"
    assert not (labels.isin(SPLITS) & (t > bounds["holdout_after"])).any(), \
        f"{horizon}: holdout rows inside a split"
    print(f" no-leakage OK {horizon:>4}: train < val < test, every label window inside its split")


# --- train-fit column selection (no reshaping: the features are already scaled) ------ #

FLAG_SUFFIX = "_was_missing"
_ROLLED = re.compile(r"_{1,2}(rolled|trail)_[^_]+$")


def column_priority(column, priority):
    """A column's priority, looked up by its BASE name: a rolled copy (x_rolled_24h,
    x__rolled_24h, x_trail_7d) takes its base's weight. A _was_missing flag gets 0 — it is
    auxiliary, so on a near-duplicate pair the real column is kept."""
    if column.endswith(FLAG_SUFFIX):
        return 0.0
    if column in priority:
        return float(priority[column])
    return float(priority.get(_ROLLED.sub("", column), 0.0))


def fit_column_selection(train_df, columns, near_constant_frac=NEAR_CONSTANT_FRAC, max_nan=0.5,
                         max_corr=0.95, priority=None):
    """Which feature columns to keep, decided on TRAIN rows only:
    drop a column blank in more than max_nan of train, one whose most common value covers
    more than near_constant_frac, and one of each pair with |Spearman| > max_corr — the
    higher priority (column_priority: by base name; flags last), then fewer blanks, then
    list order, is kept.
    Returns: {kept, dropped, reasons}."""
    columns = list(columns)
    reasons, candidates = {}, []
    for c in columns:
        s = train_df[c].astype(float)
        blank = float(s.isna().mean())
        if blank > max_nan:
            reasons[c] = f"blank in {blank:.0%} of train"
            continue
        top = float(s.value_counts(normalize=True).iloc[0]) if s.notna().any() else 1.0
        if top > near_constant_frac:
            reasons[c] = f"near-constant ({top:.1%} one value)"
            continue
        candidates.append(c)
    position, priority = {c: i for i, c in enumerate(columns)}, priority or {}
    order = sorted(candidates, key=lambda c: (-column_priority(c, priority),
                                              train_df[c].isna().mean(), position[c]))
    corr = train_df[order].astype(float).corr(method="spearman").abs()
    kept = []
    for c in order:
        twin = next((k for k in kept if corr.loc[c, k] > max_corr), None)
        if twin is not None:
            reasons[c] = f"|spearman| {corr.loc[c, twin]:.3f} with {twin}"
            continue
        kept.append(c)
    kept = [c for c in columns if c in set(kept)]
    print(f" column selection on {len(train_df)} train rows: kept {len(kept)} of {len(columns)}")
    return {"kept": kept, "dropped": [c for c in columns if c in reasons],
            "reasons": reasons}


# --- feature preparation: ONE entry point for the split notebook and for training ------- #

def feature_columns(df, non_features):
    """Numeric columns of DF_model that are model inputs (not the time key, weight, ids)."""
    return [c for c in df.columns if c not in set(non_features)
            and pd.api.types.is_numeric_dtype(df[c])]


def resolve_weighted(df, bases):
    """Columns filled with the user-weighted mean: each listed base and its rolled copies
    (ltv_utilization -> ltv_utilization_rolled_24h / _3d / …). A base with no column is
    printed, never skipped silently."""
    out, missing = [], []
    for b in bases:
        found = [c for c in df.columns if c == b or _ROLLED.sub("", c) == b]
        (out.extend(found) if found else missing.append(b))
    if missing:
        print(f" ⚠️ weighted-fill base(s) not in DF_model, filled unweighted if present: {missing}")
    return list(dict.fromkeys(out))


def prepare_features(df_model, meta, time_col=TIME_COL):
    """DF_model -> the features a model sees: the trailing-mean fill (meta['fill']), then,
    once the split has fitted it, the column selection (meta['selection']). The split
    notebook and training both call this, so their features cannot drift apart.
    meta: needs 'fill' {window, weight, weighted (bases)} and 'non_features'; 'selection'
    is optional (absent while the split is still being fitted).
    Returns: a copy — time_col + feature columns (+ _was_missing flags)."""
    import model_dataset as mds                     # lazy: model_dataset needs the domain package
    fill = meta["fill"]
    cols = feature_columns(df_model, meta["non_features"])
    out = mds.fill_trailing_mean(df_model, cols, window=fill["window"], weight=fill["weight"],
                                 weighted=resolve_weighted(df_model, fill["weighted"]),
                                 time_col=time_col, flag_suffix=FLAG_SUFFIX)
    if meta.get("selection"):
        missing = [c for c in meta["selection"]["kept"] if c not in out.columns]
        if missing:
            raise KeyError(f"kept column(s) {missing} not in DF_model — not the frame the split was fitted on")
        return out[[time_col, *meta["selection"]["kept"]]].copy()
    return out


def frame_fingerprint(df, time_col=TIME_COL):
    """rows, columns, first / last stamp of an input — stored so training can check it is
    reading the same frames the split was fitted on."""
    t = time_of(df, time_col)
    return {"rows": len(df), "cols": df.shape[1], "first": t.min(), "last": t.max(),
            "columns": list(df.columns)}


# --- binary stress labels (built at training time, never stored in DF_model) ------------ #

def stress_name(column):
    """'liquidation_debt_covered_value_usd_fwd_3d' -> 'y_stress_liquidation_debt_covered_value_usd_3d'."""
    base, _, label = column.rpartition("_fwd_")
    return f"y_stress_{base}_{label}"


def stress_thresholds(targets, spec, assignments, quantile=STRESS_QUANTILE):
    """Train-only threshold per forward column of each spec base, each on the TRAIN rows of
    its own horizon (assignments: {horizon label: assign_split labels}).
    spec: {base: 'high' | 'low'} — stress is above the train p<quantile> ('high') or below
    the train p<1 - quantile> ('low'). Any monotone scale gives the same labels.
    Returns: {column: {'side', 'quantile', 'threshold'}}."""
    out = {}
    for base, side in spec.items():
        if side not in ("high", "low"):
            raise ValueError(f"{base}: side must be 'high' or 'low', got {side!r}")
        cols = [c for c in targets.columns if c.startswith(f"{base}_fwd_")]
        if not cols:
            raise KeyError(f"no forward column of '{base}' in the target frame")
        q = quantile if side == "high" else 1 - quantile
        for c in cols:
            train = assignments[c.rpartition("_fwd_")[2]] == "train"
            out[c] = {"side": side, "quantile": q,
                      "threshold": float(targets.loc[train, c].astype(float).quantile(q))}
    return out


def binarize_stress(targets, thresholds, time_col=TIME_COL):
    """time_col + one y_stress_* column per threshold: 1.0 stress, 0.0 not, NaN where the
    target is blank (pending or unobserved)."""
    out = targets[[time_col]].copy()
    for col, th in thresholds.items():
        x = targets[col].astype(float)
        hit = x > th["threshold"] if th["side"] == "high" else x < th["threshold"]
        out[stress_name(col)] = hit.astype(float).where(x.notna())
    return out


def stress_balance(stress, assignments):
    """Positive rate per binary label per split (its own horizon's labels)."""
    rows = []
    for col in [c for c in stress.columns if c.startswith("y_stress_")]:
        lab = assignments[col.rpartition("_")[2]]
        rows.append({"label": col, **{s: round(float(stress.loc[lab == s, col].mean()), 3)
                                      for s in SPLITS}})
    return pd.DataFrame(rows)


# --- walk-forward folds inside train ------------------------------------------------- #

def walk_forward_folds(stamps, n_splits=3, horizon="24h"):
    """Expanding walk-forward folds by TIME over the train rows.
    The train span is cut into n_splits + 1 equal blocks; fold k tests on block k + 1 and
    trains on every earlier row whose label window (t, t + h] ends before that block.
    stamps: the train rows' stamps, in row order. Returns: [(train_pos, test_pos), ...]."""
    t = pd.Series(pd.DatetimeIndex(stamps)).reset_index(drop=True)
    h, first, last = pd.Timedelta(horizon), t.min(), t.max()
    edges = [first + (last - first) * k / (n_splits + 1) for k in range(n_splits + 2)]
    folds = []
    for k in range(1, n_splits + 1):
        lo, hi = edges[k], edges[k + 1]
        test = (t >= lo) & ((t < hi) if k < n_splits else (t <= hi))
        folds.append((np.flatnonzero(t + h < lo), np.flatnonzero(test)))
    return folds


def fold_summary(stamps, folds, y=None):
    """Per fold: train rows and last stamp, test period and rows, labelled test rows, and
    positives when y is a binary label (same row order as stamps)."""
    t = pd.Series(pd.DatetimeIndex(stamps)).reset_index(drop=True)
    y = None if y is None else pd.Series(np.asarray(y, dtype=float))
    rows = []
    for k, (tr, te) in enumerate(folds, 1):
        row = {"fold": k, "train_rows": len(tr), "train_last": t[tr].max(),
               "test_first": t[te].min(), "test_last": t[te].max(), "test_rows": len(te)}
        if y is not None:
            row["test_labelled"] = int(y[te].notna().sum())
            if set(y.dropna().unique()) <= {0.0, 1.0}:
                row["test_positives"] = int(y[te].sum())
        rows.append(row)
    return pd.DataFrame(rows)


# --- persistence ---------------------------------------------------------------------- #

def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return _jsonable(obj.to_dict(orient="records"))
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_split_meta(meta, folder, name="split_meta.json"):
    """Write the split definition as JSON into folder (created). Returns the path."""
    os.makedirs(str(folder), exist_ok=True)
    path = os.path.join(str(folder), name)
    with open(path, "w") as fh:
        json.dump(_jsonable(meta), fh, indent=1)
    print(f" wrote split definition -> {path}")
    return path


# --- train-fitted scaling, read back by training ---------------------------------------- #

def fit_scaler(train_df, columns):
    """Per-column mean/std/median from TRAIN only (std 0 -> 1 guard).
    train frame, feature_cols.
    Returns: {mean, std, median}; the medians double as null-imputation values."""
    params = {}
    for c in columns:
        s = train_df[c].astype(float)
        std = float(s.std())
        params[c] = {"mean": float(s.mean()),
                     "std": std if std > 0 else 1.0,
                     "median": float(s.median())}
    return params


def apply_scaler(df, params, columns=None, standardize=True):
    """Median-impute (train medians) and optionally standardize a copy of the frame.
    df, scaler_params, standardize -> False for trees/NB, True for KNN/SVM/MLP/linear.
    Returns: a new DataFrame."""
    cols = list(params) if columns is None else list(columns)
    out = df[cols].astype(float).copy()
    for c in cols:
        p = params[c]
        out[c] = out[c].fillna(p["median"])
        if standardize:
            out[c] = (out[c] - p["mean"]) / p["std"]
    return out
