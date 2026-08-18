"""Modeling stage — chronological split, stress thresholds, feature transform, leakage guards."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from model_config import (TIME_COL, HORIZONS, REG_TARGET, STRESS_TARGETS,
                          STRESS_QUANTILE, SPLIT_FRACTIONS, EMBARGO, N_CV_SPLITS,
                          SKEW_LOG1P_THRESHOLD, NEAR_CONSTANT_FRAC)


def chronological_split(df, fractions=SPLIT_FRACTIONS, time_col=TIME_COL, embargo=EMBARGO):
    """Chronological train/val/test split with an embargo before each boundary.
    df, fractions, embargo (= max horizon) -> last embargo rows of train and val dropped.
    Returns: {split: frame}; test keeps every row."""

    # those dropped rows' fwd-max labels read days inside the next split — straight leakage
    out = df.sort_values(time_col, kind="stable").reset_index(drop=True)
    n = len(out)
    n_train = int(n * fractions[0])
    n_val = int(n * fractions[1])
    splits = {"train": out.iloc[:n_train - embargo].reset_index(drop=True),
              "val": out.iloc[n_train:n_train + n_val - embargo].reset_index(drop=True),
              "test": out.iloc[n_train + n_val:].reset_index(drop=True)}
    print(f" embargo: {embargo} rows dropped before the val and test boundaries")
    for name, part in splits.items():
        print(f" {name:5s} {part.shape[0]} rows  {part[time_col].iloc[0][:10]} → "
              f"{part[time_col].iloc[-1][:10]}")
    return splits


def compute_stress_thresholds(train_df, quantile=STRESS_QUANTILE, horizons=HORIZONS):
    """Train-only stress threshold PER HORIZON: the p-quantile of that horizon's fwd max.
    train frame, horizons, quantile -> one threshold each.
    Returns: {horizon: float}, keeping every target near (1-q) positives."""

    # NOTE: one shared daily threshold used to make y_stress_7d ~76% positive
    out = {}
    for h in horizons:
        out[h] = float(train_df[f"target_fwd_max_{h}d"].astype(float).quantile(quantile))
        print(f" stress threshold {h}d (train p{int(quantile * 100)} of fwd-max): "
              f"{out[h]:,.0f} USD")
    return out


def binarize_targets(splits, thresholds, horizons=HORIZONS):
    """Add y_stress_{h}d to every split using the TRAIN thresholds only.
    splits, thresholds -> 1 when the forward h-day max exceeds the train threshold.
    Returns: a positive-rate table (split x horizon) for the balance check."""
    rows = []
    for name, part in splits.items():
        row = {"split": name, "n": len(part)}
        for h in horizons:
            y = (part[f"target_fwd_max_{h}d"].astype(float) > thresholds[h]).astype(int)
            part[f"y_stress_{h}d"] = y
            row[f"pos_rate_{h}d"] = round(float(y.mean()), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def fit_feature_transform(train_df, columns, skew_threshold=SKEW_LOG1P_THRESHOLD,
                          near_constant_frac=NEAR_CONSTANT_FRAC):
    """Train-fit column transform: log1p heavy right tails, drop near-constants.
    train frame, skew_threshold, near_constant_frac.
    Returns: {log1p_cols, dropped} — the persisted transform every runner replays."""

    # the regression target was already logged; the features were not, which skewed KNN/SVM/MLP
    log1p_cols, dropped, kept = [], [], []
    for c in columns:
        s = train_df[c].astype(float)
        top_share = s.value_counts(normalize=True, dropna=True).iloc[0] if s.notna().any() else 1.0
        if top_share > near_constant_frac:
            dropped.append(c)
            continue
        kept.append(c)
        if s.min() >= 0 and float(s.skew()) > skew_threshold:
            log1p_cols.append(c)
    print(f" transform: {len(log1p_cols)} log1p columns, {len(dropped)} near-constant "
          f"dropped {dropped if dropped else ''}")
    return {"log1p_cols": log1p_cols, "dropped": dropped, "kept": kept}


def apply_feature_transform(df, transform):
    """Apply a fitted transform: drop the near-constants, log1p the flagged columns."""
    out = df.drop(columns=[c for c in transform["dropped"] if c in df.columns]).copy()
    for c in transform["log1p_cols"]:
        if c in out.columns:
            out[c] = np.log1p(out[c].astype(float).clip(lower=0))
    return out


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


def assert_no_leakage(splits, time_col=TIME_COL, embargo=EMBARGO):
    """Hard-assert split ordering, target completeness, and the label embargo.
    splits, embargo -> raises AssertionError on any violation.
    Returns: None. At least `embargo` days must separate a split's last row from the next's first."""
    tr, va, te = splits["train"], splits["val"], splits["test"]
    assert tr[time_col].max() < va[time_col].min() < va[time_col].max() < te[time_col].min(), \
        "split time ranges overlap"
    def _days(a, b):
        f = lambda s: pd.to_datetime(s.replace(" UTC", ""))
        return (f(b) - f(a)).days
    assert _days(tr[time_col].max(), va[time_col].min()) > embargo, \
        f"train→val gap <= embargo ({embargo}d): fwd-max labels leak into val"
    assert _days(va[time_col].max(), te[time_col].min()) > embargo, \
        f"val→test gap <= embargo ({embargo}d): fwd-max labels leak into test"
    for name, part in splits.items():
        for t in list(STRESS_TARGETS) + [REG_TARGET]:
            assert part[t].notna().all(), f"{name}: null target {t}"
    print(f" no-leakage checks OK: train < val < test, {embargo}d embargo at both "
          f"boundaries, targets complete")


def walk_forward_indices(n, n_splits=N_CV_SPLITS, embargo=EMBARGO):
    """Expanding-window walk-forward folds with the same label embargo inside CV.
    n_rows, n_folds, embargo -> last embargo rows of each fold's train slice dropped.
    Returns: [(train_idx, test_idx), ...] — the in-fold copy of the ERR-23 fix."""
    folds = []
    for tr_idx, te_idx in TimeSeriesSplit(n_splits=n_splits).split(np.arange(n)):
        folds.append((tr_idx[:-embargo] if embargo else tr_idx, te_idx))
    return folds


def check_fold_positives(y, folds, min_pos=5):
    """Warn when a walk-forward fold's test slice holds too few positives for AUC."""
    y = np.asarray(y)
    for k, (_, test_idx) in enumerate(folds):
        pos = int(y[test_idx].sum())
        flag = "  ⚠️ below min" if pos < min_pos else ""
        print(f" fold {k + 1}: test {len(test_idx)} rows, {pos} positives{flag}")
