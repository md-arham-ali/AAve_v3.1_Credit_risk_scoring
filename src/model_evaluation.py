"""Modeling stage — evaluation metrics, embargoed CV, and walk-forward tuning.

Classification / regression metric dicts (with rank-score Brier guard and clipped
USD back-transform), the moving-block bootstrap AUC CI, the embargoed walk-forward
CV scorers, and the train-only grid tuners. Consumed by model_training.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import math
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             mean_squared_error, mean_absolute_error, r2_score)

from model_config import RANDOM_STATE, N_CV_SPLITS, REG_TARGET
from model_zoo import (CLASSIFIER_SPECS, REGRESSOR_SPECS, CLASSIFIER_GRIDS,
                       REGRESSOR_GRIDS, build_model, predict_scores, _xgb_overrides)
from model_split import fit_scaler, apply_scaler, walk_forward_indices


def best_f1_threshold(y_true, scores):
    """Threshold that maximizes F1 on the given (validation) scores.

    Replaces the arbitrary fixed 0.5 cut on uncalibrated / class-weighted scores;
    tuned on val only, then applied unchanged to test.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    if len(np.unique(y)) < 2:
        return 0.5
    cands = np.unique(s)
    f1s = [f1_score(y, (s >= t).astype(int), zero_division=0) for t in cands]
    return float(cands[int(np.argmax(f1s))])


def evaluate_classification(y_true, scores, threshold=0.5, proper_probs=True):
    """{roc_auc, pr_auc, f1, brier, n_pos, n} — AUCs NaN when y has one class.

    brier is NaN for rank-normalized scores (proper_probs=False): a squared error
    against ranks is not a Brier score and must not sit next to real ones.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    single = len(np.unique(y)) < 2
    return {
        "roc_auc": float("nan") if single else float(roc_auc_score(y, s)),
        "pr_auc": float("nan") if single else float(average_precision_score(y, s)),
        "f1": float(f1_score(y, (s >= threshold).astype(int), zero_division=0)),
        "brier": float(np.mean((s - y) ** 2)) if proper_probs else float("nan"),
        "n_pos": int(y.sum()), "n": int(len(y)),
    }


def evaluate_regression(y_true_log, pred_log, clip_max_log=None):
    """{rmse, mae, r2, mae_usd} on the log1p scale.

    mae_usd back-transforms through expm1 with predictions CLIPPED to the train
    target range — an extrapolating linear model at log 21+ otherwise turns into a
    billion-dollar MAE that says nothing about typical error.
    """
    yt = np.asarray(y_true_log, dtype=float)
    yp = np.asarray(pred_log, dtype=float)
    yp_clip = np.clip(yp, 0.0, clip_max_log) if clip_max_log is not None else yp
    return {
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "r2": float(r2_score(yt, yp)),
        "mae_usd": float(np.mean(np.abs(np.expm1(yt) - np.expm1(yp_clip)))),
    }


def bootstrap_auc_ci(y_true, scores, n_boot=1000, block=7, level=0.95,
                     seed=RANDOM_STATE):
    """Moving-block bootstrap CI for ROC-AUC on a time-ordered evaluation split.

    Blocks of `block` consecutive days respect the serial correlation an iid
    bootstrap would ignore. Returns {auc, lo, hi, n_boot_valid}.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    n = len(y)
    rng = np.random.default_rng(seed)
    n_blocks = max(1, int(np.ceil(n / block)))
    aucs = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(1, n - block + 1), size=n_blocks)
        idx = np.concatenate([np.arange(st, min(st + block, n)) for st in starts])[:n]
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], s[idx]))
    alpha = (1 - level) / 2
    return {"auc": float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan"),
            "lo": float(np.quantile(aucs, alpha)) if aucs else float("nan"),
            "hi": float(np.quantile(aucs, 1 - alpha)) if aucs else float("nan"),
            "n_boot_valid": len(aucs)}


def _fold_matrices(train_t, cols, tr_idx, te_idx, scaled):
    """Refit the scaler on a fold's train slice; return the fold's X matrices."""
    params = fit_scaler(train_t.iloc[tr_idx], cols)
    X_tr = apply_scaler(train_t.iloc[tr_idx], params, cols, standardize=scaled).to_numpy()
    X_te = apply_scaler(train_t.iloc[te_idx], params, cols, standardize=scaled).to_numpy()
    return X_tr, X_te


def cross_val_auc(name, data, target="y_stress_1d", n_splits=N_CV_SPLITS, params=None):
    """Embargoed walk-forward CV ROC-AUC on TRAIN (scaler re-fit inside each fold).

    Folds whose slices have a single class are skipped (NaN). Returns
    {mean, std, folds, n_valid}.
    """
    spec = CLASSIFIER_SPECS[name]
    train_t = data["frames_t"]["train"]
    cols = data["feature_cols"]
    y_all = data["y"][target]["train"]
    aucs = []
    for tr_idx, te_idx in walk_forward_indices(len(train_t), n_splits):
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]
        if len(np.unique(y_te)) < 2 or len(np.unique(y_tr)) < 2:
            aucs.append(float("nan"))
            continue
        X_tr, X_te = _fold_matrices(train_t, cols, tr_idx, te_idx, spec["scaled"])
        ov = {**(_xgb_overrides(y_tr) if name == "xgboost" else {}), **(params or {})}
        model = build_model(name, "classifier", **ov)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, y_tr)
        aucs.append(evaluate_classification(
            y_te, predict_scores(model, X_te, X_tr))["roc_auc"])
    valid = [a for a in aucs if not math.isnan(a)]
    return {"mean": float(np.mean(valid)) if valid else float("nan"),
            "std": float(np.std(valid)) if valid else float("nan"),
            "folds": aucs, "n_valid": len(valid)}


def cross_val_rmse(name, data, target=REG_TARGET, n_splits=N_CV_SPLITS, params=None):
    """Embargoed walk-forward CV RMSE(log) on TRAIN for one registry regressor."""
    spec = REGRESSOR_SPECS[name]
    train_t = data["frames_t"]["train"]
    cols = data["feature_cols"]
    y_all = data["y"][target]["train"]
    rmses = []
    for tr_idx, te_idx in walk_forward_indices(len(train_t), n_splits):
        X_tr, X_te = _fold_matrices(train_t, cols, tr_idx, te_idx, spec["scaled"])
        model = build_model(name, "regressor", **(params or {}))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, y_all[tr_idx])
        rmses.append(float(np.sqrt(mean_squared_error(
            y_all[te_idx], model.predict(X_te)))))
    return {"mean": float(np.mean(rmses)), "std": float(np.std(rmses)),
            "folds": rmses, "n_valid": len(rmses)}


def tune_classifier(name, data, target="y_stress_1d", grid=None, n_splits=N_CV_SPLITS):
    """Pick the grid config with the best embargoed walk-forward CV AUC on TRAIN.

    Returns {best_params, best_cv, table} — the table shows every config so the
    choice is auditable. Selection never touches val or test.
    """
    grid = CLASSIFIER_GRIDS.get(name, [{}]) if grid is None else grid
    rows = []
    for params in grid:
        cv = cross_val_auc(name, data, target=target, n_splits=n_splits, params=params)
        rows.append({"params": json.dumps(params), "cv_auc": cv["mean"],
                     "cv_std": cv["std"], "n_valid": cv["n_valid"]})
    table = pd.DataFrame(rows).sort_values("cv_auc", ascending=False).reset_index(drop=True)
    best = json.loads(table.iloc[0]["params"])
    print(f" {name}: tuned over {len(grid)} configs -> {best} "
          f"(CV AUC {table.iloc[0]['cv_auc']:.3f})")
    return {"best_params": best, "best_cv": float(table.iloc[0]["cv_auc"]), "table": table}


def tune_regressor(name, data, target=REG_TARGET, grid=None, n_splits=N_CV_SPLITS):
    """Pick the grid config with the lowest walk-forward CV RMSE(log) on TRAIN."""
    grid = REGRESSOR_GRIDS.get(name, [{}]) if grid is None else grid
    rows = []
    for params in grid:
        cv = cross_val_rmse(name, data, target=target, n_splits=n_splits, params=params)
        rows.append({"params": json.dumps(params), "cv_rmse": cv["mean"],
                     "cv_std": cv["std"]})
    table = pd.DataFrame(rows).sort_values("cv_rmse").reset_index(drop=True)
    best = json.loads(table.iloc[0]["params"])
    print(f" {name}: tuned over {len(grid)} configs -> {best} "
          f"(CV RMSE {table.iloc[0]['cv_rmse']:.3f})")
    return {"best_params": best, "best_cv": float(table.iloc[0]["cv_rmse"]), "table": table}
