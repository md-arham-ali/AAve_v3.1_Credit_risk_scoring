"""Modeling stage — evaluation metrics, embargoed walk-forward CV, feature ranking, and tuning.

Tuning only ever reads TRAIN: every config is scored on the embargoed walk-forward folds, and
val and test are never looked at to pick a config.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import difflib
import json
import math
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             mean_squared_error, mean_absolute_error, r2_score)

from model_config import RANDOM_STATE, CI_BLOCK_ROWS, REG_TARGET, STRESS_TARGETS, TIME_COL
from model_zoo import (CLASSIFIER_SPECS, REGRESSOR_SPECS, CLASSIFIER_GRIDS, REGRESSOR_GRIDS,
                       SIMPLICITY, SEED_ARG, build_model, expand_grid, predict_scores, _plain, _xgb_overrides)
from model_split import (fit_scaler, apply_scaler, time_of, walk_forward_folds, fold_summary,
                         FLAG_SUFFIX, _ROLLED)


# =========================================================================== #
# Metrics
# =========================================================================== #

def best_f1_threshold(y_true, scores):
    """Threshold maximizing F1 on the given (validation) scores.
    y_true, scores -> sweep over candidate cuts.
    Returns: float threshold. Tuned on val only, applied unchanged to test."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    if len(np.unique(y)) < 2:
        return 0.5
    cands = np.unique(s)
    f1s = [f1_score(y, (s >= t).astype(int), zero_division=0) for t in cands]
    return float(cands[int(np.argmax(f1s))])


def evaluate_classification(y_true, scores, threshold=0.5, proper_probs=True):
    """Classification metrics for one split.
    y_true, scores, threshold, proper_probs -> AUCs NaN when y has one class.
    Returns: {roc_auc, pr_auc, f1, brier, n_pos, n}; brier NaN for rank scores."""
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


def evaluate_regression(y_true, y_pred, clip_max=None, to_usd=None):
    """Regression metrics on the target's own scale: the scaled 0-100 score.
    y_true, y_pred: scores; clip_max: the train target max, applied before the USD reverse only;
    to_usd: score array -> USD array (model_scaling.target_unscaler), None for a target with no
    reverse. Returns: {rmse, mae, r2} in score points + mae_usd in real USD (NaN without to_usd)."""

    # clip matters: the reverse is exponential in the score, so one extrapolating linear
    # prediction would otherwise dominate the USD MAE. rmse/mae/r2 keep the unclipped prediction.
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    yp_clip = np.clip(yp, 0.0, clip_max) if clip_max is not None else yp
    return {
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "r2": float(r2_score(yt, yp)),
        "mae_usd": (float(np.mean(np.abs(to_usd(yt) - to_usd(yp_clip))))
                    if to_usd is not None else float("nan")),
    }


def bootstrap_auc_ci(y_true, scores, n_boot=1000, block=CI_BLOCK_ROWS, level=0.95,
                     seed=RANDOM_STATE):
    """Moving-block bootstrap CI for ROC-AUC on a time-ordered split.
    y_true, scores, block (consecutive ROWS per block, CI_BLOCK_ROWS = 7 days at 2h), n_boot.
    Returns: {auc, lo, hi, n_boot_valid}."""
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


# =========================================================================== #
# Features (train only): names, groups, ranking
# =========================================================================== #

RANK_METHODS = ("forest", "alone")


def rank_features(X, y, cols, regression=False, method="forest"):
    """Train-only feature ranking, strongest first.
    method "forest": a seeded RandomForest's importance (as before; the default).
    method "alone": each column on its own as the score, |ROC-AUC - 0.5| against the label (classifier
        targets only). No model in between, so a column cannot rank high by memorising a period.
    X, y: TRAIN rows only (the full train block, or one fold's train part); cols: X's column names.
    Returns: cols ordered strongest first."""
    if method not in RANK_METHODS:
        raise ValueError(f"method={method!r}: use one of {RANK_METHODS}")
    if method == "alone":
        if regression:
            raise ValueError("method='alone' ranks against a binary label: use 'forest' for a regressor")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if len(np.unique(y)) < 2:
            return list(cols)
        # a column blank on every row here (an asset not listed yet) has no ranking power: strength 0
        strength = [0.0 if np.isnan(X[:, j]).all() else
                    abs(float(roc_auc_score(y, np.nan_to_num(X[:, j], nan=np.nanmedian(X[:, j])))) - 0.5)
                    for j in range(X.shape[1])]
        return [cols[i] for i in np.argsort(strength, kind="stable")[::-1]]
    rf = (RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1) if regression
          else RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                      random_state=RANDOM_STATE, n_jobs=-1))
    rf.fit(X, y)
    return [cols[i] for i in np.argsort(rf.feature_importances_)[::-1]]


def check_features(cols, known):
    """Stop the run on a bad feature list before any fitting.
    A typo would otherwise surface as a pandas KeyError deep inside a CV fold, and an empty list
    would quietly train on every feature (the runners read `features or all`).
    cols: names to fit on; known: the split's feature_cols. Returns: cols as a plain list, order kept."""
    if isinstance(cols, str):
        raise ValueError(f"features= takes a list of names, not one string: [{cols!r}]")
    cols = list(cols)
    nested = [c for c in cols if isinstance(c, (list, tuple, set))]
    if nested:
        raise ValueError("features= got a list inside the list: add groups with + (a + b), not [a, b]")
    if not cols:
        raise ValueError("features= is empty: pass None for every feature")
    dupes = sorted({c for c in cols if cols.count(c) > 1})
    if dupes:
        raise ValueError(f"features= names {dupes} more than once")
    bad = [c for c in cols if c not in set(known)]
    if bad:
        hints = {b: difflib.get_close_matches(str(b), [str(k) for k in known], n=2) for b in bad}
        raise ValueError("unknown feature(s) "
                         + ", ".join(f"{b!r}" + (f" (did you mean {h}?)" if h else "")
                                     for b, h in hints.items())
                         + ". data['feature_cols'] lists the names.")
    return cols


def _window_of(col):
    """'x_rolled_12h' -> '12h', 'x_was_missing' -> 'flag', a plain column -> 'point'."""
    if col.endswith(FLAG_SUFFIX):
        return "flag"
    m = _ROLLED.search(col)
    return m.group(0).rpartition("_")[2] if m else "point"


def _signal_of(col):
    """The base signal a column is a copy of: the name without its window or flag suffix."""
    if col.endswith(FLAG_SUFFIX):
        return col[: -len(FLAG_SUFFIX)]
    return _ROLLED.sub("", col)


def feature_groups(cols, by="signal"):
    """Columns grouped by base signal ('leverage_indicator' + its rolled copies) or by window
    ('point', '12h', '3d', '7d', 'flag'). The rolled copies are strongly correlated, so a whole
    group is the honest unit to drop or permute: one copy alone hides behind the others.
    Returns: {group: [cols]} in column order (windows shortest first)."""
    if by not in ("signal", "window"):
        raise ValueError("by= is 'signal' or 'window'")
    key = _signal_of if by == "signal" else _window_of
    groups = {}
    for c in cols:
        groups.setdefault(key(c), []).append(c)
    if by == "window":
        def order(w):
            return (0, 0) if w == "point" else (2, 0) if w == "flag" else (1, pd.Timedelta(w).total_seconds())
        groups = {w: groups[w] for w in sorted(groups, key=order)}
    print(f" {len(groups)} {by} group(s): " + ", ".join(f"{g} {len(v)}" for g, v in groups.items()))
    return groups


def without(cols, drop):
    """cols minus the names in `drop`, order kept. Every name in drop must be in cols, so a typo
    stops here instead of silently keeping the column. drop=[] returns a copy of cols."""
    drop = list(drop)
    nested = [d for d in drop if isinstance(d, (list, tuple, set))]
    if nested:
        raise ValueError("drop= got a list inside the list: join groups with + (a + b), not [a, b]")
    bad = [d for d in drop if d not in set(cols)]
    if bad:
        hints = {b: difflib.get_close_matches(str(b), [str(c) for c in cols], n=2) for b in bad}
        raise ValueError("cannot drop " + ", ".join(f"{b!r}" + (f" (did you mean {h}?)" if h else "")
                                                    for b, h in hints.items()) + ": not in the list")
    gone = set(drop)
    return [c for c in cols if c not in gone]


def ablation_sets(cols, by="window", mode="drop"):
    """Named feature sets for cv_feature_sets: "all", then one set per group.
    mode="drop": every column except that group ('- 12h'); mode="only": that group alone ('only 12h').
    Empty sets are skipped. Returns: {label: [cols]}."""
    if mode not in ("drop", "only"):
        raise ValueError("mode= is 'drop' or 'only'")
    sets = {"all": list(cols)}
    for g, members in feature_groups(cols, by=by).items():
        chosen = without(cols, members) if mode == "drop" else list(members)
        if chosen:
            sets[f"- {g}" if mode == "drop" else f"only {g}"] = chosen
    return sets


# =========================================================================== #
# Walk-forward CV
# =========================================================================== #

def _cv_folds(data, target, n_splits=None):
    """The split's walk-forward folds on one target's train rows: cut by TIME, each fold's train
    part ends one of THAT target's horizons before its test part (model_split.walk_forward_folds).
    n_splits: defaults to the fold count the split fixed (data["n_folds"]).
    Returns: [(train_pos, test_pos), ...] over data["frames_t"][target]["train"]."""
    return walk_forward_folds(time_of(data["frames_t"][target]["train"]),
                              n_splits=n_splits or data["n_folds"],
                              horizon=data["label_history"][target]["horizon"])


def _fold_matrices(train_t, cols, tr_idx, te_idx, scaled):
    """Refit the scaler on a fold's train slice; return the fold's X matrices."""
    # scaler is re-fit inside the fold, never on the full train block
    params = fit_scaler(train_t.iloc[tr_idx], cols)
    X_tr = apply_scaler(train_t.iloc[tr_idx], params, cols, standardize=scaled).to_numpy()
    X_te = apply_scaler(train_t.iloc[te_idx], params, cols, standardize=scaled).to_numpy()
    return X_tr, X_te


def _fold_cols(train_t, cols, tr_idx, y_tr, top_n, regression, rank="forest"):
    """The columns one fold fits on: all of `cols`, or the fold's own top-N (rank: see rank_features).
    The ranking reads only the fold's train part, so the fold's test rows never pick features."""
    if not top_n:
        return cols
    X_rank, _ = _fold_matrices(train_t, cols, tr_idx, tr_idx[:1], scaled=False)
    return rank_features(X_rank, y_tr, cols, regression=regression, method=rank)[:top_n]


def cv_summary(folds, guards=None):
    """Fold scores -> {mean, std, se, folds, n_valid}, plus <guard>_mean for each guard metric.
    NaN folds (single-class) are skipped. std is the plain spread (ddof=0, as before);
    se = sample std (ddof=1) / sqrt(n_valid), NaN with fewer than 2 valid folds."""
    valid = [f for f in folds if not math.isnan(f)]
    n = len(valid)
    out = {"mean": float(np.mean(valid)) if n else float("nan"),
           "std": float(np.std(valid)) if n else float("nan"),
           "se": float(np.std(valid, ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
           "folds": [float(f) for f in folds], "n_valid": n}
    for name, values in (guards or {}).items():
        v = [x for x in values if not math.isnan(x)]
        out[f"{name}_mean"] = float(np.mean(v)) if v else float("nan")
    return out


def fold_auc(y_true, scores):
    """One fold's classifier scores -> {"main": ROC-AUC, "pr_auc": PR-AUC}, NaN when single-class."""
    m = evaluate_classification(y_true, scores, proper_probs=False)
    return {"main": m["roc_auc"], "pr_auc": m["pr_auc"]}


def fold_table(data, target=STRESS_TARGETS[0], n_splits=None):
    """The walk-forward folds every config is scored on: each fold's train rows, its test period,
    and the positive rate on both sides. The folds are different market periods (fold 1 learns
    from the fewest rows), which is why configs are compared fold by fold, not by the mean alone.
    Returns: one row per fold."""
    y = np.asarray(data["y"][target]["train"], dtype=float)
    folds = _cv_folds(data, target, n_splits)
    table = fold_summary(time_of(data["frames_t"][target]["train"]), folds, y)
    table["train_rate"] = [round(float(y[tr].mean()), 3) for tr, _ in folds]
    table["test_rate"] = [round(float(y[te].mean()), 3) for _, te in folds]
    return table


def _fold_cols_of(table):
    return [c for c in table.columns if re.fullmatch(r"fold_\d+", str(c))]


def paired_folds(table, ref=None, horizon=None):
    """Fold-by-fold comparison inside any CV table with fold_1..k columns (tuning, cv_top_n,
    cv_feature_sets, seed_spread). Every row is scored on the same folds, so the difference to a
    reference row cancels the periods: the plain SE of a CV mean is mostly the gap between an easy
    and a hard period, the paired SE is the gap between the configs.
    ref: index label of the reference row; default the 'chosen' row, else the first row.
    horizon: which horizon of a multi-horizon tuning table (default '24h').
    Returns: a copy with d_vs_ref (mean paired difference, + = better), d_se (its SE),
    wins (folds better than ref) and n_pair. For an error metric (cv_rmse) lower counts as better."""
    t = table.copy()
    if "horizon" in t.columns and (horizon is not None or t["horizon"].nunique() > 1):
        h = horizon or "24h"
        t = t[t["horizon"] == h].copy()
        if t.empty:
            raise ValueError(f"no {h!r} rows in this table: {sorted(table['horizon'].unique())}")
    fold_cols = _fold_cols_of(t)
    if not fold_cols or t.empty:
        raise ValueError("paired_folds needs a table with rows and fold_1..k columns")
    if ref is None:
        chosen = t.index[t["chosen"].astype(bool)] if "chosen" in t.columns else []
        ref = chosen[0] if len(chosen) else t.index[0]
    sign = -1.0 if "cv_rmse" in t.columns else 1.0
    F = t[fold_cols].to_numpy(dtype=float)
    D = sign * (F - t.loc[ref, fold_cols].to_numpy(dtype=float))
    d_mean, d_se, wins, n_pair = [], [], [], []
    for row in D:
        d = row[~np.isnan(row)]
        d_mean.append(float(d.mean()) if len(d) else float("nan"))
        d_se.append(float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan"))
        wins.append(int((d > 0).sum()))
        n_pair.append(len(d))
    t["d_vs_ref"], t["d_se"], t["wins"], t["n_pair"] = d_mean, d_se, wins, n_pair
    return t


def vs_reference(tuning, reference, min_gain=0.01, min_wins=2):
    """Reads a round-two stage: every row of the tuning table against the CHALLENGED option, not against
    the row the run's select rule chose and not against the best row (the table is sorted best first).
    reference: the first grid row, model_zoo.grid_around(...)["reference"]. Its row is found by value: every
    parameter column that varies must equal the reference's value (blank where the reference names none).
    A row `passes` when its mean paired gain is at least min_gain (0.01) and it wins in at least min_wins folds
    (the RF showcase rule, Descisions.md 2026-11-11); the reference never passes. A smaller gain is inside the
    noise. Sequence nets move 0.01-0.014 with the seed alone (cells lstm-01 / tfm-01): their stages use 0.02.
    Returns: the table with is_ref, d_vs_ref, d_se, wins, n_pair and passes added."""
    cols = list(tuning.columns)
    first_cv = next(i for i, c in enumerate(cols) if c.startswith("cv_"))
    params = [c for c in cols[:first_cv] if c not in ("horizon", "chosen")]      # [horizon] chosen | varying parameters | cv_* ...
    def same(row):
        for k in params:
            want, got = _cell(reference.get(k)), row[k]
            if want is None or (isinstance(want, float) and math.isnan(want)):
                if not pd.isna(got):
                    return False
            elif pd.isna(got) or got != want:
                return False
        return True
    hits = [i for i in tuning.index if same(tuning.loc[i])]
    if not hits:
        raise ValueError(f"no row of the table equals the reference {reference}: columns {params}")
    out = paired_folds(tuning, ref=hits[0])
    out.insert(1, "is_ref", [i == hits[0] for i in out.index])
    out["passes"] = (out["d_vs_ref"] >= min_gain) & (out["wins"] >= min_wins) & ~out["is_ref"]
    n = int(out["passes"].sum())
    print(f" {n} of {len(out) - 1} option(s) pass (paired gain >= {min_gain}, better in >= {min_wins} folds)"
          + (": " + ", ".join(str(i) for i in out.index[out["passes"]]) if n else ""))
    return out


def cross_val_auc(name, data, target=STRESS_TARGETS[0], n_splits=None, params=None, cols=None,
                  top_n=None, rank="forest"):
    """Embargoed walk-forward CV ROC-AUC on TRAIN; scaler re-fit inside each fold.
    Folds from _cv_folds; single-class folds skipped as NaN. PR-AUC rides along as the guard.
    cols: feature subset to fold on (e.g. a linear model's top-N) — defaults to every feature.
    top_n: rank features inside each fold on its train part and keep the top N of `cols`.
    rank: how a fold ranks them, "forest" or "alone" (rank_features).
    Returns: cv_summary shape + pr_auc_mean."""
    spec = CLASSIFIER_SPECS[name]
    train_t = data["frames_t"][target]["train"]
    cols = cols or data["feature_cols"]
    y_all = data["y"][target]["train"]
    aucs, prs = [], []
    for tr_idx, te_idx in _cv_folds(data, target, n_splits):
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]
        if len(np.unique(y_te)) < 2 or len(np.unique(y_tr)) < 2:
            aucs.append(float("nan"))
            prs.append(float("nan"))
            continue
        use = _fold_cols(train_t, cols, tr_idx, y_tr, top_n, regression=False, rank=rank)
        X_tr, X_te = _fold_matrices(train_t, use, tr_idx, te_idx, spec["scaled"])
        ov = {**(_xgb_overrides(y_tr) if name == "xgboost" else {}), **(params or {})}
        model = build_model(name, "classifier", **ov)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, y_tr)
        m = evaluate_classification(y_te, predict_scores(model, X_te, X_tr))
        aucs.append(m["roc_auc"])
        prs.append(m["pr_auc"])
    return cv_summary(aucs, {"pr_auc": prs})


def cross_val_rmse(name, data, target=REG_TARGET, n_splits=None, params=None, cols=None,
                   top_n=None):
    """Embargoed walk-forward CV RMSE (score points) on TRAIN for one registry regressor.
    MAE rides along as the guard: RMSE favours the squared-error loss, MAE does not.
    cols / top_n: as in cross_val_auc. Returns: cv_summary shape + mae_mean."""
    # same embargoed folds as cross_val_auc, error metric instead
    spec = REGRESSOR_SPECS[name]
    train_t = data["frames_t"][target]["train"]
    cols = cols or data["feature_cols"]
    y_all = data["y"][target]["train"]
    rmses, maes = [], []
    for tr_idx, te_idx in _cv_folds(data, target, n_splits):
        use = _fold_cols(train_t, cols, tr_idx, y_all[tr_idx], top_n, regression=True)
        X_tr, X_te = _fold_matrices(train_t, use, tr_idx, te_idx, spec["scaled"])
        model = build_model(name, "regressor", **(params or {}))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, y_all[tr_idx])
        pred = model.predict(X_te)
        rmses.append(float(np.sqrt(mean_squared_error(y_all[te_idx], pred))))
        maes.append(float(mean_absolute_error(y_all[te_idx], pred)))
    return cv_summary(rmses, {"mae": maes})


def cross_val_custom(data, target, fold_fn, n_splits=None):
    """Walk-forward CV for the runners outside the registry (LSTM, Transformer, Cox, IsolationForest):
    the same embargoed folds as cross_val_auc; the model's own fit and score live in fold_fn.
    fold_fn(train_pos, test_pos) -> {"main": float, <guard>: float, ...}, NaN when a fold can't be scored.
    Returns: cv_summary shape."""
    mains, guards = [], {}
    for tr_idx, te_idx in _cv_folds(data, target, n_splits):
        scores = dict(fold_fn(tr_idx, te_idx))
        mains.append(scores.pop("main"))
        for k, v in scores.items():
            guards.setdefault(k, []).append(v)
    return cv_summary(mains, guards)


def cv_top_n(name, data, ns, kind="classifier", target=None, fixed=None, rank="forest"):
    """Feature count checked on CV. For each N, every fold ranks the features on ITS OWN train part
    and fits on its top N, so no N is picked with rows the fold is scored on.
    Pick N from this table, then pass top_n_features(data, target, n=N) as the runner's features=.
    ns: e.g. (10, 20, 30); fixed: the model's fixed values for this check.
    rank: "forest" (importance, the default) or "alone" (each column's own AUC; classifiers only).
    Returns: one row per N — CV metric, std, se, guard and folds."""
    target = target or (STRESS_TARGETS[0] if kind == "classifier" else REG_TARGET)
    rows = []
    for n in ns:
        cv = (cross_val_auc(name, data, target, params=fixed, top_n=n, rank=rank) if kind == "classifier"
              else cross_val_rmse(name, data, target, params=fixed, top_n=n))
        metric, guard = ("auc", "pr_auc") if kind == "classifier" else ("rmse", "mae")
        rows.append({"top_n": n, f"cv_{metric}": cv["mean"], "cv_std": cv["std"],
                     "cv_se": cv["se"], f"cv_{guard}": cv[f"{guard}_mean"],
                     **{f"fold_{i}": f for i, f in enumerate(cv["folds"], 1)}})
    table = pd.DataFrame(rows)
    print(f" {name} ({kind}) [{_horizon(target)}] feature count on CV, ranked by {rank!r} inside each fold:")
    print(table.round(4).to_string(index=False))
    return table


# =========================================================================== #
# CV evidence for one classifier: feature sets, permutation importance, tree count, seeds
# =========================================================================== #

def _fit_fold(name, train_t, y_all, cols, tr_idx, te_idx, params):
    """One fold fitted the way cross_val_auc fits it (scaler re-fit on the fold's train part,
    xgboost's weight from the fold's own labels). cross_val_auc keeps its own copy: the official
    CV numbers come from it and stay untouched.
    Returns: (model, X_tr, X_te), or None when either part is single-class (a NaN fold)."""
    y_tr, y_te = y_all[tr_idx], y_all[te_idx]
    if len(np.unique(y_te)) < 2 or len(np.unique(y_tr)) < 2:
        return None
    X_tr, X_te = _fold_matrices(train_t, cols, tr_idx, te_idx, CLASSIFIER_SPECS[name]["scaled"])
    ov = {**(_xgb_overrides(y_tr) if name == "xgboost" else {}), **(params or {})}
    model = build_model(name, "classifier", **ov)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_tr, y_tr)
    return model, X_tr, X_te


def _cols_or_all(cols, data):
    return check_features(cols, data["feature_cols"]) if cols is not None else list(data["feature_cols"])


OOF_COLUMNS = [TIME_COL, "horizon", "fold", "y_true", "score"]


def oof_frame(data, target, parts):
    """Out-of-fold rows of one horizon in the shape every model shares.
    parts: [(fold number, test positions in the train rows, scores), ...].
    Returns: DataFrame time_bucket, horizon, fold, y_true, score."""
    times, y = np.asarray(data["times"][target]["train"]), data["y"][target]["train"]
    rows = [pd.DataFrame({TIME_COL: times[te], "horizon": _horizon(target), "fold": k,
                          "y_true": np.asarray(y[te], dtype=float), "score": np.asarray(sc, dtype=float)})
            for k, te, sc in parts]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=OOF_COLUMNS)


def oof_data(data, oof):
    """The bundle an out-of-fold pass runs on: oof=True keeps the split's own folds, an int asks for
    that many (more folds = more rows for a blend to learn from, each fold a shorter period)."""
    return data if oof is True else {**data, "n_folds": int(oof)}


def cv_oof(name, data, *, target=STRESS_TARGETS[0], params=None, cols=None, n_splits=None):
    """Out-of-fold scores of one registry classifier: every walk-forward fold fitted the way
    cross_val_auc fits it and scored on its own test part. Train rows only, and every score comes
    from a model that never saw that row: this is what an ensemble may learn its weights from.
    A single-class fold is left out (cross_val_auc scores it NaN).
    Returns: oof_frame shape."""
    train_t, y_all = data["frames_t"][target]["train"], data["y"][target]["train"]
    cols = _cols_or_all(cols, data)
    parts = []
    for k, (tr_idx, te_idx) in enumerate(_cv_folds(data, target, n_splits), start=1):
        fit = _fit_fold(name, train_t, y_all, cols, tr_idx, te_idx, params)
        if fit is not None:
            model, X_tr, X_te = fit
            parts.append((k, te_idx, predict_scores(model, X_te, X_tr)))
    return oof_frame(data, target, parts)


def cv_feature_sets(name, data, sets, *, target=STRESS_TARGETS[0], fixed=None, n_splits=None):
    """The same config on several feature sets, on the tuning folds (train only).
    sets: {label: [cols] or None for every feature}, e.g. ablation_sets(cols, by="window").
    Every set is checked before anything is fitted.
    Returns: one row per set — n_features, cv_auc, cv_std, cv_se, cv_pr_auc, folds, and the paired
    difference to the FIRST set (d_vs_ref, d_se, wins)."""
    checked = {label: _cols_or_all(cols, data) for label, cols in sets.items()}
    rows = []
    for label, cols in checked.items():
        cv = cross_val_auc(name, data, target, n_splits, params=fixed, cols=cols)
        rows.append({"set": label, "n_features": len(cols), "cv_auc": cv["mean"], "cv_std": cv["std"],
                     "cv_se": cv["se"], "cv_pr_auc": cv["pr_auc_mean"],
                     **{f"fold_{i}": f for i, f in enumerate(cv["folds"], 1)}})
    table = paired_folds(pd.DataFrame(rows), ref=0)
    print(f" {name} [{_horizon(target)}]: {len(rows)} feature set(s) on CV, paired against {rows[0]['set']!r}")
    return table


def cv_permutation_importance(name, data, *, target=STRESS_TARGETS[0], fixed=None, cols=None,
                              groups=None, n_repeats=5, seed=RANDOM_STATE, n_splits=None):
    """What the model leans on where it has to generalise. Each walk-forward fold is fitted once on
    its train part; then one group of columns at a time is shuffled on that fold's TEST part (rows
    permuted together) and the AUC drop is recorded. No refit, no val, no test. A signal and its
    rolled copies are shuffled as one group, or the copies would cover for each other.
    groups: {label: [cols]} (feature_groups), None = one group per column; cut to cols.
    Returns: one row per group, biggest drop first — n_cols, drop_mean, drop_std (across folds),
    folds_pos (folds where shuffling hurt) and the drop per fold. Below 0 = the model did better on
    unseen periods with that group scrambled."""
    cols = _cols_or_all(cols, data)
    pos = {c: i for i, c in enumerate(cols)}
    groups = groups or {c: [c] for c in cols}
    groups = {g: [pos[c] for c in members if c in pos] for g, members in groups.items()}
    groups = {g: idx for g, idx in groups.items() if idx}
    train_t = data["frames_t"][target]["train"]
    y_all = data["y"][target]["train"]
    drops, base = {g: [] for g in groups}, []
    for k, (tr_idx, te_idx) in enumerate(_cv_folds(data, target, n_splits)):
        fit = _fit_fold(name, train_t, y_all, cols, tr_idx, te_idx, fixed)
        if fit is None:
            base.append(float("nan"))
            for g in groups:
                drops[g].append(float("nan"))
            continue
        model, X_tr, X_te = fit
        y_te = y_all[te_idx]
        auc0 = float(roc_auc_score(y_te, predict_scores(model, X_te, X_tr)))
        base.append(auc0)
        for gi, (g, idx) in enumerate(groups.items()):
            reps = []
            for r in range(n_repeats):
                p = np.random.default_rng([seed, k, gi, r]).permutation(len(te_idx))
                X_p = X_te.copy()
                X_p[:, idx] = X_te[p][:, idx]
                reps.append(auc0 - float(roc_auc_score(y_te, predict_scores(model, X_p, X_tr))))
            drops[g].append(float(np.mean(reps)))
    rows = []
    for g, idx in groups.items():
        d = np.asarray(drops[g], dtype=float)
        ok = d[~np.isnan(d)]
        rows.append({"group": g, "n_cols": len(idx),
                     "drop_mean": float(ok.mean()) if len(ok) else float("nan"),
                     "drop_std": float(ok.std()) if len(ok) else float("nan"),
                     "folds_pos": int((ok > 0).sum()),
                     **{f"fold_{i}": v for i, v in enumerate(drops[g], 1)}})
    table = pd.DataFrame(rows).sort_values("drop_mean", ascending=False).reset_index(drop=True)
    print(f" {name} [{_horizon(target)}] permutation importance on the CV test parts: {len(groups)} group(s), "
          f"{n_repeats} shuffle(s) each, base AUC per fold {[round(b, 3) for b in base]}")
    return table


def cv_forest_curve(name, data, *, target=STRESS_TARGETS[0], fixed=None, cols=None, step=25,
                    n_splits=None):
    """CV AUC against the number of trees, train only. Each fold fits ONE forest with all the trees
    asked for (fixed n_estimators), then scores the running mean of its first k trees on the fold's
    test part: a forest's probability IS the mean of its trees', so this is the k-tree forest.
    The tree count is read here, where the curve stops rising; OOB never picks it (2h rows have their
    neighbours in the bag). Bagged forests only (random_forest, extra_trees).
    Returns: one row per k — cv_auc (mean over folds), cv_std and the AUC per fold."""
    cols = _cols_or_all(cols, data)
    train_t = data["frames_t"][target]["train"]
    y_all = data["y"][target]["train"]
    per_fold, ks = [], None
    for tr_idx, te_idx in _cv_folds(data, target, n_splits):
        fit = _fit_fold(name, train_t, y_all, cols, tr_idx, te_idx, fixed)
        if fit is None:
            per_fold.append(None)
            continue
        model, _, X_te = fit
        trees = getattr(model, "estimators_", None)
        if trees is None or not hasattr(model, "estimators_samples_"):
            raise ValueError(f"{name} is not a bagged forest: no per-tree curve")
        X32 = np.asarray(X_te, dtype=np.float32)
        P = np.vstack([t.predict_proba(X32, check_input=False)[:, 1] for t in trees])
        running = np.cumsum(P, axis=0) / np.arange(1, len(trees) + 1)[:, None]
        # the last point must be the forest itself, or this is not an averaging forest
        if not np.allclose(running[-1], model.predict_proba(X_te)[:, 1], atol=1e-9):
            raise RuntimeError(f"{name}: the mean of the trees does not match predict_proba")
        ks = ks or sorted({1, len(trees), *range(step, len(trees) + 1, step)})
        y_te = y_all[te_idx]
        per_fold.append([float(roc_auc_score(y_te, running[k - 1])) for k in ks])
    if ks is None:
        raise ValueError("every fold was single-class: nothing to score")
    F = np.array([f if f is not None else [float("nan")] * len(ks) for f in per_fold], dtype=float).T
    table = pd.DataFrame({"n_trees": ks, "cv_auc": np.nanmean(F, axis=1), "cv_std": np.nanstd(F, axis=1),
                          **{f"fold_{i}": F[:, i - 1] for i in range(1, F.shape[1] + 1)}})
    print(f" {name} [{_horizon(target)}] CV AUC by tree count: {ks[0]} -> {ks[-1]} trees, "
          f"{table['cv_auc'].iloc[0]:.3f} -> {table['cv_auc'].iloc[-1]:.3f}")
    return table


def boost_rounds(name, model):
    """How many boosting rounds a fitted xgboost / catboost / lightgbm model holds."""
    if name == "xgboost":
        return int(model.get_booster().num_boosted_rounds())
    if name == "catboost":
        return int(model.tree_count_)
    if name == "lightgbm":
        return int(model.n_estimators_)
    raise ValueError(f"{name} is not a boosted model here (xgboost, catboost, lightgbm)")


def staged_scores(name, model, X, ks):
    """A boosted model's probability after its first k rounds, for every k in ks: boosting adds one
    tree's correction per round, so the first k rounds ARE the k-round model."""
    if name == "xgboost":
        return [model.predict_proba(X, iteration_range=(0, k))[:, 1] for k in ks]
    if name == "catboost":
        return [model.predict_proba(X, ntree_end=k)[:, 1] for k in ks]
    if name == "lightgbm":
        return [model.predict_proba(X, num_iteration=k)[:, 1] for k in ks]
    raise ValueError(f"{name} is not a boosted model here (xgboost, catboost, lightgbm)")


def cv_boost_curve(name, data, *, target=STRESS_TARGETS[0], fixed=None, cols=None, step=25, n_splits=None):
    """CV AUC against the number of boosting rounds, train only. Each fold fits ONE model with all the
    rounds asked for (fixed n_estimators / iterations) and scores its first k rounds on the fold's
    test part. The round count is read here, together with the learning rate it was run at: a smaller
    rate needs more rounds, and past the peak more rounds only fit the train period harder.
    Returns: one row per k — n_trees, cv_auc (mean over folds), cv_std and the AUC per fold."""
    cols = _cols_or_all(cols, data)
    train_t, y_all = data["frames_t"][target]["train"], data["y"][target]["train"]
    per_fold, ks = [], None
    for tr_idx, te_idx in _cv_folds(data, target, n_splits):
        fit = _fit_fold(name, train_t, y_all, cols, tr_idx, te_idx, fixed)
        if fit is None:
            per_fold.append(None)
            continue
        model, _, X_te = fit
        n = boost_rounds(name, model)
        ks = ks or sorted({1, n, *range(step, n + 1, step)})
        per_fold.append([float(roc_auc_score(y_all[te_idx], s)) for s in staged_scores(name, model, X_te, ks)])
    if ks is None:
        raise ValueError("every fold was single-class: nothing to score")
    F = np.array([f if f is not None else [float("nan")] * len(ks) for f in per_fold], dtype=float).T
    table = pd.DataFrame({"n_trees": ks, "cv_auc": np.nanmean(F, axis=1), "cv_std": np.nanstd(F, axis=1),
                          **{f"fold_{i}": F[:, i - 1] for i in range(1, F.shape[1] + 1)}})
    best = table.loc[table["cv_auc"].idxmax()]
    print(f" {name} [{_horizon(target)}] CV AUC by round: best {best['cv_auc']:.3f} at {int(best['n_trees'])} "
          f"round(s), {table['cv_auc'].iloc[-1]:.3f} at {ks[-1]}")
    return table


def cv_column_auc(data, *, target=STRESS_TARGETS[0], cols=None, n_splits=None):
    """Each column on its own as the risk score, no model at all: ROC-AUC on each fold's test part.
    This is the bar a model has to clear. 0.5 = no ranking power; under 0.5 = the column ranks stress
    LOW (a model can flip it). flips = True when the folds sit on both sides of 0.5: the column
    changes meaning between periods, which is what makes a model fail out of period.
    Returns: one row per column, strongest first — auc (mean over folds), fold_k, flips."""
    cols = _cols_or_all(cols, data)
    train_t, y_all = data["frames_t"][target]["train"], data["y"][target]["train"]
    folds = [te for _, te in _cv_folds(data, target, n_splits) if len(np.unique(y_all[te])) > 1]
    rows = []
    for c in cols:
        x = train_t[c].to_numpy(dtype=float)
        aucs = [float(roc_auc_score(y_all[te], np.nan_to_num(x[te], nan=np.nanmedian(x[te])))) for te in folds]
        rows.append({"column": c, "auc": float(np.mean(aucs)), **{f"fold_{i}": a for i, a in enumerate(aucs, 1)},
                     "flips": min(aucs) < 0.5 < max(aucs)})
    table = pd.DataFrame(rows)
    table = table.loc[(table["auc"] - 0.5).abs().sort_values(ascending=False, kind="stable").index]
    print(f" [{_horizon(target)}] {len(cols)} column(s) alone: best {table['column'].iloc[0]} "
          f"{table['auc'].iloc[0]:.3f}; {int(table['flips'].sum())} flip sides between folds")
    return table.reset_index(drop=True)


def cv_backward(name, data, *, target=STRESS_TARGETS[0], fixed=None, cols=None, n_splits=None, stop=1):
    """Backward elimination on the tuning folds (train only): from `cols`, drop at each step the one
    column whose removal leaves the best CV AUC, down to `stop` columns. Greedy, so it reads the same
    folds many times: confirm the set you take with smallest_within, cv_feature_sets and more folds.
    Returns: the path, one row per step — n_features, dropped, cv_auc, fold_k, features (the list)."""
    cur = _cols_or_all(cols, data)
    score = lambda c: cross_val_auc(name, data, target, n_splits, params=fixed, cols=c)
    row = lambda c, dropped, cv: {"n_features": len(c), "dropped": dropped, "cv_auc": cv["mean"],
                                  **{f"fold_{i}": f for i, f in enumerate(cv["folds"], 1)}, "features": list(c)}
    rows = [row(cur, None, score(cur))]
    while len(cur) > stop:
        trials = [(d, score([c for c in cur if c != d])) for d in cur]
        dropped, cv = max(trials, key=lambda tr: -1.0 if math.isnan(tr[1]["mean"]) else tr[1]["mean"])
        cur = [c for c in cur if c != dropped]
        rows.append(row(cur, dropped, cv))
    table = pd.DataFrame(rows)
    best = table.loc[table["cv_auc"].idxmax()]
    print(f" {name} [{_horizon(target)}] backward elimination: {int(table['n_features'].iloc[0])} -> {stop} "
          f"column(s), CV {table['cv_auc'].iloc[0]:.3f} -> best {best['cv_auc']:.3f} at {int(best['n_features'])}")
    return table


def smallest_within(path, delta=0.01):
    """The smallest feature set on a cv_backward path whose CV AUC is within delta of the path's best:
    fewer columns unless the extra ones buy a real gain. Returns: the column list."""
    ok = path[path["cv_auc"] >= path["cv_auc"].max() - delta]
    return list(ok.loc[ok["n_features"].idxmin(), "features"])


def score_sets(sets, score_fn, known, label=""):
    """Named feature sets scored by one CV function: cv_feature_sets for the runners outside the
    registry (sequence, Cox, series). score_fn(cols) -> cv_summary shape; cols None = the runner's default.
    Returns: one row per set — n_features, cv_auc, cv_std, folds, paired against the FIRST set."""
    checked = {lab: (check_features(cols, known) if cols is not None else None) for lab, cols in sets.items()}
    rows = []
    for lab, cols in checked.items():
        cv = score_fn(cols)
        rows.append({"set": lab, "n_features": len(cols) if cols is not None else len(known),
                     "cv_auc": cv["mean"], "cv_std": cv["std"], "cv_se": cv["se"],
                     **{f"fold_{i}": f for i, f in enumerate(cv["folds"], 1)}})
    table = paired_folds(pd.DataFrame(rows), ref=0)
    print(f" {label}: {len(rows)} feature set(s) on CV, paired against {rows[0]['set']!r}")
    return table


def seed_spread(name, data, *, fixed=None, seeds=(42, 0, 1, 2, 3), target=STRESS_TARGETS[0], cols=None,
                n_splits=None):
    """The noise floor: one config on the same folds with different random seeds. A gain smaller
    than this spread is the luck of the draw, not the setting. It never picks a seed.
    Returns: one row per seed — cv_auc, cv_pr_auc and the folds."""
    cols = _cols_or_all(cols, data) if cols is not None else None
    rows = []
    for s in seeds:
        cv = cross_val_auc(name, data, target, n_splits, params={**(fixed or {}), SEED_ARG.get(name, "random_state"): s}, cols=cols)
        rows.append({"seed": s, "cv_auc": cv["mean"], "cv_pr_auc": cv["pr_auc_mean"],
                     **{f"fold_{i}": f for i, f in enumerate(cv["folds"], 1)}})
    table = pd.DataFrame(rows)
    fold_sd = table[_fold_cols_of(table)].std(ddof=1).round(4).tolist()
    print(f" {name} [{_horizon(target)}] seed spread over {len(seeds)} seeds: CV AUC "
          f"{table['cv_auc'].min():.4f}-{table['cv_auc'].max():.4f}, SD {table['cv_auc'].std(ddof=1):.4f} "
          f"| SD per fold {fold_sd}")
    return table


# =========================================================================== #
# Tuning: configs, selection rule, table
# =========================================================================== #

SELECT_RULES = ("best", "one_se", "mean_minus_std")


def build_configs(grid, fixed=None, base=None):
    """Every config the tuner tries, lowest priority first: base < grid point < fixed.
    base: the loss settings; fixed: the notebook's fixed values, held in EVERY config and winning
    any clash with the grid. A grid key that fixed also sets stops varying; the duplicate configs
    that leaves are dropped.
    Returns: [(grid_point, config)] — grid_point is the grid's own part, reused on other horizons."""
    fixed, base = fixed or {}, base or {}
    clash = sorted(set().union(*expand_grid(grid)) & set(fixed))
    if clash:
        print(f"   note: fixed sets {clash}, so the grid does not vary them")
    out, seen = [], set()
    for point in expand_grid(grid):
        cfg = {**base, **point, **fixed}
        key = json.dumps(cfg, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            out.append(({k: v for k, v in point.items() if k not in fixed}, cfg))
    return out


def _simplicity_key(config, order):
    """Sort key for 'one_se': smaller = simpler. None counts as no limit (inf); non-numbers count as equal."""
    key = []
    for param, simpler in order:
        v = config.get(param, 0)
        v = math.inf if v is None else v
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            v = 0
        key.append(v if simpler == "low" else -v)
    return tuple(key)


def select_config(cvs, configs, rule, higher_is_better=True, simplicity=None):
    """Index of the winning config.
    best           : the best mean CV score (the old rule).
    one_se         : the SIMPLEST config whose mean is within one standard error of the best; guards
                     against a config that only won by noise. Falls back to best when se is NaN
                     (fewer than 2 valid folds) or the model has no simplicity order.
    mean_minus_std : the best mean - std (mean + std for an error metric); rewards stable configs.
    Returns: int, an index into configs."""
    if rule not in SELECT_RULES:
        raise ValueError(f"select={rule!r}: use one of {SELECT_RULES}")
    sign = 1.0 if higher_is_better else -1.0
    scored = [i for i, cv in enumerate(cvs) if not math.isnan(cv["mean"])]
    if not scored:
        return 0
    best = max(scored, key=lambda i: sign * cvs[i]["mean"])
    if rule == "best":
        return best
    if rule == "mean_minus_std":
        return max(scored, key=lambda i: sign * cvs[i]["mean"] - np.nan_to_num(cvs[i]["std"]))
    se = cvs[best]["se"]
    if math.isnan(se) or not simplicity:
        return best
    near = [i for i in scored if sign * (cvs[i]["mean"] - cvs[best]["mean"]) >= -se]
    return min(near, key=lambda i: (_simplicity_key(configs[i], simplicity), -sign * cvs[i]["mean"]))


def _cell(value):
    """A config value as a table cell: dicts and lists as JSON text, the rest as they are."""
    return json.dumps(value) if isinstance(value, (dict, list, tuple)) else value


def tune(label, configs, cv_fn, select="best", higher_is_better=True, metric="auc",
         guard="pr_auc", simplicity=None):
    """Score every config on walk-forward CV (TRAIN only) and pick one by `select`.
    configs: build_configs output; cv_fn(config) -> cv_summary dict. guard: None for no guard column.
    Returns: {best_params, grid_point, best_cv, table, n_trials, select}.
    table, one row per config: chosen, the parameters that vary, cv_<metric>, cv_std, cv_se,
    cv_<guard>, n_valid, fold_1..k, then the parameters held constant. Best score first."""
    points, cfgs = [p for p, _ in configs], [c for _, c in configs]
    cvs = [cv_fn(c) for c in cfgs]
    pick = select_config(cvs, cfgs, select, higher_is_better, simplicity)

    varying = [k for k in dict.fromkeys(k for c in cfgs for k in c)
               if len({json.dumps(c.get(k), default=str) for c in cfgs}) > 1]
    constant = [k for k in dict.fromkeys(k for c in cfgs for k in c) if k not in varying]
    rows = []
    for i, (cfg, cv) in enumerate(zip(cfgs, cvs)):
        rows.append({"chosen": i == pick,
                     **{k: _cell(cfg.get(k)) for k in varying},
                     f"cv_{metric}": cv["mean"], "cv_std": cv["std"], "cv_se": cv["se"],
                     **({f"cv_{guard}": cv.get(f"{guard}_mean", float("nan"))} if guard else {}),
                     "n_valid": cv["n_valid"],
                     **{f"fold_{j}": f for j, f in enumerate(cv["folds"], 1)},
                     **{k: _cell(cfg.get(k)) for k in constant}})
    table = (pd.DataFrame(rows)
             .sort_values(f"cv_{metric}", ascending=not higher_is_better, na_position="last", kind="stable")
             .reset_index(drop=True))

    means = [cv["mean"] for cv in cvs if not math.isnan(cv["mean"])]
    best_mean = (max(means) if higher_is_better else min(means)) if means else float("nan")
    shown = {k: cfgs[pick].get(k) for k in varying} or "the only config"
    print(f" {label}: {len(cfgs)} config(s), select={select} -> {shown} | "
          f"CV {metric.upper()} {cvs[pick]['mean']:.3f} ± {cvs[pick]['std']:.3f} "
          f"(best {best_mean:.3f})")
    return {"best_params": cfgs[pick], "grid_point": points[pick], "best_cv": cvs[pick],
            "table": table, "n_trials": len(cfgs), "select": select}


def _horizon(target):
    """'y_stress_..._24h' -> '24h'; REG_TARGET has no horizon suffix and is returned whole."""
    return target.rpartition("_")[2] if target != REG_TARGET else target


def tune_classifier(name, data, target=STRESS_TARGETS[0], configs=None, select="best",
                    cols=None, n_splits=None):
    """Pick a classifier config on embargoed walk-forward CV AUC (TRAIN only).
    configs: build_configs output — defaults to the model's default grid.
    cols: feature subset to tune on — defaults to every feature.
    Returns: tune()'s dict."""
    configs = configs or build_configs(CLASSIFIER_GRIDS.get(name))
    return tune(f"{name} [{_horizon(target)}]", configs,
                lambda cfg: cross_val_auc(name, data, target, n_splits, params=cfg, cols=cols),
                select, higher_is_better=True, metric="auc", guard="pr_auc",
                simplicity=SIMPLICITY.get(name))


def tune_regressor(name, data, target=REG_TARGET, configs=None, select="best", cols=None,
                   n_splits=None):
    """Pick a regressor config on walk-forward CV RMSE (score points, TRAIN only); MAE shown beside it.
    Returns: tune()'s dict."""
    # train-only selection, same rule as tune_classifier
    configs = configs or build_configs(REGRESSOR_GRIDS.get(name))
    return tune(f"{name} [regressor]", configs,
                lambda cfg: cross_val_rmse(name, data, target, n_splits, params=cfg, cols=cols),
                select, higher_is_better=False, metric="rmse", guard="mae",
                simplicity=SIMPLICITY.get(name))


def _from_cell(value):
    """A tuning-table cell back to a value for RANKING only: NaN -> None (no limit), numpy -> Python."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return _plain(value)


def compare_rules(result, *, horizon="24h", name=None):
    """Which config each selection rule (best / one_se / mean_minus_std) would pick from a tuning
    table, without refitting: the same scores, three ways of reading them.
    result: a runner result (its tuning table, name and select rule), or a tuning table plus name=.
    The configs are rebuilt from the table only to rank them; a config to FIT always comes from
    config_by_target (pandas turns [6, None] into 6.0 / NaN, which sklearn refuses).
    Returns: one row per rule — what it picks, its CV, se and folds, and whether it is the run's rule."""
    if isinstance(result, pd.DataFrame):
        table, run_rule = result, None
    else:
        table, name, run_rule = result.get("tuning"), result["name"], result.get("select")
        if table is None:
            raise ValueError(f"{result.get('name')}: no tuning table (a baseline, or a result read back from disk)")
    t = table
    if "horizon" in t.columns:
        t = t[t["horizon"] == horizon]
        if t.empty:
            raise ValueError(f"no {horizon!r} rows in this table: {sorted(table['horizon'].unique())}")
    fold_cols = _fold_cols_of(t)
    cv_cols = [c for c in t.columns if str(c).startswith("cv_")]
    metric = cv_cols[0]                                  # tune() writes cv_<metric> first, then std, se, guard
    skip = {"horizon", "chosen", "n_valid", *fold_cols, *cv_cols}
    params = [c for c in t.columns if c not in skip and t[c].notna().any()]
    varying = [c for c in params if t[c].astype(str).nunique() > 1]
    cfgs = [{c: _from_cell(row[c]) for c in params} for _, row in t.iterrows()]
    cvs = [{"mean": float(row[metric]), "std": float(row["cv_std"]), "se": float(row["cv_se"])}
           for _, row in t.iterrows()]
    higher = metric != "cv_rmse"
    chosen = list(t["chosen"].astype(bool)) if "chosen" in t.columns else None
    rows = []
    for rule in SELECT_RULES:
        # the table is sorted by CV with a stable sort, so exact ties keep the order tune() saw them in
        i = select_config(cvs, cfgs, rule, higher, SIMPLICITY.get(name))
        row = t.iloc[i]
        rows.append({"rule": rule, **{c: row[c] for c in varying}, metric: row[metric], "cv_se": row["cv_se"],
                     **{c: row[c] for c in fold_cols}, "run_rule": rule == run_rule,
                     "same_as_chosen": bool(chosen[i]) if chosen is not None else False})
    out = pd.DataFrame(rows)
    if run_rule in SELECT_RULES and not out.loc[out["run_rule"], "same_as_chosen"].all():
        print(" note: the table cannot reproduce the run's own pick (a list-of-dicts grid with missing keys? "
              "pass it through model_zoo.fill_defaults)")
    return out
