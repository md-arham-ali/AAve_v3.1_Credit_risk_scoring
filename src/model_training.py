"""Modeling stage — model runners (the training pass) + the PRI meta-learner.

One runner per kind: tuned classifiers/regressors, baselines, IsolationForest, Cox PH,
and the LSTM / Temporal Transformer sequence classifiers.

Every training control except initialization is set from the notebook through the runner's
arguments; the defaults live in model_zoo.py. Each result carries the settings it actually used
(params, params_by_target), what changed from the house defaults, a settings_hash, the tuning
table and the fitted models (in memory only, for save_fitted).
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hashlib
import json
import math
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from model_config import (TIME_COL, STRESS_TARGETS, REG_TARGET, HORIZONS,
                          RANDOM_STATE, LIFELINES_AVAILABLE, TORCH_AVAILABLE,
                          rank_normalize)
from model_zoo import (CLASSIFIER_SPECS, REGRESSOR_SPECS, CLASSIFIER_GRIDS, REGRESSOR_GRIDS,
                       RUNNER_SPECS, RUNNER_GRIDS, SIMPLICITY, SEQUENCE_LOSSES,
                       build_model, check_names, default_select, expand_grid, loss_params,
                       model_settings, _plain, _xgb_overrides, predict_scores,
                       score_is_probability)
from model_evaluation import (best_f1_threshold, evaluate_classification,
                              evaluate_regression, bootstrap_auc_ci, build_configs,
                              check_features, cross_val_custom, fold_auc, rank_features,
                              tune_classifier, tune_regressor, _fold_cols, _fold_matrices, _horizon,
                              cv_oof, oof_frame, oof_data, score_sets, fold_auc)
from model_evaluation import tune as run_tuning
from model_split import fit_scaler, apply_scaler, time_of


# =========================================================================== #
# Shared helpers
# =========================================================================== #

def _importances(model, columns):
    """Native feature importances as {column: weight} (trees) or |coef| (linear)."""
    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        vals = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
    else:
        return None
    return {c: float(v) for c, v in zip(columns, vals)}


def top_n_features(data, target, n=15, x_key="raw", method="forest"):
    """Train-only feature ranking for collinearity-sensitive models (linear, Cox: the rolled
    windows are strongly correlated). method "forest": a seeded RandomForest's importance, strongest
    first; classifier for a stress target, regressor for the scaled volume target. method "alone":
    each column's own |AUC - 0.5| against the stress label (model_evaluation.rank_features).
    Check the count first with model_evaluation.cv_top_n, which ranks inside every fold.
    n: how many to keep; None = every feature, ranked (slice it per model: ranked[:8]).
    Returns: the top-n column names, in order of rank (not alphabetical)."""
    ranked = rank_features(data["X"][target][x_key]["train"], data["y"][target]["train"],
                           data["feature_cols"], regression=(target == REG_TARGET), method=method)
    return ranked if n is None else ranked[:n]


def _select(X, feature_cols, cols):
    """Subset each split's array to `cols`, by name, in `cols`' order."""
    return {s: pd.DataFrame(x, columns=feature_cols)[cols].to_numpy() for s, x in X.items()}


def monotone_signs(data, target, cols, min_gap=0.02):
    """The direction each column ranks the stress label in on the TRAIN rows, for monotone_constraints:
    +1 (higher value, more stress), -1 (less), 0 (free) when the column's own AUC is within min_gap of 0.5.
    Val and test never choose a sign. A column whose direction flips between periods is exactly what a
    constraint should stop a tree from chasing (round two, 2026-11-16).
    Returns: {column: sign}, in cols' order: tuple(signs.values()) for XGBoost, list(...) for CatBoost."""
    from sklearn.metrics import roc_auc_score
    X = _select(data["X"][target]["raw"], data["feature_cols"], list(cols))["train"]
    y = data["y"][target]["train"]
    out = {}
    for i, c in enumerate(cols):
        x = np.nan_to_num(X[:, i].astype(float))
        a = float(roc_auc_score(y, x)) if len(np.unique(y)) > 1 and np.ptp(x) > 0 else 0.5
        out[c] = 0 if abs(a - 0.5) < min_gap else (1 if a > 0.5 else -1)
    return out


def _per_horizon(values, targets):
    """Notebook values -> {target: dict}.
    values: None, a flat dict (the same for every horizon), or a dict keyed by horizon:
    {"24h": {...}, "7d": {...}}. A horizon left out takes the 24h values.
    Raises when horizon keys and parameter names are mixed, or a horizon is not in this run."""
    if not values:
        return {t: {} for t in targets}
    keys = set(values)
    horizon_keys = keys & set(HORIZONS)
    if not horizon_keys:
        return {t: dict(values) for t in targets}
    if horizon_keys != keys:
        raise ValueError(f"per-horizon values mixed with parameter names {sorted(keys - horizon_keys)}: "
                         "give one dict per horizon, e.g. {'24h': {...}, '7d': {...}}")
    missing = horizon_keys - {_horizon(t) for t in targets}
    if missing:
        raise ValueError(f"no {sorted(missing)} horizon in this run: {[_horizon(t) for t in targets]}")
    fallback = values.get(HORIZONS[0], {})
    return {t: dict(values.get(_horizon(t), fallback)) for t in targets}


def _features_by_horizon(features, targets, known):
    """Notebook features -> {target: [cols] or None (every feature)}, every list checked before any fit.
    features: None, one list (every horizon), or a dict keyed by horizon: {"24h": [...], "7d": [...]}
    (a horizon left out takes the 24h list, or every feature when there is no 24h list; None = every feature).
    A longer horizon can want slower features than a short one, so each horizon may get its own list."""
    if features is None:
        return {t: None for t in targets}
    if not isinstance(features, dict):
        cols = check_features(features, known)
        return {t: cols for t in targets}
    bad = sorted(set(features) - set(HORIZONS))
    if bad:
        raise ValueError(f"features= keys must be horizons {list(HORIZONS)}, got {bad}")
    missing = set(features) - {_horizon(t) for t in targets}
    if missing:
        raise ValueError(f"no {sorted(missing)} horizon in this run: {[_horizon(t) for t in targets]}")
    checked = {h: (check_features(cols, known) if cols is not None else None) for h, cols in features.items()}
    fallback = checked.get(HORIZONS[0])
    return {t: checked.get(_horizon(t), fallback) for t in targets}


def _settings_hash(payload):
    """Short hash of everything that decides a fit: used settings per horizon, features, loss.
    Same run key + same hash = trained the same way. PRI records it per component."""
    text = json.dumps(_plain(payload), sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _finish(out, primary, used, defaults, tuned, fitted, save):
    """Fields every trained result carries.
    used / defaults: {target: settings} actually used / with no notebook values (house defaults).
    params            the primary horizon's used settings, flat (what MLflow and older readers read)
    params_by_target  used settings per horizon
    changed           names whose value differs from the house default on any horizon
    settings_hash     see _settings_hash
    tuning            every tuning table as one DataFrame, a `horizon` column first (saved as CSV)
    n_trials          configs scored on CV for this result, so the search effort stays visible
    fitted            {target: fitted model}: in memory for save_fitted, never written to the JSON
    trial             save=False: save_model_result and save_fitted write nothing"""
    # compared as JSON text: xgboost's missing=nan would never equal itself otherwise
    same = lambda a, b: json.dumps(_plain(a), sort_keys=True) == json.dumps(_plain(b), sort_keys=True)
    out["params"] = used[primary]
    out["params_by_target"] = used
    out["changed"] = sorted({k for t, s in used.items() for k, v in s.items()
                             if not same(defaults[t].get(k), v)})
    # per-horizon features enter the hash only when given, so every older hash stays the same
    out["settings_hash"] = _settings_hash({"name": out["name"], "kind": out["kind"], "used": used,
                                           "features": out.get("features_by_target") or out.get("features_used"),
                                           "loss": out.get("loss")})
    tables = []
    for t, tn in tuned.items():
        table = tn["table"].copy()
        table.insert(0, "horizon", _horizon(t))
        tables.append(table)
    out["tuning"] = pd.concat(tables, ignore_index=True) if tables else None
    out["n_trials"] = sum(tn["n_trials"] for tn in tuned.values())
    out["fitted"] = fitted
    out["trial"] = not save
    if not save:
        print(f"   trial run (save=False): {out['name']} will not be saved")
    return out


def _persistence(data, target):
    """The persistence baseline: at each row t, the latest label ALREADY KNOWN at t.
    A label stamped s covers (s, s + h], so it is known only once s + h <= t: the row one full
    horizon back (24h -> 12 rows at 2h), never the row just before. Matched by time on every
    labelled row (data["label_history"], embargo and warm-up rows included: their outcomes are
    public by then), so a split boundary or the seam hole just reads the latest earlier label.
    A row with no known label yet gets the train median: there is nothing to persist.
    Returns: {split: array}, same shape as data["y"][target]."""
    hist = data["label_history"][target]
    labels, horizon = hist["labels"], hist["horizon"]
    fallback = float(np.median(data["y"][target]["train"]))
    out = {}
    for s in ("train", "val", "test"):
        times = pd.Series(data["times"][target][s]).astype(str).str.replace(" UTC", "", regex=False)
        cutoff = pd.DatetimeIndex(pd.to_datetime(times, utc=True)) - horizon
        pos = labels.index.searchsorted(cutoff, side="right") - 1
        out[s] = np.where(pos >= 0, labels.to_numpy(dtype=float)[np.maximum(pos, 0)], fallback)
    return out


def _prediction_frame(data, target, scores_by_split, y_by_split):
    """Tidy full-timeline prediction frame: time_bucket, split, y_true, score.
    target: whose {train,val,test} row set data["times"] to read (each target has its own)."""
    parts = []
    for s in ("train", "val", "test"):
        parts.append(pd.DataFrame({TIME_COL: data["times"][target][s], "split": s,
                                   "y_true": np.asarray(y_by_split[s]).astype(float),
                                   "score": np.asarray(scores_by_split[s], dtype=float)}))
    return pd.concat(parts, ignore_index=True)


# =========================================================================== #
# Registry classifiers and regressors
# =========================================================================== #

def run_classifier(name, data, targets=None, cv_target=STRESS_TARGETS[0], fixed=None, grid=None,
                   tune=True, tune_per_horizon=False, select=None, loss=None, features=None,
                   save=True, *, oof=False):
    """Tune (walk-forward, train-only) on cv_target, fit one model PER horizon in `targets`
    (each on its OWN train/val/test — a 3d model and a 15d model never see the same rows),
    report train/val/test. cv_target's full-timeline scores are what the PRI blend consumes.
    targets: defaults to data["stress_targets"] — this split's own horizons.
    fixed: values held fixed. A flat dict = every horizon; keyed by horizon ({"24h": {...},
        "7d": {...}}) = per horizon, a horizon left out takes the 24h values. Fixed wins over the grid.
    grid: replaces the default grid (model_zoo.CLASSIFIER_GRIDS), same grid on every horizon.
    tune: False = no search, the defaults + fixed values only (CV still scores that config).
    tune_per_horizon: tune every horizon on its own folds, instead of reusing the cv_target pick.
    select: "best" / "one_se" / "mean_minus_std"; default model_zoo.default_select(name).
    loss: a name from model_zoo.CLASSIFIER_LOSSES; None keeps the library's own loss.
    features: column subset to fit on (e.g. top_n_features, for a linear model) — defaults
        to every feature; tuning and CV run on the same subset, not the full set. Per horizon as
        {"24h": [...], "7d": [...]}, a horizon left out takes the 24h list (like fixed).
    save: False = trial run, nothing is written by save_model_result / save_fitted.
    oof: True adds out["oof"], the out-of-fold scores of every horizon's final config
        (model_evaluation.cv_oof): one more CV per horizon. The ensemble learns from these.
        An int = that many folds instead of the split's own (oof=5).
    Returns: result dict; cv_target test AUC carries a block-bootstrap CI. features_used is
    cv_target's list; features_by_target is added when features were given per horizon."""
    targets = targets if targets is not None else data["stress_targets"]
    if cv_target not in targets:
        raise ValueError(f"cv_target {_horizon(cv_target)} is not in targets {[_horizon(t) for t in targets]}")
    # before any fitting: a typo or an empty list must stop the run, not fail inside a fold
    feats_by_t = _features_by_horizon(features, targets, data["feature_cols"])
    spec = CLASSIFIER_SPECS[name]
    xkey = "scaled" if spec["scaled"] else "raw"
    cols_by_t = {t: feats_by_t[t] or data["feature_cols"] for t in targets}
    fixed_by_t = _per_horizon(fixed, targets)
    grid = CLASSIFIER_GRIDS.get(name) if grid is None else grid
    check_names(name, "classifier", *fixed_by_t.values(), *expand_grid(grid))
    # an int max_features is a column COUNT: above a horizon's column count sklearn quietly uses them all
    too_many = sorted({(_horizon(t), mf) for t in targets for p in expand_grid(grid)
                       for mf in [{**p, **fixed_by_t[t]}.get("max_features")]
                       if isinstance(mf, int) and not isinstance(mf, bool) and mf > len(cols_by_t[t])})
    if too_many:
        raise ValueError(f"max_features above a horizon's column count {too_many}: use a smaller int or a share (float)")
    base = loss_params(name, "classifier", loss)
    select = select or default_select(name)

    # search once on cv_target and reuse the pick on every horizon (as before), or per horizon
    tuned = {}
    for t in (targets if (tune and tune_per_horizon) else [cv_target]):
        configs = build_configs(grid if tune else None, fixed_by_t[t], base)
        tuned[t] = tune_classifier(name, data, t, configs, select, cols=feats_by_t[t])

    out = {"name": name, "kind": "classifier", "status": "trained",
           "scaled_input": spec["scaled"], "select": select, "loss": loss,
           "features_used": cols_by_t[cv_target], "config_by_target": {}, "thresholds_f1": {}, "targets": {}}
    if isinstance(features, dict):
        out["features_by_target"] = cols_by_t
    used, defaults, fitted, proper = {}, {}, {}, None
    for t in targets:
        X_t = data["X"][t][xkey]
        X = _select(X_t, data["feature_cols"], cols_by_t[t]) if feats_by_t[t] else X_t
        y_tr = data["y"][t]["train"]
        point = tuned.get(t, tuned[cv_target])["grid_point"]
        config = {**base, **point, **fixed_by_t[t]}
        auto = _xgb_overrides(y_tr) if name == "xgboost" else {}
        model = build_model(name, "classifier", **{**auto, **config})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X["train"], y_tr)
        proper = score_is_probability(model)
        scores = {s: predict_scores(model, X[s], X["train"]) for s in ("train", "val", "test")}
        thr = best_f1_threshold(data["y"][t]["val"], scores["val"])
        out["thresholds_f1"][t] = thr
        out["targets"][t] = {s: evaluate_classification(
            data["y"][t][s], scores[s], threshold=thr, proper_probs=proper)
            for s in ("train", "val", "test")}
        out["config_by_target"][t] = config
        used[t] = model_settings(model)
        defaults[t] = model_settings(build_model(name, "classifier", **auto))
        fitted[t] = model
        if t == cv_target:
            out["importances"] = _importances(model, cols_by_t[t])
            out["predictions"] = _prediction_frame(data, t, scores, data["y"][t])
            out["test_auc_ci"] = bootstrap_auc_ci(data["y"][t]["test"], scores["test"])
    out["proper_probs"] = proper
    if oof:
        out["oof"] = pd.concat([cv_oof(name, oof_data(data, oof), target=t, params=out["config_by_target"][t],
                                       cols=feats_by_t[t]) for t in targets], ignore_index=True)
    out["proper_probs"] = proper
    out["cv"] = tuned[cv_target]["best_cv"]
    m, ci = out["targets"][cv_target], out["test_auc_ci"]
    # a trial run hides test: configs are picked on CV, and a test number printed at every
    # stage becomes one more thing to tune on. The result still holds it (stage_table test_for=)
    scores = (f"train/val/test ROC-AUC {m['train']['roc_auc']:.3f}/{m['val']['roc_auc']:.3f}/"
              f"{m['test']['roc_auc']:.3f} [{ci['lo']:.3f},{ci['hi']:.3f}]" if save else
              f"train/val ROC-AUC {m['train']['roc_auc']:.3f}/{m['val']['roc_auc']:.3f} (test hidden on a trial run)")
    print(f" {name}: {scores} | CV {out['cv']['mean']:.3f}±{out['cv']['std']:.3f}")
    return _finish(out, cv_target, used, defaults, tuned, fitted, save)


def run_regressor(name, data, target=REG_TARGET, fixed=None, grid=None, tune=True, select=None,
                  loss=None, features=None, save=True):
    """Tune (walk-forward, train-only), fit on the scaled score, report train/val/test.
    fixed / grid / tune / select / save: as in run_classifier (one target, so fixed is one flat dict).
    loss: a name from model_zoo.REGRESSOR_LOSSES; None keeps the library's own loss.
    features: column subset to fit on (e.g. top_n_features, for a linear model) — defaults
    to every feature; tuning runs on the same subset, not the full set.
    Returns: result dict; rmse/mae/r2 in score points; mae_usd reversed to real USD, with the
    prediction clipped to the train target max first."""
    if features is not None:
        features = check_features(features, data["feature_cols"])
    spec = REGRESSOR_SPECS[name]
    xkey = "scaled" if spec["scaled"] else "raw"
    cols = features or data["feature_cols"]
    X_t = data["X"][target][xkey]
    X = _select(X_t, data["feature_cols"], cols) if features else X_t
    y = data["y"][target]
    clip = float(np.max(y["train"]))
    grid = REGRESSOR_GRIDS.get(name) if grid is None else grid
    check_names(name, "regressor", fixed or {}, *expand_grid(grid))
    base = loss_params(name, "regressor", loss)
    select = select or default_select(name)

    configs = build_configs(grid if tune else None, fixed, base)
    tuned = tune_regressor(name, data, target, configs, select, cols=features)
    config = tuned["best_params"]
    model = build_model(name, "regressor", **config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X["train"], y["train"])
    preds = {s: np.asarray(model.predict(X[s]), dtype=float) for s in ("train", "val", "test")}
    cv = tuned["best_cv"]
    out = {"name": name, "kind": "regressor", "status": "trained",
           "scaled_input": spec["scaled"], "target": target, "select": select, "loss": loss,
           "features_used": cols, "config_by_target": {target: config},
           "cv": {"rmse_mean": cv["mean"], "rmse_std": cv["std"], "rmse_se": cv["se"],
                  "mae_mean": cv["mae_mean"], "folds": cv["folds"]},
           "metrics": {s: evaluate_regression(y[s], preds[s], clip_max=clip,
                                              to_usd=data["to_usd"].get(target))
                       for s in ("train", "val", "test")},
           "importances": _importances(model, cols),
           "predictions": _prediction_frame(data, target, preds, y)}
    m = out["metrics"]
    print(f" {name}: train/val/test R² {m['train']['r2']:.3f}/{m['val']['r2']:.3f}/"
          f"{m['test']['r2']:.3f} | test MAE {m['test']['mae']:.3f} pts "
          f"| MAE ${m['test']['mae_usd']:,.0f}")
    return _finish(out, target, {target: model_settings(model)},
                   {target: model_settings(build_model(name, "regressor"))},
                   {target: tuned}, {target: model}, save)


# =========================================================================== #
# Baselines
# =========================================================================== #

def run_baseline_classifiers(data, cv_target=STRESS_TARGETS[0]):
    """Two no-model references per stress target: persistence and prevalence.
    persistence = the latest stress flag already known at t, one full horizon back (_persistence);
    prevalence = constant train positive rate (AUC 0.5 anchor). Each horizon uses its own
    train/val/test rows.
    Returns: a list of result dicts. Baselines stay out of the PRI blend (_is_meta_component)."""
    out = []
    for base in ("persistence", "prevalence"):
        res = {"name": f"baseline_{base}", "kind": "classifier", "status": "trained",
               "params": {}, "thresholds_f1": {}, "proper_probs": False, "targets": {}}
        for t in data["stress_targets"]:
            if base == "persistence":
                scores = _persistence(data, t)
            else:
                rate = float(np.mean(data["y"][t]["train"]))
                scores = {s: np.full(len(data["y"][t][s]), rate) for s in ("train", "val", "test")}
            thr = best_f1_threshold(data["y"][t]["val"], scores["val"])
            res["thresholds_f1"][t] = thr
            res["targets"][t] = {s: evaluate_classification(
                data["y"][t][s], scores[s], threshold=thr, proper_probs=False)
                for s in ("train", "val", "test")}
            if t == cv_target:
                res["predictions"] = _prediction_frame(data, t, scores, data["y"][t])
                res["test_auc_ci"] = bootstrap_auc_ci(data["y"][t]["test"], scores["test"])
        m = res["targets"][cv_target]["test"]
        print(f" baseline_{base}: test ROC-AUC {m['roc_auc']:.3f} | PR-AUC {m['pr_auc']:.3f}")
        out.append(res)
    return out


def run_baseline_regressors(data, target=REG_TARGET):
    """Two no-model references for the volume target: persistence and train median.
    persistence = the latest score already known at t, one full horizon back (_persistence)."""
    y = data["y"][target]
    clip = float(np.max(y["train"]))
    med = float(np.median(y["train"]))
    out = []
    for base, preds in (("persistence", _persistence(data, target)),
                        ("median", {s: np.full(len(y[s]), med) for s in y})):
        res = {"name": f"baseline_{base}", "kind": "regressor", "status": "trained",
               "target": target, "params": {},
               "metrics": {s: evaluate_regression(y[s], preds[s], clip_max=clip,
                                                  to_usd=data["to_usd"].get(target))
                           for s in ("train", "val", "test")},
               "predictions": _prediction_frame(data, target, preds, y)}
        m = res["metrics"]["test"]
        print(f" baseline_{base}: test RMSE {m['rmse']:.3f} pts | R² {m['r2']:.3f} "
              f"| MAE ${m['mae_usd']:,.0f}")
        out.append(res)
    return out


# =========================================================================== #
# IsolationForest
# =========================================================================== #

def _iforest(settings):
    """IsolationForest from its settings (model_zoo.RUNNER_SPECS["isolation_forest"] keys)."""
    return IsolationForest(**settings, random_state=RANDOM_STATE, n_jobs=-1)


def _iforest_cv(data, target, settings):
    """Walk-forward CV AUC of the anomaly score: fit on a fold's train features (no labels),
    score its test part. The scaler is re-fit inside the fold, like the registry models."""
    train_t, cols, y = data["frames_t"][target]["train"], data["feature_cols"], data["y"][target]["train"]

    def fold(tr, te):
        X_tr, X_te = _fold_matrices(train_t, cols, tr, te, scaled=False)
        model = _iforest(settings).fit(X_tr)
        return fold_auc(y[te], -model.score_samples(X_te))
    return cross_val_custom(data, target, fold)


def run_isolation_forest(data, target=STRESS_TARGETS[0], tune=False, grid=None, select=None,
                         save=True, **settings):
    """Unsupervised anomaly model: fit on TRAIN features only, no labels.
    Anomaly score = 1 - ECDF_train(score_samples), higher = more anomalous. Single horizon
    (target, "24h" by default) — one fit/score pass, own train/val/test rows throughout.
    settings: any key of model_zoo.RUNNER_SPECS["isolation_forest"] (n_estimators, contamination,
    max_samples, max_features). tune: True picks from the grid on walk-forward CV AUC.
    Returns: result dict with rank metrics only — Brier is meaningless here."""
    spec = RUNNER_SPECS["isolation_forest"]
    grid = RUNNER_GRIDS.get("isolation_forest") if grid is None else grid
    check_names("isolation_forest", "runner", settings, *expand_grid(grid))
    select = select or default_select("isolation_forest")
    tuned = {}
    if tune:
        tuned[target] = run_tuning(
            f"isolation_forest [{_horizon(target)}]", build_configs(grid, settings, base=spec),
            lambda s: _iforest_cv(data, target, s), select, higher_is_better=True,
            metric="auc", guard="pr_auc", simplicity=SIMPLICITY.get("isolation_forest"))
    s = {**spec, **(tuned[target]["grid_point"] if tuned else {}), **settings}

    X = data["X"][target]["raw"]
    model = _iforest(s)
    model.fit(X["train"])
    raw = {sp: -model.score_samples(X[sp]) for sp in ("train", "val", "test")}
    scores = {sp: rank_normalize(raw[sp], raw["train"]) for sp in ("train", "val", "test")}
    out = {"name": "isolation_forest", "kind": "anomaly", "status": "trained",
           "proper_probs": False, "select": select if tune else None,
           "features_used": data["feature_cols"], "config_by_target": {target: s},
           "targets": {target: {sp: evaluate_classification(data["y"][target][sp], scores[sp],
                                                             proper_probs=False)
                                for sp in ("train", "val", "test")}},
           "test_auc_ci": bootstrap_auc_ci(data["y"][target]["test"], scores["test"]),
           "predictions": _prediction_frame(data, target, scores, data["y"][target])}
    if tuned:
        out["cv"] = tuned[target]["best_cv"]
    m = out["targets"][target]["test"]
    print(f" isolation_forest: test ROC-AUC {m['roc_auc']:.3f} | PR-AUC {m['pr_auc']:.3f} "
          f"(anomaly score vs {target} — judge it against baseline_persistence)")
    return _finish(out, target, {target: model_settings(model)},
                   {target: model_settings(_iforest(spec))}, tuned, {target: model}, save)


# =========================================================================== #
# Cox PH
# =========================================================================== #

def build_survival_frame(times, stress_flags):
    """Per-row survival target: HOURS until the next row flagged as stress.
    times, stress flags, in time order -> duration from t (exclusive), event = 1.
    Hours, not whole days: at the 2h grain a whole-day count floors every gap under 24h to 0.
    Returns: the survival frame; rows after the last stress row are right-censored."""
    stamps = pd.to_datetime(pd.Series(times).astype(str).str.replace(" UTC", "", regex=False))
    hours = ((stamps - stamps.iloc[0]) / pd.Timedelta("1h")).to_numpy()
    stress_idx = np.flatnonzero(np.asarray(stress_flags).astype(int))
    nxt = np.searchsorted(stress_idx, np.arange(len(hours)), side="right")
    has_next = nxt < len(stress_idx)
    end = np.where(has_next, hours[stress_idx[np.minimum(nxt, len(stress_idx) - 1)]], hours[-1]) \
        if len(stress_idx) else np.full(len(hours), hours[-1])
    return pd.DataFrame({"duration": end - hours, "event": has_next.astype(int)})


def _cox_frame(X, cols, times, flags):
    """Covariates + survival target for one block of rows; the final row (duration 0) is dropped."""
    sv = build_survival_frame(times, flags)
    keep = sv["duration"] > 0
    return pd.concat([pd.DataFrame(X, columns=cols), sv], axis=1).loc[keep].reset_index(drop=True)


def _cox_fitter(settings):
    """A CoxPHFitter from its settings. l1_ratio 0 = pure L2 (the old behaviour), 1 = pure L1."""
    from lifelines import CoxPHFitter
    return CoxPHFitter(penalizer=settings["penalizer"], l1_ratio=settings["l1_ratio"])


def _cox_cv(data, target, settings, cols=None, parts=None):
    """Walk-forward CV C-index on TRAIN. Each fold ranks its own top-N features on its train part
    and re-fits the scaler there, so the fold's test rows never pick features or scale them.
    cols: an explicit column list instead of the fold's top-N. The AUC of the risk score against
    the stress label rides along as the guard (auc_mean): the number other models are compared on.
    parts collects (fold, test rows, risk scores) for the out-of-fold frame."""
    from lifelines.utils import concordance_index
    train_t, cols_all = data["frames_t"][target]["train"], data["feature_cols"]
    y, times = data["y"][target]["train"], data["times"][target]["train"]

    count = iter(range(1, 10 ** 6))

    def fold(tr, te):
        k = next(count)
        if len(np.unique(y[tr])) < 2:
            return {"main": float("nan"), "auc": float("nan")}
        use = cols or _fold_cols(train_t, cols_all, tr, y[tr], settings["top_n_features"], regression=False)
        X_tr, X_te = _fold_matrices(train_t, use, tr, te, scaled=True)
        f_tr = _cox_frame(X_tr, use, [times[i] for i in tr], y[tr])
        f_te = _cox_frame(X_te, use, [times[i] for i in te], y[te])
        if f_te["event"].sum() == 0:
            return {"main": float("nan"), "auc": float("nan")}
        cph = _cox_fitter(settings)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph.fit(f_tr, duration_col="duration", event_col="event")
            haz = np.asarray(cph.predict_partial_hazard(f_te[use]), dtype=float).ravel()
            risk = np.asarray(cph.predict_partial_hazard(pd.DataFrame(X_te, columns=use)), dtype=float).ravel()
        if parts is not None:
            parts.append((k, te, risk))
        return {"main": float(concordance_index(f_te["duration"], -haz, f_te["event"])),
                "auc": fold_auc(y[te], risk)["main"]}
    return cross_val_custom(data, target, fold)


def run_cox(data, target=STRESS_TARGETS[0], tune=False, grid=None, select=None, save=True, *,
            features=None, oof=False, **settings):
    """Cox proportional-hazards on time-to-next-stress-row (lifelines, optional). Single
    horizon (target, "24h" by default) throughout — its own train/val/test rows.
    Features = top-N by seeded RandomForest importance on train, scaled; C-index on val/test.
    settings: any key of model_zoo.RUNNER_SPECS["cox_ph"] (penalizer, l1_ratio, top_n_features).
    tune: True picks from the grid on walk-forward CV C-index.
    features: an explicit column list (checked before any fit) instead of the top-N ranking.
    oof: True adds out["oof"], the out-of-fold risk scores of the final config.
    Returns: result dict, or a skipped-result dict if lifelines is absent or the fit diverges.
    metrics[split] holds the C-index and roc_auc (the risk score against the stress label)."""

    # top-N is collinearity control: the rolled windows are strongly correlated
    if not LIFELINES_AVAILABLE:
        print(" cox: SKIPPED — lifelines not installed")
        return {"name": "cox_ph", "kind": "survival", "status": "skipped",
                "reason": "lifelines not installed"}
    spec = RUNNER_SPECS["cox_ph"]
    features = check_features(features, data["feature_cols"]) if features is not None else None
    grid = RUNNER_GRIDS.get("cox_ph") if grid is None else grid
    check_names("cox_ph", "runner", settings, *expand_grid(grid))
    select = select or default_select("cox_ph")
    try:
        from lifelines.utils import concordance_index

        tuned = {}
        if tune:
            tuned[target] = run_tuning(
                f"cox_ph [{_horizon(target)}]", build_configs(grid, settings, base=spec),
                lambda s: _cox_cv(data, target, s, cols=features), select, higher_is_better=True,
                metric="c_index", guard="auc", simplicity=SIMPLICITY.get("cox_ph"))
        s = {**spec, **(tuned[target]["grid_point"] if tuned else {}), **settings}

        cols = features or top_n_features(data, target, n=s["top_n_features"])
        X = _select(data["X"][target]["scaled"], data["feature_cols"], cols)
        frames = {sp: _cox_frame(X[sp], cols, data["times"][target][sp], data["y"][target][sp])
                  for sp in ("train", "val", "test")}
        cph = _cox_fitter(s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph.fit(frames["train"], duration_col="duration", event_col="event")

        metrics, scores = {}, {}
        for sp in ("train", "val", "test"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores[sp] = np.asarray(cph.predict_partial_hazard(
                    pd.DataFrame(X[sp], columns=cols)), dtype=float).ravel()
                haz_kept = np.asarray(cph.predict_partial_hazard(frames[sp][cols]), dtype=float).ravel()
            f = frames[sp]
            metrics[sp] = {"c_index": float(concordance_index(f["duration"], -haz_kept, f["event"])),
                           "roc_auc": fold_auc(data["y"][target][sp], scores[sp])["main"]}
        norm = {sp: rank_normalize(scores[sp], scores["train"]) for sp in scores}
        out = {"name": "cox_ph", "kind": "survival", "status": "trained",
               "select": select if tune else None, "features_used": cols,
               "config_by_target": {target: s}, "metrics": metrics,
               "predictions": _prediction_frame(data, target, norm, data["y"][target])}
        if tuned:
            out["cv"] = {"c_index_mean": tuned[target]["best_cv"]["mean"],
                         "c_index_std": tuned[target]["best_cv"]["std"],
                         "auc_mean": tuned[target]["best_cv"].get("auc_mean")}
        if oof:
            parts = []
            _cox_cv(oof_data(data, oof), target, s, cols=features, parts=parts)
            out["oof"] = oof_frame(data, target, parts)
        # a trial run hides test, as run_classifier does
        tail = (f"test {metrics['test']['c_index']:.3f} | ROC-AUC val/test {metrics['val']['roc_auc']:.3f}/"
                f"{metrics['test']['roc_auc']:.3f}" if save else
                f"ROC-AUC val {metrics['val']['roc_auc']:.3f} (test hidden on a trial run)")
        print(f" cox_ph: C-index train {metrics['train']['c_index']:.3f} | "
              f"val {metrics['val']['c_index']:.3f} | {tail} ({len(cols)} features)")
        return _finish(out, target, {target: _plain(s)}, {target: _plain(spec)}, tuned,
                       {target: cph}, save)
    except Exception as exc:
        print(f" cox: FAILED — {type(exc).__name__}: {exc}")
        return {"name": "cox_ph", "kind": "survival", "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}"}


# =========================================================================== #
# Sequence models (LSTM / Transformer, torch optional)
# =========================================================================== #

def make_sequences(X, lookback):
    """Sliding windows over the full timeline: sequence i ends at row i (features
    up to and including row i — the target is already forward-looking)."""
    return np.stack([X[i - lookback + 1:i + 1] for i in range(lookback - 1, len(X))])


def _torch():
    """Import torch (optional dependency) the safe way for this machine."""
    # macOS: xgboost/lightgbm already loaded Homebrew's libomp, and torch ships its own —
    # the duo deadlocks on the first tensor op unless the dup is allowed and torch is single-threaded.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import torch
    torch.set_num_threads(1)
    return torch


# schedulers that must step every BATCH; this loop steps once per epoch, so they would misbehave
_PER_BATCH_SCHEDULERS = ("OneCycleLR", "CyclicLR")


def _check_sequence(s):
    """Stop before training when a sequence setting can't work: unknown optimizer, scheduler or
    loss, a per-batch scheduler, or a plateau scheduler that early stopping would always beat."""
    torch = _torch()
    opt = getattr(torch.optim, str(s["optimizer"]), None)
    if not (isinstance(opt, type) and issubclass(opt, torch.optim.Optimizer)):
        raise ValueError(f"optimizer={s['optimizer']!r} is not in torch.optim (e.g. 'Adam', 'AdamW', 'SGD')")
    if s["scheduler"] is not None:
        sched = getattr(torch.optim.lr_scheduler, str(s["scheduler"]), None)
        if not isinstance(sched, type) or s["scheduler"] in _PER_BATCH_SCHEDULERS:
            raise ValueError(f"scheduler={s['scheduler']!r}: use a per-epoch torch.optim.lr_scheduler "
                             f"(e.g. 'ReduceLROnPlateau', 'CosineAnnealingLR', 'StepLR'), "
                             f"not {_PER_BATCH_SCHEDULERS}")
        if s["scheduler"] == "ReduceLROnPlateau":
            # both watch the same val loss: a patience at or above early stopping's never fires
            sp = s["scheduler_params"].get("patience", 10)
            if sp >= s["patience"]:
                raise ValueError(f"ReduceLROnPlateau patience {sp} must be below the early-stop "
                                 f"patience {s['patience']}, e.g. scheduler_params={{'patience': 2}}")
    if s["loss"] not in SEQUENCE_LOSSES:
        raise ValueError(f"loss={s['loss']!r}: use one of {SEQUENCE_LOSSES}")
    if s.get("stop", "val") not in ("val", "train_tail"):
        raise ValueError(f"stop={s['stop']!r}: use 'val' or 'train_tail'")
    if s.get("pool", "last") not in ("last", "mean"):
        raise ValueError(f"pool={s['pool']!r}: use 'last' or 'mean'")
    allowed = {"bce": {"smoothing"}, "focal": {"gamma"}, "pairwise": set()}[s["loss"]]
    bad = set(s["loss_params"]) - allowed
    if bad:
        raise ValueError(f"loss_params {sorted(bad)} not used by loss={s['loss']!r} (takes {sorted(allowed) or 'none'})")
    smoothing = s["loss_params"].get("smoothing", 0.0)
    if not 0.0 <= float(smoothing) < 1.0:
        raise ValueError(f"loss_params smoothing={smoothing!r}: use a share in [0, 1)")
    # round two (2026-11-16): weight averaging, input noise, what early stopping watches
    ema = s.get("ema")
    if ema is not None and not 0.0 < float(ema) < 1.0:
        raise ValueError(f"ema={ema!r}: a decay per batch in (0, 1), e.g. 0.99, or None")
    if float(s.get("input_noise", 0.0)) < 0.0:
        raise ValueError(f"input_noise={s['input_noise']!r}: a standard deviation, 0 or more")
    if s.get("stop_metric", "loss") not in ("loss", "auc"):
        raise ValueError(f"stop_metric={s['stop_metric']!r}: use 'loss' or 'auc'")


def _build_net(arch, n_features, s):
    """The two sequence networks, built from their settings only, so a saved state_dict can be
    rebuilt from the same settings (save_fitted stores them)."""
    torch = _torch()
    nn = torch.nn

    class _LSTMNet(nn.Module):
        def __init__(self):
            super().__init__()
            # n_layers > 1 stacks LSTMs; torch applies dropout between the stacked layers only
            layers = s.get("n_layers", 1)
            self.lstm = nn.LSTM(n_features, s["hidden"], num_layers=layers, batch_first=True,
                                dropout=s["dropout"] if layers > 1 else 0.0)
            # nn.LSTM's own dropout only acts between stacked layers; one layer needs its own
            self.drop = nn.Dropout(s["dropout"])
            self.head = nn.Linear(s["hidden"], 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(self.drop(out[:, -1, :])).squeeze(-1)

    class _TransformerNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(n_features, s["d_model"])
            self.pos = nn.Parameter(torch.zeros(1, s["lookback"], s["d_model"]))
            layer = nn.TransformerEncoderLayer(
                d_model=s["d_model"], nhead=s["n_heads"], dim_feedforward=s["d_model"] * s.get("ff_mult", 2),
                dropout=s["dropout"], batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=s["n_layers"])
            self.head = nn.Linear(s["d_model"], 1)

        def forward(self, x):
            z = self.encoder(self.proj(x) + self.pos)
            # pool: "last" reads the newest position (as before), "mean" averages the window
            z = z.mean(dim=1) if s.get("pool", "last") == "mean" else z[:, -1, :]
            return self.head(z).squeeze(-1)

    return _LSTMNet() if arch == "lstm" else _TransformerNet()


def _loss_fn(s, y_fit):
    """BCE with pos_weight = n_neg / n_pos of the fit rows (as before), or focal on top of it:
    focal scales each row's BCE by (1 - p_t)^gamma, so rows the model already gets right count less."""
    torch = _torch()
    pos = float(y_fit.sum())
    pos_weight = torch.tensor((len(y_fit) - pos) / pos if pos else 1.0)
    if s["loss"] == "pairwise":
        # an AUC surrogate: softplus(-(score of a positive - score of a negative)) over every positive-negative
        # pair of the batch. It cares only about the ORDER of the scores, which is what ROC-AUC measures, so the
        # class weight does not enter. A batch without both classes has no pair: a zero that keeps the graph.
        def pairwise(logits, y):
            hi, lo = logits[y > 0.5], logits[y <= 0.5]
            if len(hi) == 0 or len(lo) == 0:
                return logits.sum() * 0.0
            return torch.nn.functional.softplus(-(hi[:, None] - lo[None, :])).mean()
        return pairwise
    if s["loss"] == "bce":
        eps = float(s["loss_params"].get("smoothing", 0.0))
        if not eps:
            return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        def smoothed(logits, y):
            # label smoothing: a 1 becomes 1 - eps / 2 and a 0 becomes eps / 2
            return torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y * (1 - eps) + 0.5 * eps, pos_weight=pos_weight)
        return smoothed
    gamma = float(s["loss_params"].get("gamma", 2.0))

    def focal(logits, y):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y, pos_weight=pos_weight, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * y + (1 - p) * (1 - y)
        return ((1 - p_t) ** gamma * bce).mean()
    return focal


def _logits(net, x, chunk=1024):
    """Forward pass in chunks, so scoring every window at once cannot exhaust memory."""
    torch = _torch()
    return torch.cat([net(x[i:i + chunk]) for i in range(0, len(x), chunk)])


def _train_net(arch, n_features, x_fit, y_fit, x_stop, y_stop, s, seed=RANDOM_STATE):
    """Seeded fit with early stopping on (x_stop, y_stop): val for the real fit, a slice of the
    fold's own train part inside CV. Optimizer and scheduler come by name from torch.
    Warmup: epochs 1..W run at lr * epoch / W, then the plain lr; the scheduler steps only after.
    Returns: (net at its best epoch, {history, best_epoch, stop_epoch}); history has one row per
    epoch: train_loss, val_loss (the stopping loss), lr."""
    torch = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = _build_net(arch, n_features, s)
    loss_fn = _loss_fn(s, y_fit)
    opt = getattr(torch.optim, s["optimizer"])(net.parameters(), lr=s["lr"], **s["optimizer_params"])
    sched = (getattr(torch.optim.lr_scheduler, s["scheduler"])(opt, **s["scheduler_params"])
             if s["scheduler"] else None)
    plateau = isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)
    gen = torch.Generator().manual_seed(seed)
    warm, clip = s["warmup_epochs"], s.get("clip_norm")
    # round two (2026-11-16), all off by default: weight averaging (the averaged net is the one that is stopped
    # on and returned), Gaussian noise on the training windows, and early stopping on the stop rows' AUC
    decay, noise = s.get("ema"), float(s.get("input_noise", 0.0))
    watch_auc = s.get("stop_metric", "loss") == "auc"
    ema = (torch.optim.swa_utils.AveragedModel(
        net, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(float(decay))) if decay else None)
    judged = ema.module if ema is not None else net

    best_val, best_state, stale, best_epoch, history = float("inf"), None, 0, 0, []
    for epoch in range(1, s["epochs"] + 1):
        if warm and epoch <= warm + 1:
            for g in opt.param_groups:
                g["lr"] = s["lr"] * min(epoch / warm, 1.0)
        lr_now = opt.param_groups[0]["lr"]
        net.train()
        total, n = 0.0, 0
        for batch in torch.randperm(len(x_fit), generator=gen).split(s["batch_size"]):
            opt.zero_grad()
            xb = x_fit[batch]
            if noise:
                xb = xb + noise * torch.randn(xb.shape, generator=gen)
            loss = loss_fn(net(xb), y_fit[batch])
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            if ema is not None:
                ema.update_parameters(net)
            total, n = total + float(loss) * len(batch), n + len(batch)
        judged.eval()
        with torch.no_grad():
            stop_logits = _logits(judged, x_stop)
            val_loss = float(loss_fn(stop_logits, y_stop))
        watch = val_loss                                  # lower is better: the stop rows' loss, or 1 - their AUC
        if watch_auc and len(torch.unique(y_stop)) > 1:
            from sklearn.metrics import roc_auc_score
            watch = 1.0 - float(roc_auc_score(y_stop.numpy(), stop_logits.numpy()))
        history.append({"epoch": epoch, "train_loss": total / max(n, 1), "val_loss": val_loss,
                        "lr": lr_now})
        if watch_auc:
            history[-1]["val_auc"] = 1.0 - watch
        if sched is not None and epoch > warm:
            sched.step(watch) if plateau else sched.step()
        if watch < best_val - 1e-5:
            best_val, stale, best_epoch = watch, 0, epoch
            best_state = {k: v.clone() for k, v in judged.state_dict().items()}
        else:
            stale += 1
            if stale >= s["patience"]:
                break
    if best_state is not None:
        judged.load_state_dict(best_state)
    return judged, {"history": history, "best_epoch": best_epoch, "stop_epoch": len(history)}


def _train_seeds(arch, n_features, x_fit, y_fit, x_stop, y_stop, s, seed=RANDOM_STATE):
    """One net per seed (seed, seed + 1, ...): n_seeds, 1 unless the notebook names it.
    Returns: [(net, run), ...]."""
    return [_train_net(arch, n_features, x_fit, y_fit, x_stop, y_stop, s, seed=seed + i)
            for i in range(int(s.get("n_seeds", 1)))]


def _mean_probs(nets, x):
    """The seed-averaged probability of every window (one net = that net's probability)."""
    torch = _torch()
    with torch.no_grad():
        return np.mean([torch.sigmoid(_logits(net.eval(), x)).numpy() for net, _ in nets], axis=0)


def _tail_split(rows, stamps, horizon, stop_frac=0.15):
    """Early-stopping rows cut from the END of a block of train rows: the last stop_frac of them,
    with the fit rows ending one horizon before (the same embargo rule as the folds).
    Returns: (fit rows, stop rows)."""
    cut = int(len(rows) * (1 - stop_frac))
    fit, stop = rows[:cut], rows[cut:]
    return fit[stamps[fit] + horizon < stamps[stop[0]]], stop


def _sequence_cv(data, arch, target, s, stop_frac=0.15, cols=None, parts=None, seed=RANDOM_STATE):
    """Walk-forward CV AUC for a sequence model, on TRAIN rows only.
    Per fold: the scaler is re-fit on the fold's fit rows, windows are built over the train rows and
    a window belongs to the fold part of its END row. Early stopping uses the last 15% of the fold's
    train part, cut one horizon before it (the same embargo rule as the folds) — never val and
    never the fold's test part.
    cols: feature subset (default every feature). parts: a list that collects each fold's
    (fold number, test rows, scores) for the out-of-fold frame."""
    torch = _torch()
    train_t, cols = data["frames_t"][target]["train"], cols or data["feature_cols"]
    y = data["y"][target]["train"]
    stamps = pd.DatetimeIndex(time_of(train_t))
    h, L = data["label_history"][target]["horizon"], s["lookback"]
    count = iter(range(1, 10 ** 6))

    def fold(tr, te):
        k = next(count)
        tr, te = tr[tr >= L - 1], te[te >= L - 1]           # rows with a full window
        if len(tr) < 2 or len(te) == 0:
            return {"main": float("nan"), "pr_auc": float("nan")}
        fit, stop = _tail_split(tr, stamps, h, stop_frac)
        if len(fit) == 0 or len(np.unique(y[fit])) < 2 or len(np.unique(y[te])) < 2:
            return {"main": float("nan"), "pr_auc": float("nan")}
        params = fit_scaler(train_t.iloc[fit], cols)
        X = apply_scaler(train_t, params, cols, standardize=True).to_numpy()
        seqs = torch.tensor(make_sequences(X, L), dtype=torch.float32)
        yt = torch.tensor(y.astype(np.float32))
        w = lambda rows: rows - (L - 1)                      # row -> index of the window ending there
        nets = _train_seeds(arch, X.shape[1], seqs[w(fit)], yt[fit], seqs[w(stop)], yt[stop], s, seed)
        probs = _mean_probs(nets, seqs[w(te)])
        if parts is not None:
            parts.append((k, te, probs))
        return fold_auc(y[te], probs)
    return cross_val_custom(data, target, fold)


def _sequence_fit(data, arch, target, s, cols=None):
    """The real fit for one horizon: windows over train + val + test, fit on train windows.
    Early stopping reads val windows (stop="val", as before) or the embargoed tail of train
    (stop="train_tail": then val takes no part in training, only in the F1 threshold).
    Returns: scores and labels per split, the net (a list of nets with n_seeds > 1) and its history."""
    torch = _torch()
    sizes = {sp: len(data["times"][target][sp]) for sp in ("train", "val", "test")}
    X_all = data["X"][target]["scaled"]
    X_all = _select(X_all, data["feature_cols"], cols) if cols else X_all
    X_full = np.vstack([X_all[sp] for sp in ("train", "val", "test")])
    y_full = np.concatenate([data["y"][target][sp] for sp in ("train", "val", "test")])
    L = s["lookback"]
    ends = np.arange(L - 1, len(X_full))
    split_of = np.where(ends < sizes["train"], "train",
                        np.where(ends < sizes["train"] + sizes["val"], "val", "test"))
    xt = torch.tensor(make_sequences(X_full, L), dtype=torch.float32)
    yt = torch.tensor(y_full[ends], dtype=torch.float32)
    masks = {sp: torch.tensor(split_of == sp) for sp in ("train", "val", "test")}

    if s.get("stop", "val") == "train_tail":
        stamps = pd.DatetimeIndex(time_of(data["frames_t"][target]["train"]))
        rows = np.arange(L - 1, sizes["train"])
        fit, stop = _tail_split(rows, stamps, data["label_history"][target]["horizon"])
        w = lambda r: torch.tensor(r - (L - 1))
        nets = _train_seeds(arch, X_full.shape[1], xt[w(fit)], yt[w(fit)], xt[w(stop)], yt[w(stop)], s)
    else:
        nets = _train_seeds(arch, X_full.shape[1], xt[masks["train"]], yt[masks["train"]],
                            xt[masks["val"]], yt[masks["val"]], s)
    probs = _mean_probs(nets, xt)
    # ⚠️ the first lookback-1 train rows have no window; PRI treats them as unscored
    scores = {sp: probs[split_of == sp] for sp in ("train", "val", "test")}
    ys = {sp: y_full[ends][split_of == sp] for sp in ("train", "val", "test")}
    net = nets[0][0] if len(nets) == 1 else [n for n, _ in nets]
    return {"net": net, "run": nets[0][1], "runs": [r for _, r in nets], "scores": scores, "ys": ys}


def run_lstm(data, targets=None, per_horizon=None, tune=False, tune_per_horizon=False, grid=None,
             select=None, save=True, *, features=None, oof=False, **settings):
    """Single-layer LSTM classifier on 7-day feature windows (84 rows at 2h; torch, optional).
    settings: any key of model_zoo.RUNNER_SPECS["lstm"] — lookback, hidden, dropout, epochs,
    batch_size, patience, lr, optimizer, optimizer_params, scheduler, scheduler_params,
    warmup_epochs, loss, loss_params; or of model_zoo.RUNNER_OPTIONAL["lstm"] — n_seeds, stop,
    clip_norm, n_layers (off unless named). The same on every horizon.
    Arguments: see _run_sequence_model."""
    return _run_sequence_model(data, "lstm", targets, per_horizon, tune, tune_per_horizon,
                               grid, select, save, settings, features, oof)


def run_transformer(data, targets=None, per_horizon=None, tune=False, tune_per_horizon=False,
                    grid=None, select=None, save=True, *, features=None, oof=False, **settings):
    """Temporal Transformer classifier on 7-day windows (84 rows at 2h; torch, optional).
    Learned input projection + additive positional embedding -> small TransformerEncoder.
    The last position's representation feeds the classification head.
    settings: any key of model_zoo.RUNNER_SPECS["transformer"] (d_model, n_heads, n_layers + the
    LSTM's training keys) or of RUNNER_OPTIONAL["transformer"] (n_seeds, stop, clip_norm, pool,
    ff_mult; off unless named). Arguments: see _run_sequence_model."""
    return _run_sequence_model(data, "transformer", targets, per_horizon, tune, tune_per_horizon,
                               grid, select, save, settings, features, oof)


def _run_sequence_model(data, arch, targets, per_horizon, tune, tune_per_horizon, grid, select,
                        save, settings, features=None, oof=False):
    """Shared runner for the sequence classifiers (LSTM / Transformer).
    targets: horizons to train, each on its own rows — defaults to the shortest only (as before).
    per_horizon: extra settings for one horizon on top of `settings`, e.g. {"7d": {"epochs": 60}}.
    tune: True picks from the grid (default model_zoo.RUNNER_GRIDS) on walk-forward CV, on the
    first target, or on every target with tune_per_horizon. Off by default: it trains the network
    folds x configs more times.
    features: column subset (default every feature); per horizon as {"24h": [...], "7d": [...]}.
    oof: True adds out["oof"], the out-of-fold scores of every horizon's final config.
    Windows only look BACK; each sequence belongs to the split of its END row. Mini-batches; early
    stop on val loss (or on the tail of train with stop="train_tail").
    Returns: result dict, or a skipped-result dict when torch is missing."""

    # a window crossing a split boundary is safe — the embargo gap just shows up as a jump
    if not TORCH_AVAILABLE:
        print(f" {arch}: SKIPPED — torch not installed")
        return {"name": arch, "kind": "classifier", "status": "skipped",
                "reason": "torch not installed"}
    targets = ([data["stress_targets"][0]] if targets is None
               else [targets] if isinstance(targets, str) else list(targets))
    primary = targets[0]
    spec = RUNNER_SPECS[arch]
    extra_by_t = _per_horizon(per_horizon, targets)
    feats_by_t = _features_by_horizon(features, targets, data["feature_cols"])
    grid = RUNNER_GRIDS.get(arch) if grid is None else grid
    check_names(arch, "runner", settings, *extra_by_t.values(), *expand_grid(grid))
    for t in targets:
        for point in expand_grid(grid if tune else None):
            _check_sequence({**spec, **point, **settings, **extra_by_t[t]})
    select = select or default_select(arch)

    tuned = {}
    for t in ((targets if tune_per_horizon else [primary]) if tune else []):
        tuned[t] = run_tuning(
            f"{arch} [{_horizon(t)}]", build_configs(grid, {**settings, **extra_by_t[t]}, base=spec),
            lambda s, t=t: _sequence_cv(data, arch, t, s, cols=feats_by_t[t]), select, higher_is_better=True,
            metric="auc", guard="pr_auc", simplicity=SIMPLICITY.get(arch))

    out = {"name": arch, "kind": "classifier", "status": "trained", "proper_probs": True,
           "select": select if tune else None, "loss": settings.get("loss", spec["loss"]),
           "features_used": feats_by_t[primary] or data["feature_cols"], "config_by_target": {},
           "thresholds_f1": {}, "targets": {}, "history": {}, "best_epoch": {}, "stop_epoch": {}}
    if isinstance(features, dict):
        out["features_by_target"] = {t: feats_by_t[t] or data["feature_cols"] for t in targets}
    oof_parts = []
    used, defaults, fitted = {}, {}, {}
    for t in targets:
        point = tuned.get(t, tuned.get(primary, {"grid_point": {}}))["grid_point"]
        s = {**spec, **point, **settings, **extra_by_t[t]}
        fit = _sequence_fit(data, arch, t, s, cols=feats_by_t[t])
        if oof:
            parts = []
            _sequence_cv(oof_data(data, oof), arch, t, s, cols=feats_by_t[t], parts=parts)
            oof_parts.append(oof_frame(data, t, parts))
        scores, ys = fit["scores"], fit["ys"]
        thr = best_f1_threshold(ys["val"], scores["val"])
        out["thresholds_f1"][t] = thr
        out["targets"][t] = {sp: evaluate_classification(ys[sp], scores[sp], threshold=thr)
                             for sp in ("train", "val", "test")}
        out["config_by_target"][t] = _plain(s)
        out["history"][t] = fit["run"]["history"]
        out["best_epoch"][t], out["stop_epoch"][t] = fit["run"]["best_epoch"], fit["run"]["stop_epoch"]
        used[t], defaults[t], fitted[t] = _plain(s), _plain(spec), fit["net"]
        if t == primary:
            out["lookback"] = s["lookback"]
            out["test_auc_ci"] = bootstrap_auc_ci(ys["test"], scores["test"])
            out["predictions"] = pd.concat([
                pd.DataFrame({TIME_COL: np.asarray(data["times"][t][sp])[-len(scores[sp]):]
                              if sp == "train" else data["times"][t][sp],
                              "split": sp, "y_true": ys[sp].astype(float),
                              "score": scores[sp].astype(float)})
                for sp in ("train", "val", "test")], ignore_index=True)
        m = out["targets"][t]
        # a trial run hides test, as run_classifier does
        shown = (f"train/val/test ROC-AUC {m['train']['roc_auc']:.3f}/{m['val']['roc_auc']:.3f}/"
                 f"{m['test']['roc_auc']:.3f}" if save else
                 f"train/val ROC-AUC {m['train']['roc_auc']:.3f}/{m['val']['roc_auc']:.3f} (test hidden on a trial run)")
        print(f" {arch} [{_horizon(t)}]: {shown} (lookback {s['lookback']} rows, "
              f"best epoch {out['best_epoch'][t]} of {out['stop_epoch'][t]})")
    if tuned:
        out["cv"] = tuned[primary]["best_cv"]
    if oof:
        out["oof"] = pd.concat(oof_parts, ignore_index=True)
    return _finish(out, primary, used, defaults, tuned, fitted, save)


def cv_sequence_sets(arch, data, sets, *, target=STRESS_TARGETS[0], **settings):
    """One LSTM / Transformer config on several feature sets, on the tuning folds (train only).
    sets: {label: [cols] or None for every feature}. settings: as run_lstm / run_transformer.
    Returns: model_evaluation.score_sets table (cv_auc per set, paired against the first)."""
    check_names(arch, "runner", settings)
    s = {**RUNNER_SPECS[arch], **settings}
    _check_sequence(s)
    return score_sets(sets, lambda cols: _sequence_cv(data, arch, target, s, cols=cols),
                      data["feature_cols"], label=f"{arch} [{_horizon(target)}]")


def sequence_seed_spread(arch, data, *, target=STRESS_TARGETS[0], features=None, seeds=(42, 43, 44), **settings):
    """The same network trained from several seeds, on the tuning folds: the noise floor every
    sequence comparison has to clear. Small nets on a few thousand rows move a lot with the seed.
    Returns: one row per seed — seed, cv_auc, fold_k; the spread is printed."""
    check_names(arch, "runner", settings)
    s = {**RUNNER_SPECS[arch], **settings, "n_seeds": 1}
    cols = check_features(features, data["feature_cols"]) if features is not None else None
    rows = []
    for seed in seeds:
        cv = _sequence_cv(data, arch, target, s, cols=cols, seed=seed)
        rows.append({"seed": seed, "cv_auc": cv["mean"], **{f"fold_{i}": f for i, f in enumerate(cv["folds"], 1)}})
    table = pd.DataFrame(rows)
    print(f" {arch} [{_horizon(target)}] over {len(seeds)} seed(s): CV {table['cv_auc'].mean():.3f}, "
          f"SD {table['cv_auc'].std(ddof=1):.4f}, range {table['cv_auc'].min():.3f}-{table['cv_auc'].max():.3f}")
    return table


def cv_cox_sets(data, sets, *, target=STRESS_TARGETS[0], **settings):
    """One Cox config on several feature sets, on the tuning folds (train only).
    sets: {label: [cols] or None for the fold's own top-N}. cv_auc here is the C-index; auc is the
    risk score against the stress label. Returns: model_evaluation.score_sets table + auc."""
    check_names("cox_ph", "runner", settings)
    s = {**RUNNER_SPECS["cox_ph"], **settings}
    aucs = []

    def score(cols):
        cv = _cox_cv(data, target, s, cols=cols)
        aucs.append(cv.get("auc_mean"))
        return cv
    table = score_sets(sets, score, data["feature_cols"], label=f"cox_ph [{_horizon(target)}] (C-index)")
    table.insert(3, "auc", aucs)
    return table.rename(columns={"cv_auc": "c_index"})


# =========================================================================== #
# PRI meta-learner
# =========================================================================== #

def _component_skill(res, target=STRESS_TARGETS[0]):
    """A component's skill for meta-weighting: mean of val AUC and train CV AUC.
    result dict -> survival components fall back to the val C-index.
    Returns: float. Averaging two estimates steadies a val AUC read off one market regime."""
    if res.get("status") != "trained":
        return None
    if res["kind"] == "survival":
        return res["metrics"]["val"]["c_index"]
    tgt = res.get("targets", {}).get(target)
    if not tgt:
        return None
    val_auc = tgt["val"]["roc_auc"]
    cv_mean = (res.get("cv") or {}).get("mean")
    if cv_mean is not None and not math.isnan(cv_mean):
        return float(np.mean([val_auc, cv_mean]))
    return val_auc


def _is_meta_component(res):
    """PRI components: trained non-regressor, non-baseline models with predictions."""
    # kind "meta" = a blend (the PRI itself, model_ensemble.run_ensemble): never a component of another
    return (res.get("kind") not in ("regressor", "meta") and "predictions" in res
            and not str(res.get("name", "")).startswith("baseline_")
            and res.get("name") != "meta_pri")


def fit_meta_weights(results, target=STRESS_TARGETS[0], floor=0.02, shrink=0.30):
    """Skill-proportional weights, floored and shrunk toward equal weights.
    w_i proportional to (1-shrink)*max(skill_i - 0.5, floor) + shrink*(1/n).
    Returns: {component: weight}."""

    # without the shrink, AUC noise on one val regime can zero most of the models
    raw = {}
    for res in results.values():
        if not _is_meta_component(res):
            continue
        skill = _component_skill(res, target)
        if skill is None or math.isnan(skill):
            continue
        raw[res["name"]] = max(skill - 0.5, floor)
    if not raw:
        return {}
    total = sum(raw.values())
    n = len(raw)
    if total <= 0:
        return {k: 1.0 / n for k in raw}
    return {k: (1 - shrink) * v / total + shrink / n for k, v in raw.items()}


def compute_pri(score_frame, weights):
    """PRI = 100 * weighted mean of the rank-normalized component columns.
    component frame, weights -> NaN components excluded per row, weights renormalized.
    Returns: the PRI Series."""
    cols = [c for c in weights if c in score_frame.columns]
    w = np.array([weights[c] for c in cols])
    vals = score_frame[cols].to_numpy(dtype=float)
    mask = ~np.isnan(vals)
    weighted = np.nansum(vals * w, axis=1)
    denom = (mask * w).sum(axis=1)
    return 100.0 * weighted / np.where(denom > 0, denom, np.nan)


def run_meta_pri(results, data, target=STRESS_TARGETS[0]):
    """The meta-learner: blend every trained component score into the PRI series.
    Components rank-normalized against their VAL values (out of sample: a model's train scores are
    in-sample, so an overfit model would rank almost every unseen row near the top), weighted by
    val skill, averaged to 0-100.
    Returns: result dict, evaluated like any classifier against the stress targets on test.
    component_hashes: each component's settings_hash, so the blend names the exact fits it read."""
    frame = None
    hashes = {}
    for res in results.values():
        if not _is_meta_component(res):
            continue
        if _component_skill(res, target) is None:
            continue
        hashes[res["name"]] = res.get("settings_hash")
        preds = res["predictions"]
        ref = preds.loc[preds["split"] == "val", "score"]
        col = pd.DataFrame({
            TIME_COL: preds[TIME_COL], "split": preds["split"],
            res["name"]: rank_normalize(preds["score"], ref)})
        frame = col if frame is None else frame.merge(
            col, on=[TIME_COL, "split"], how="outer")
    frame = frame.sort_values(TIME_COL, kind="stable").reset_index(drop=True)

    weights = fit_meta_weights(results, target)
    frame["pri"] = compute_pri(frame, weights)

    y_true = pd.concat([pd.DataFrame({TIME_COL: data["times"][target][s], "split": s,
                                      "y_true": data["y"][target][s]})
                        for s in ("train", "val", "test")], ignore_index=True)
    frame = frame.merge(y_true, on=[TIME_COL, "split"], how="left")

    out = {"name": "meta_pri", "kind": "meta", "status": "trained",
           "proper_probs": False, "weights": weights, "n_components": len(weights),
           "component_hashes": {k: v for k, v in hashes.items() if k in weights},
           "targets": {}}
    for t in data["stress_targets"]:
        # t != target: a (time, split) match only where the two horizons' own embargo
        # cuts happen to agree near a boundary — an approximate cross-check, not the
        # primary metric (that's `target`'s own row, evaluated exactly above).
        yt = pd.concat([pd.DataFrame({TIME_COL: data["times"][t][s], "split": s,
                                      "yy": data["y"][t][s]})
                        for s in ("train", "val", "test")], ignore_index=True)
        m = frame.merge(yt, on=[TIME_COL, "split"], how="left").dropna(subset=["pri", "yy"])
        out["targets"][t] = {
            s: evaluate_classification(m.loc[m["split"] == s, "yy"],
                                       m.loc[m["split"] == s, "pri"] / 100.0,
                                       proper_probs=False)
            for s in ("train", "val", "test")}
        if t == target:
            mt = m.loc[m["split"] == "test"]
            out["test_auc_ci"] = bootstrap_auc_ci(mt["yy"], mt["pri"])
    out["predictions"] = frame[[TIME_COL, "split", "y_true", "pri"]].rename(
        columns={"pri": "score"})
    out["pri_frame"] = frame
    m = out["targets"][target]["test"]
    print(f" meta_pri: {len(weights)} components | test ROC-AUC {m['roc_auc']:.3f} | "
          f"PR-AUC {m['pr_auc']:.3f}")
    print(" weights:", {k: round(v, 3) for k, v in
                        sorted(weights.items(), key=lambda kv: -kv[1])})
    return out
