"""Modeling stage — series models: SARIMAX and the HMM regime model.

They read a time-ordered series, not only a feature matrix, so they live outside the registry.
Both follow the series contract: fit on train rows only, then move through later rows with the
fitted parameters fixed, each row scored from data that was public at that row.
statsmodels and hmmlearn are optional: without them the runner returns a skipped result.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import warnings

import numpy as np
import pandas as pd

from model_config import RANDOM_STATE, STRESS_TARGETS, TIME_COL
from model_zoo import RUNNER_SPECS, RUNNER_GRIDS, SIMPLICITY, check_names, default_select, expand_grid, _plain
from model_evaluation import (best_f1_threshold, evaluate_classification, bootstrap_auc_ci, build_configs,
                              cross_val_custom, fold_auc, oof_frame, oof_data, score_sets, _horizon)
from model_evaluation import tune as run_tuning
from model_split import fit_scaler, apply_scaler, time_of
from model_training import _finish, _prediction_frame, _features_by_horizon

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
try:
    from hmmlearn.hmm import GaussianHMM, GMMHMM
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False

logging.getLogger("hmmlearn").setLevel(logging.ERROR)      # EM's 'not converging' notes: read monitor_ instead
STEP = pd.Timedelta("2h")                     # the row grain of the panel


# =========================================================================== #
# Shared: rows in time order, and the label history that is public at a row
# =========================================================================== #

def _stamps(times):
    """Row stamps (strings like '2024-04-08 02:00:00.000 UTC') -> a UTC DatetimeIndex."""
    text = pd.Series(list(times)).astype(str).str.replace(" UTC", "", regex=False)
    return pd.DatetimeIndex(pd.to_datetime(text, utc=True))


def known_label(data, target, stamps):
    """The label that became public exactly at each stamp: the one stamped one horizon earlier.
    A label stamped s covers (s, s + h], so it is known at s + h and not before (the same as-of
    rule as the persistence baseline). NaN where no label has that stamp.
    Returns: float array, one value per stamp."""
    hist = data["label_history"][target]
    labels = hist["labels"].astype(float)
    return labels.reindex(pd.DatetimeIndex(stamps) - hist["horizon"]).to_numpy()


def _blocks(data, target, splits):
    """Rows of the given splits as one time-ordered block. Returns: (frame, y, stamps, split names)."""
    frames = [data["frames_t"][target][s] for s in splits]
    frame = pd.concat(frames, ignore_index=True)
    y = np.concatenate([data["y"][target][s] for s in splits])
    names = np.concatenate([np.full(len(data["y"][target][s]), s) for s in splits])
    return frame, y, _stamps(time_of(frame)), names


# =========================================================================== #
# SARIMAX: forecast the label one horizon ahead from its public history (+ features)
# =========================================================================== #

def _sarimax_scores(frame, stamps, known, cols, fit_rows, s, steps):
    """Fit on fit_rows, then score every row of the block with the parameters fixed.
    The series z is the public label history: z at stamp t is the label of the window that just
    closed at t. The label of row t is z at t + horizon, so the score of row t is the forecast
    `steps` rows ahead, made from the state filtered at t (data up to t only), plus the row's own
    features times their coefficients (the features sit `steps` rows back in the model, so the
    forecast needs no future feature).
    Returns: (scores for every row, info dict)."""
    grid = pd.date_range(stamps.min(), stamps.max() + steps * STEP, freq=STEP)
    pos = grid.get_indexer(stamps)
    params = fit_scaler(frame.iloc[fit_rows], cols) if cols else None
    X = apply_scaler(frame, params, cols, standardize=True).to_numpy() if cols else None

    z = np.full(len(grid), np.nan)
    z[pos] = known
    exog = None
    if cols:
        # features of row t explain z at t + steps; a grid stamp with no origin row has no observation
        exog = np.zeros((len(grid), len(cols)))
        tgt = pos + steps
        exog[tgt] = X
        seen = np.zeros(len(grid), dtype=bool)
        seen[tgt] = True
        z[~seen] = np.nan
    mean = float(np.nanmean(z[pos[fit_rows]]))
    z = z - mean                                             # no trend term: the series is centred on train

    last_fit = pos[fit_rows].max()
    # (P, D, Q) across windows: lag 1 of the seasonal part is one horizon back unless period is set
    seasonal = (*tuple(s["seasonal_order"])[:3], int(s["period"] or steps))
    kw = dict(order=tuple(s["order"]), seasonal_order=seasonal, trend="n",
              enforce_stationarity=s["enforce_stationarity"], enforce_invertibility=s["enforce_invertibility"],
              concentrate_scale=s["concentrate_scale"])
    if s.get("measurement_error"):                           # round two: off unless the notebook names it
        kw["measurement_error"] = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(z[:last_fit + 1], exog=None if exog is None else exog[:last_fit + 1], **kw)
        res = model.fit(method=s["method"], maxiter=s["maxiter"], disp=0)
        full = res.apply(z, exog=exog)                       # same parameters, no refit
    T = full.filter_results.transition[:, :, 0]
    Z = full.filter_results.design[:, :, 0]
    ahead = (Z @ np.linalg.matrix_power(T, steps) @ full.filter_results.filtered_state[:, pos]).ravel()
    if cols:
        beta = np.array([res.params[i] for i, n in enumerate(model.param_names) if n in model.exog_names])
        ahead = ahead + X @ beta
    info = {"aic": float(res.aic), "bic": float(res.bic), "llf": float(res.llf),
            "converged": bool(res.mle_retvals.get("converged", True)) if res.mle_retvals else True,
            "iterations": int(res.mle_retvals.get("iterations", 0)) if res.mle_retvals else 0,
            "params": {n: float(v) for n, v in zip(model.param_names, res.params)}, "train_mean": mean}
    return ahead + mean, info


def _steps(data, target):
    return int(data["label_history"][target]["horizon"] / STEP)


def _sarimax_cv(data, target, s, cols=None, parts=None):
    """Walk-forward CV AUC on TRAIN rows: per fold, estimate on the fold's train part and score its
    test part with the parameters fixed. parts collects (fold, test rows, scores) for the OOF frame."""
    frame = data["frames_t"][target]["train"]
    y, stamps = data["y"][target]["train"], _stamps(time_of(frame))
    known = known_label(data, target, stamps)
    count = iter(range(1, 10 ** 6))

    def fold(tr, te):
        k = next(count)
        if len(np.unique(y[te])) < 2:
            return {"main": float("nan"), "pr_auc": float("nan")}
        scores, _ = _sarimax_scores(frame, stamps, known, cols, tr, s, _steps(data, target))
        if parts is not None:
            parts.append((k, te, scores[te]))
        return fold_auc(y[te], scores[te])
    return cross_val_custom(data, target, fold)


def run_sarimax(data, targets=None, *, tune=False, tune_per_horizon=False, grid=None, select=None,
                features=None, save=True, oof=False, **settings):
    """SARIMAX on the public label history (statsmodels, optional): one model per horizon.
    The series is the stress label that has become public (a label is known one horizon after its
    stamp); the model forecasts it one horizon ahead, which is the row's own label. features are
    exogenous regressors (default NONE: the series alone), each row's own values.
    settings: any key of model_zoo.RUNNER_SPECS["sarimax"] — order (p, d, q) row to row,
    seasonal_order (P, D, Q) horizon to horizon, period (None = the horizon in rows),
    enforce_stationarity, enforce_invertibility, concentrate_scale, method, maxiter.
    The label series is built from overlapping windows, so a row-to-row AR term reads that overlap
    as persistence and over-carries it one horizon ahead; the seasonal part steps a full window back.
    targets: horizons to run (default the shortest). tune / tune_per_horizon / grid / select /
    save / oof: as in run_classifier; the grid's CV estimates every config on every fold.
    Returns: result dict (kind "classifier", family "series"), or a skipped result without statsmodels."""
    if not STATSMODELS_AVAILABLE:
        print(" sarimax: SKIPPED — statsmodels not installed")
        return {"name": "sarimax", "kind": "classifier", "status": "skipped", "reason": "statsmodels not installed"}
    return _run_series("sarimax", _sarimax_cv, _sarimax_fit, data, targets, tune, tune_per_horizon, grid,
                       select, features, save, oof, settings)


def _sarimax_fit(data, target, s, cols):
    """The real fit: estimate on train, then filter through val and test with the parameters fixed."""
    frame, y, stamps, names = _blocks(data, target, ("train", "val", "test"))
    known = known_label(data, target, stamps)
    scores, info = _sarimax_scores(frame, stamps, known, cols, np.flatnonzero(names == "train"), s,
                                   _steps(data, target))
    return {"scores": {sp: scores[names == sp] for sp in ("train", "val", "test")}, "info": info,
            "fitted": {"settings": _plain(s), "features": cols, **info}}


# =========================================================================== #
# HMM regime model: filtered state probabilities -> stress rate of each state
# =========================================================================== #

def forward_filter(model, X):
    """P(state at t | observations up to t) for every row: the forward algorithm, nothing else.
    hmmlearn's predict_proba is the SMOOTHED probability (it also reads later rows) and Viterbi
    reads the whole path, so neither may score a row. Appending rows never changes earlier rows here.
    Returns: array (n_rows, n_states)."""
    from scipy.special import logsumexp
    from scipy.stats import multivariate_normal
    if hasattr(model, "n_mix"):
        # a Gaussian-mixture HMM (emission="gmm"): the model's own per-state likelihood of every row
        logb = np.asarray(model._compute_log_likelihood(X), dtype=float)
    else:
        logb = np.column_stack([multivariate_normal.logpdf(X, model.means_[k], model.covars_[k], allow_singular=True)
                                for k in range(model.n_components)]).reshape(len(X), -1)
    with np.errstate(divide="ignore"):
        log_start, log_trans = np.log(model.startprob_), np.log(model.transmat_)
    out = np.empty_like(logb)
    alpha = log_start + logb[0]
    out[0] = alpha - logsumexp(alpha)
    for t in range(1, len(X)):
        alpha = logsumexp(out[t - 1][:, None] + log_trans, axis=0) + logb[t]
        out[t] = alpha - logsumexp(alpha)
    return np.exp(out)


def _make_hmm(s, seed):
    """The HMM of one EM start: GaussianHMM (the house model) or, with emission="gmm", GMMHMM with n_mix
    Gaussians per state. The priors (transmat_prior, startprob_prior, covars_prior, covars_weight) are
    hmmlearn's own unless the notebook names them (round two, 2026-11-16)."""
    kw = dict(n_components=s["n_components"], covariance_type=s["covariance_type"], n_iter=s["n_iter"],
              tol=s["tol"], min_covar=s["min_covar"], random_state=seed)
    kw.update({k: s[k] for k in ("transmat_prior", "startprob_prior", "covars_prior", "covars_weight")
               if s.get(k) is not None})
    if s.get("emission", "gaussian") == "gmm":
        return GMMHMM(n_mix=int(s.get("n_mix", 2)), **kw)
    if s.get("emission", "gaussian") != "gaussian":
        raise ValueError(f"emission={s['emission']!r}: use 'gaussian' or 'gmm'")
    return GaussianHMM(**kw)


def _hmm_scores(X, y, fit_rows, s):
    """Fit on fit_rows (EM, the best of n_init seeded starts), filter every row forward, and score
    each row as the stress rate it expects: sum over states of P(state | past) x that state's
    stress rate on the fit rows. States are unsupervised; the labels only name them.
    Returns: (scores for every row, info dict)."""
    best = None
    for i in range(int(s["n_init"])):
        m = _make_hmm(s, RANDOM_STATE + i)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X[fit_rows])
            ll = m.score(X[fit_rows])
        if best is None or ll > best[1]:
            best = (m, ll)
    model, ll = best
    post = forward_filter(model, X)
    w = post[fit_rows]
    rate = (w * y[fit_rows][:, None]).sum(axis=0) / np.maximum(w.sum(axis=0), 1e-9)
    order = np.argsort(rate)                                 # states named by their stress rate: low -> high
    info = {"loglik": float(ll), "converged": bool(model.monitor_.converged),
            "em_iterations": int(model.monitor_.iter), "loglik_history": [float(v) for v in model.monitor_.history],
            "state_stress_rate": [float(rate[k]) for k in order],
            "state_share": [float(w[:, k].mean()) for k in order],
            "stay_probability": [float(model.transmat_[k, k]) for k in order]}
    return post @ rate, info, model, rate


def _hmm_matrix(frame, cols, fit_rows):
    params = fit_scaler(frame.iloc[fit_rows], cols)
    return apply_scaler(frame, params, cols, standardize=True).to_numpy()


def _hmm_cv(data, target, s, cols=None, parts=None):
    """Walk-forward CV AUC on TRAIN rows: per fold, EM on the fold's train part, then filter forward
    through its test part with the parameters fixed."""
    frame, cols = data["frames_t"][target]["train"], cols or data["feature_cols"]
    y = data["y"][target]["train"]
    count = iter(range(1, 10 ** 6))

    def fold(tr, te):
        k = next(count)
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            return {"main": float("nan"), "pr_auc": float("nan")}
        rows = np.concatenate([tr, te])                      # in time order: train part, then test part
        X = _hmm_matrix(frame.iloc[rows].reset_index(drop=True), cols, np.arange(len(tr)))
        scores, _, _, _ = _hmm_scores(X, y[rows], np.arange(len(tr)), s)
        if parts is not None:
            parts.append((k, te, scores[len(tr):]))
        return fold_auc(y[te], scores[len(tr):])
    return cross_val_custom(data, target, fold)


def _hmm_fit(data, target, s, cols):
    """The real fit: EM on train, then filter forward through val and test."""
    cols = cols or data["feature_cols"]
    frame, y, _, names = _blocks(data, target, ("train", "val", "test"))
    fit_rows = np.flatnonzero(names == "train")
    X = _hmm_matrix(frame, cols, fit_rows)
    scores, info, model, rate = _hmm_scores(X, y, fit_rows, s)
    return {"scores": {sp: scores[names == sp] for sp in ("train", "val", "test")}, "info": info,
            "fitted": {"model": model, "state_stress_rate": rate, "features": cols}}


def run_hmm(data, targets=None, *, tune=False, tune_per_horizon=False, grid=None, select=None,
            features=None, save=True, oof=False, **settings):
    """Gaussian HMM regime model (hmmlearn, optional): one model per horizon.
    Unsupervised states on a few observables, FILTERED forward (forward_filter); a row's score is the
    stress rate its state probabilities expect. features: the observables (default every feature;
    a handful works better). settings: any key of model_zoo.RUNNER_SPECS["hmm_regime"] —
    n_components, covariance_type, n_iter, tol, n_init, min_covar.
    Other arguments: as run_sarimax.
    Returns: result dict (kind "classifier", family "series"), or a skipped result without hmmlearn."""
    if not HMMLEARN_AVAILABLE:
        print(" hmm_regime: SKIPPED — hmmlearn not installed")
        return {"name": "hmm_regime", "kind": "classifier", "status": "skipped", "reason": "hmmlearn not installed"}
    return _run_series("hmm_regime", _hmm_cv, _hmm_fit, data, targets, tune, tune_per_horizon, grid,
                       select, features, save, oof, settings)


def cv_series_sets(name, data, sets, *, target=STRESS_TARGETS[0], **settings):
    """One SARIMAX / HMM config on several feature sets, on the tuning folds (train only).
    name: "sarimax" (sets are exogenous regressors; None = the series alone) or "hmm_regime" (sets are
    the observables; None = every feature). Returns: model_evaluation.score_sets table."""
    check_names(name, "runner", settings)
    s = {**RUNNER_SPECS[name], **settings}
    cv_fn = {"sarimax": _sarimax_cv, "hmm_regime": _hmm_cv}[name]
    return score_sets(sets, lambda cols: cv_fn(data, target, s, cols=cols), data["feature_cols"],
                      label=f"{name} [{_horizon(target)}]")


# =========================================================================== #
# The shared runner
# =========================================================================== #

def _run_series(name, cv_fn, fit_fn, data, targets, tune, tune_per_horizon, grid, select, features,
                save, oof, settings):
    """Tune on walk-forward CV (optional), fit each horizon, report train / val / test like a classifier."""
    targets = ([data["stress_targets"][0]] if targets is None
               else [targets] if isinstance(targets, str) else list(targets))
    primary = targets[0]
    spec = RUNNER_SPECS[name]
    feats_by_t = _features_by_horizon(features, targets, data["feature_cols"])
    grid = RUNNER_GRIDS.get(name) if grid is None else grid
    check_names(name, "runner", settings, *expand_grid(grid if tune else None))
    select = select or default_select(name)

    tuned = {}
    for t in (targets if tune_per_horizon else [primary]):
        configs = build_configs(grid if tune else None, settings, base=spec)
        tuned[t] = run_tuning(f"{name} [{_horizon(t)}]", configs,
                              lambda s, t=t: cv_fn(data, t, s, cols=feats_by_t[t]), select,
                              higher_is_better=True, metric="auc", guard="pr_auc",
                              simplicity=SIMPLICITY.get(name))

    out = {"name": name, "kind": "classifier", "family": "series", "status": "trained",
           "proper_probs": False, "select": select if tune else None,
           "features_used": feats_by_t[primary] or ([] if name == "sarimax" else data["feature_cols"]),
           "config_by_target": {}, "thresholds_f1": {}, "targets": {}, "diagnostics": {}}
    if isinstance(features, dict):
        out["features_by_target"] = {t: feats_by_t[t] or [] for t in targets}
    used, defaults, fitted, oof_parts = {}, {}, {}, []
    for t in targets:
        s = {**spec, **tuned.get(t, tuned[primary])["grid_point"], **settings}
        fit = fit_fn(data, t, s, feats_by_t[t])
        scores, ys = fit["scores"], data["y"][t]
        thr = best_f1_threshold(ys["val"], scores["val"])
        out["thresholds_f1"][t] = thr
        out["targets"][t] = {sp: evaluate_classification(ys[sp], scores[sp], threshold=thr, proper_probs=False)
                             for sp in ("train", "val", "test")}
        out["config_by_target"][t] = _plain(s)
        out["diagnostics"][t] = fit["info"]
        used[t], defaults[t], fitted[t] = _plain(s), _plain(spec), fit["fitted"]
        if oof:
            parts = []
            cv_fn(oof_data(data, oof), t, s, cols=feats_by_t[t], parts=parts)
            oof_parts.append(oof_frame(data, t, parts))
        if t == primary:
            out["predictions"] = _prediction_frame(data, t, scores, ys)
            out["test_auc_ci"] = bootstrap_auc_ci(ys["test"], scores["test"])
        m = out["targets"][t]
        shown = (f"train/val/test ROC-AUC {m['train']['roc_auc']:.3f}/{m['val']['roc_auc']:.3f}/"
                 f"{m['test']['roc_auc']:.3f}" if save else
                 f"train/val ROC-AUC {m['train']['roc_auc']:.3f}/{m['val']['roc_auc']:.3f} (test hidden on a trial run)")
        print(f" {name} [{_horizon(t)}]: {shown}")
    out["cv"] = tuned[primary]["best_cv"]
    if oof:
        out["oof"] = pd.concat(oof_parts, ignore_index=True)
    return _finish(out, primary, used, defaults, tuned, fitted, save)
