"""Modeling stage — model runners (the training pass) + the PRI meta-learner.

One runner per model kind: tuned classifiers/regressors, the two baseline families,
IsolationForest, Cox PH survival, and the LSTM / Temporal Transformer sequence
classifiers — each reporting train/val/test with a bootstrap test-AUC CI. The
meta-learner blends every trained component's rank-normalized score (skill-shrunk
weights, baselines excluded) into the Protocol Risk Index. Orchestrated by
notebooks/model_training.ipynb.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import os
import math
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier

from model_config import (TIME_COL, STRESS_TARGETS, REG_TARGET, TARGET_BASE_COL,
                          RANDOM_STATE, LIFELINES_AVAILABLE, TORCH_AVAILABLE,
                          rank_normalize)
from model_zoo import (CLASSIFIER_SPECS, REGRESSOR_SPECS, build_model,
                       _xgb_overrides, predict_scores, score_is_probability)
from model_evaluation import (best_f1_threshold, evaluate_classification,
                              evaluate_regression, bootstrap_auc_ci,
                              cross_val_auc, tune_classifier, tune_regressor)


def _importances(model, columns):
    """Native feature importances as {column: weight} (trees) or |coef| (linear)."""
    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        vals = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
    else:
        return None
    return {c: float(v) for c, v in zip(columns, vals)}


def _prediction_frame(data, scores_by_split, y_by_split):
    """Tidy full-timeline prediction frame: time_bucket, split, y_true, score."""
    parts = []
    for s in ("train", "val", "test"):
        parts.append(pd.DataFrame({TIME_COL: data["times"][s], "split": s,
                                   "y_true": np.asarray(y_by_split[s]).astype(float),
                                   "score": np.asarray(scores_by_split[s], dtype=float)}))
    return pd.concat(parts, ignore_index=True)


def run_classifier(name, data, targets=STRESS_TARGETS, cv_target="y_stress_1d",
                   params=None, tune=True):
    """Tune (walk-forward, train-only), fit per stress horizon, report train/val/test.

    F1 is computed at the VAL-optimal threshold (stored per target, reused on test).
    Train metrics expose the overfit gap; the test ROC-AUC of cv_target gets a
    block-bootstrap CI. The cv_target full-timeline scores are the PRI component.
    """
    spec = CLASSIFIER_SPECS[name]
    xkey = "scaled" if spec["scaled"] else "raw"
    X = data["X"][xkey]
    tuned = tune_classifier(name, data, target=cv_target) if (tune and params is None) \
        else {"best_params": params or {}, "best_cv": float("nan"), "table": None}
    proper = None
    out = {"name": name, "kind": "classifier", "status": "trained",
           "scaled_input": spec["scaled"], "params": tuned["best_params"],
           "thresholds_f1": {}, "targets": {}}
    for t in targets:
        y_tr = data["y"][t]["train"]
        ov = {**(_xgb_overrides(y_tr) if name == "xgboost" else {}), **tuned["best_params"]}
        model = build_model(name, "classifier", **ov)
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
        if t == cv_target:
            out["importances"] = _importances(model, data["feature_cols"])
            out["predictions"] = _prediction_frame(data, scores, data["y"][t])
            out["test_auc_ci"] = bootstrap_auc_ci(data["y"][t]["test"], scores["test"])
    out["proper_probs"] = proper
    out["cv"] = cross_val_auc(name, data, target=cv_target, params=tuned["best_params"])
    m, ci = out["targets"][cv_target], out["test_auc_ci"]
    print(f" {name}: train/val/test ROC-AUC {m['train']['roc_auc']:.3f}/"
          f"{m['val']['roc_auc']:.3f}/{m['test']['roc_auc']:.3f} "
          f"[{ci['lo']:.3f},{ci['hi']:.3f}] | CV {out['cv']['mean']:.3f}"
          f"±{out['cv']['std']:.3f}")
    return out


def run_regressor(name, data, target=REG_TARGET, params=None, tune=True):
    """Tune (walk-forward, train-only), fit on log1p volume, report train/val/test.

    mae_usd back-transform is clipped to the train target max (no expm1 blow-ups).
    """
    spec = REGRESSOR_SPECS[name]
    xkey = "scaled" if spec["scaled"] else "raw"
    X = data["X"][xkey]
    y = data["y"][target]
    clip = float(np.max(y["train"]))
    tuned = tune_regressor(name, data, target=target) if (tune and params is None) \
        else {"best_params": params or {}, "best_cv": float("nan"), "table": None}
    model = build_model(name, "regressor", **tuned["best_params"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X["train"], y["train"])
    preds = {s: np.asarray(model.predict(X[s]), dtype=float) for s in ("train", "val", "test")}
    out = {"name": name, "kind": "regressor", "status": "trained",
           "scaled_input": spec["scaled"], "target": target,
           "params": tuned["best_params"],
           "cv": {"rmse_mean": tuned["best_cv"]},
           "metrics": {s: evaluate_regression(y[s], preds[s], clip_max_log=clip)
                       for s in ("train", "val", "test")},
           "importances": _importances(model, data["feature_cols"]),
           "predictions": _prediction_frame(data, preds, y)}
    m = out["metrics"]
    print(f" {name}: train/val/test R² {m['train']['r2']:.3f}/{m['val']['r2']:.3f}/"
          f"{m['test']['r2']:.3f} | test MAE(log) {m['test']['mae']:.3f} "
          f"| MAE ${m['test']['mae_usd']:,.0f}")
    return out


def run_baseline_classifiers(data, cv_target="y_stress_1d"):
    """Two no-model references per stress target: persistence and prevalence.

    persistence — score = today's same-day stress flag (liquidation autocorrelation
    is the thing to beat); prevalence — constant train positive rate (AUC 0.5 anchor).
    Persistence's cv_target scores join the PRI component pool on equal terms.
    """
    out = []
    for base in ("persistence", "prevalence"):
        res = {"name": f"baseline_{base}", "kind": "classifier", "status": "trained",
               "params": {}, "thresholds_f1": {}, "proper_probs": False, "targets": {}}
        for t in STRESS_TARGETS:
            if base == "persistence":
                scores = {s: data["stress"][s].astype(float) for s in ("train", "val", "test")}
            else:
                rate = float(np.mean(data["y"][t]["train"]))
                scores = {s: np.full(len(data["y"][t][s]), rate) for s in ("train", "val", "test")}
            thr = best_f1_threshold(data["y"][t]["val"], scores["val"])
            res["thresholds_f1"][t] = thr
            res["targets"][t] = {s: evaluate_classification(
                data["y"][t][s], scores[s], threshold=thr, proper_probs=False)
                for s in ("train", "val", "test")}
            if t == cv_target:
                res["predictions"] = _prediction_frame(data, scores, data["y"][t])
                res["test_auc_ci"] = bootstrap_auc_ci(data["y"][t]["test"], scores["test"])
        m = res["targets"][cv_target]["test"]
        print(f" baseline_{base}: test ROC-AUC {m['roc_auc']:.3f} | PR-AUC {m['pr_auc']:.3f}")
        out.append(res)
    return out


def run_baseline_regressors(data, target=REG_TARGET):
    """Two no-model references for the volume target: persistence and train median."""
    y = data["y"][target]
    clip = float(np.max(y["train"]))
    today_log = {s: np.log1p(np.asarray(
        data["splits"][s][TARGET_BASE_COL], dtype=float)) for s in ("train", "val", "test")}
    med = float(np.median(y["train"]))
    out = []
    for base, preds in (("persistence", today_log),
                        ("median", {s: np.full(len(y[s]), med) for s in y})):
        res = {"name": f"baseline_{base}", "kind": "regressor", "status": "trained",
               "target": target, "params": {},
               "metrics": {s: evaluate_regression(y[s], preds[s], clip_max_log=clip)
                           for s in ("train", "val", "test")},
               "predictions": _prediction_frame(data, preds, y)}
        m = res["metrics"]["test"]
        print(f" baseline_{base}: test RMSE(log) {m['rmse']:.3f} | R² {m['r2']:.3f} "
              f"| MAE ${m['mae_usd']:,.0f}")
        out.append(res)
    return out


def run_isolation_forest(data, contamination="auto"):
    """Unsupervised anomaly model: fit on TRAIN features only, no labels.

    Anomaly score = 1 - ECDF_train(score_samples) in [0, 1] (higher = more anomalous),
    evaluated as a pseudo-classifier against every stress target (rank metrics only —
    Brier is approximate for an uncalibrated score).
    """
    X = data["X"]["raw"]
    model = IsolationForest(n_estimators=400, contamination=contamination,
                            random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X["train"])
    raw = {s: -model.score_samples(X[s]) for s in ("train", "val", "test")}
    scores = {s: rank_normalize(raw[s], raw["train"]) for s in ("train", "val", "test")}
    out = {"name": "isolation_forest", "kind": "anomaly", "status": "trained",
           "proper_probs": False,
           "targets": {t: {s: evaluate_classification(data["y"][t][s], scores[s],
                                                      proper_probs=False)
                           for s in ("train", "val", "test")}
                       for t in STRESS_TARGETS},
           "test_auc_ci": bootstrap_auc_ci(data["y"]["y_stress_1d"]["test"], scores["test"]),
           "predictions": _prediction_frame(data, scores, data["y"]["y_stress_1d"])}
    m = out["targets"]["y_stress_1d"]["test"]
    print(f" isolation_forest: test ROC-AUC {m['roc_auc']:.3f} | PR-AUC {m['pr_auc']:.3f} "
          f"(anomaly score vs next-day stress — NOTE: features include same-day "
          f"liquidation columns, so this partly measures target autocorrelation; "
          f"judge it against baseline_persistence)")
    return out


def build_survival_frame(times, stress_flags):
    """Per-day survival target: days until the NEXT same-day stress day.

    duration = calendar days from t (exclusive) to the next stress day; event = 1.
    Days after the last stress day are right-censored at the panel end (event = 0).
    """
    days = pd.to_datetime(pd.Series(times).astype(str).str.replace(" UTC", "", regex=False))
    flags = np.asarray(stress_flags).astype(int)
    stress_idx = np.flatnonzero(flags)
    dur, ev = [], []
    for i in range(len(days)):
        nxt = stress_idx[stress_idx > i]
        if len(nxt):
            dur.append((days.iloc[nxt[0]] - days.iloc[i]).days)
            ev.append(1)
        else:
            dur.append((days.iloc[len(days) - 1] - days.iloc[i]).days)
            ev.append(0)
    return pd.DataFrame({"duration": dur, "event": ev})


def run_cox(data, penalizer=0.10, top_n_features=15):
    """Cox proportional-hazards on time-to-next-stress-day (lifelines, optional).

    Features = top-N by a seeded RandomForest importance (collinearity control on
    ~250 rows), scaled. C-index on val/test; the rank-normalized partial hazard is
    the survival PRI component. Returns a skipped-result dict when lifelines is
    missing or the fit fails to converge.
    """
    if not LIFELINES_AVAILABLE:
        print(" cox: SKIPPED — lifelines not installed")
        return {"name": "cox_ph", "kind": "survival", "status": "skipped",
                "reason": "lifelines not installed"}
    try:
        from lifelines import CoxPHFitter
        from lifelines.utils import concordance_index

        rf = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                    random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(data["X"]["raw"]["train"], data["y"]["y_stress_1d"]["train"])
        order = np.argsort(rf.feature_importances_)[::-1][:top_n_features]
        cols = [data["feature_cols"][i] for i in order]

        frames, survs = {}, {}
        for s in ("train", "val", "test"):
            xs = pd.DataFrame(data["X"]["scaled"][s], columns=data["feature_cols"])[cols]
            sv = build_survival_frame(data["times"][s], data["stress"][s])
            keep = sv["duration"] > 0
            frames[s] = pd.concat([xs, sv], axis=1).loc[keep].reset_index(drop=True)
            survs[s] = keep
        cph = CoxPHFitter(penalizer=penalizer)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph.fit(frames["train"], duration_col="duration", event_col="event")

        metrics, scores = {}, {}
        for s in ("train", "val", "test"):
            xs_all = pd.DataFrame(data["X"]["scaled"][s], columns=data["feature_cols"])[cols]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                haz = np.asarray(cph.predict_partial_hazard(xs_all), dtype=float).ravel()
            scores[s] = haz
            f = frames[s]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                haz_kept = np.asarray(cph.predict_partial_hazard(f[cols]), dtype=float).ravel()
            metrics[s] = {"c_index": float(concordance_index(
                f["duration"], -haz_kept, f["event"]))}
        norm = {s: rank_normalize(scores[s], scores["train"]) for s in scores}
        out = {"name": "cox_ph", "kind": "survival", "status": "trained",
               "features_used": cols, "metrics": metrics,
               "predictions": _prediction_frame(data, norm, data["y"]["y_stress_1d"])}
        print(f" cox_ph: C-index train {metrics['train']['c_index']:.3f} | "
              f"val {metrics['val']['c_index']:.3f} | "
              f"test {metrics['test']['c_index']:.3f} ({len(cols)} features)")
        return out
    except Exception as exc:
        print(f" cox: FAILED — {type(exc).__name__}: {exc}")
        return {"name": "cox_ph", "kind": "survival", "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}"}


def make_sequences(X, lookback):
    """Sliding windows over the full timeline: sequence i ends at row i (features
    up to and including day i — the target is already forward-looking)."""
    return np.stack([X[i - lookback + 1:i + 1] for i in range(lookback - 1, len(X))])


def run_lstm(data, target="y_stress_1d", lookback=14, hidden=32, epochs=200,
             lr=1e-3, patience=20):
    """Single-layer LSTM classifier on 14-day feature windows (torch, optional)."""
    return _run_sequence_model(data, arch="lstm", target=target, lookback=lookback,
                               hidden=hidden, epochs=epochs, lr=lr, patience=patience)


def run_transformer(data, target="y_stress_1d", lookback=14, d_model=32, n_heads=4,
                    n_layers=2, epochs=200, lr=1e-3, patience=20):
    """Temporal Transformer classifier on 14-day windows (torch, optional).

    A learned input projection + additive positional embedding feeds a small
    TransformerEncoder; the last position's representation is the classification
    head input — the plan.md temporal-attention counterpart to the LSTM.
    """
    return _run_sequence_model(data, arch="transformer", target=target,
                               lookback=lookback, hidden=d_model, n_heads=n_heads,
                               n_layers=n_layers, epochs=epochs, lr=lr, patience=patience)


def _run_sequence_model(data, arch, target="y_stress_1d", lookback=14, hidden=32,
                        n_heads=4, n_layers=2, epochs=200, lr=1e-3, patience=20):
    """Shared torch training loop for the sequence classifiers (LSTM / Transformer).

    Sequences are built over the stacked timeline (windows only ever look BACK, so
    a window crossing a split boundary uses past features only — no label leakage;
    the embargo gap days simply appear as a jump inside those windows). Each
    sequence belongs to the split of its END day. Early stopping on val BCE.
    Returns a skipped-result dict when torch is missing.
    """
    if not TORCH_AVAILABLE:
        print(f" {arch}: SKIPPED — torch not installed")
        return {"name": arch, "kind": "classifier", "status": "skipped",
                "reason": "torch not installed"}
    # macOS: xgboost/lightgbm already loaded Homebrew's libomp in this process;
    # torch ships its own OpenMP copy and the duo deadlocks on first tensor op
    # unless the duplicate runtime is allowed and torch stays single-threaded.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import torch
    from torch import nn

    torch.set_num_threads(1)
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    sizes = {s: len(data["times"][s]) for s in ("train", "val", "test")}
    X_full = np.vstack([data["X"]["scaled"][s] for s in ("train", "val", "test")])
    y_full = np.concatenate([data["y"][target][s] for s in ("train", "val", "test")])
    seqs = make_sequences(X_full, lookback)
    ends = np.arange(lookback - 1, len(X_full))
    split_of = np.where(ends < sizes["train"], "train",
                        np.where(ends < sizes["train"] + sizes["val"], "val", "test"))

    xt = torch.tensor(seqs, dtype=torch.float32)
    yt = torch.tensor(y_full[ends], dtype=torch.float32)
    masks = {s: torch.tensor(split_of == s) for s in ("train", "val", "test")}

    class _LSTMNet(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    class _TransformerNet(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.proj = nn.Linear(n_features, hidden)
            self.pos = nn.Parameter(torch.zeros(1, lookback, hidden))
            layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 2,
                dropout=0.1, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            z = self.encoder(self.proj(x) + self.pos)
            return self.head(z[:, -1, :]).squeeze(-1)

    net = _LSTMNet(X_full.shape[1]) if arch == "lstm" else _TransformerNet(X_full.shape[1])
    y_tr = yt[masks["train"]]
    pos = float(y_tr.sum())
    pos_weight = torch.tensor((len(y_tr) - pos) / pos if pos else 1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    best_val, best_state, stale = float("inf"), None, 0
    for _ in range(epochs):
        net.train()
        opt.zero_grad()
        loss = loss_fn(net(xt[masks["train"]]), y_tr)
        loss.backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(net(xt[masks["val"]]), yt[masks["val"]]))
        if val_loss < best_val - 1e-5:
            best_val, stale = val_loss, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)

    net.eval()
    with torch.no_grad():
        probs = torch.sigmoid(net(xt)).numpy()

    # ⚠️ the first lookback-1 train days have no window; PRI treats them as unscored
    scores, ys = {}, {}
    for s in ("train", "val", "test"):
        m = (split_of == s)
        scores[s], ys[s] = probs[m], y_full[ends][m]
    thr = best_f1_threshold(ys["val"], scores["val"])
    out = {"name": arch, "kind": "classifier", "status": "trained",
           "lookback": lookback, "proper_probs": True,
           "thresholds_f1": {target: thr},
           "targets": {target: {s: evaluate_classification(ys[s], scores[s], threshold=thr)
                                for s in ("train", "val", "test")}},
           "test_auc_ci": bootstrap_auc_ci(ys["test"], scores["test"]),
           "predictions": pd.concat([
               pd.DataFrame({TIME_COL: np.asarray(data["times"][s])[-len(scores[s]):]
                             if s == "train" else data["times"][s],
                             "split": s, "y_true": ys[s].astype(float),
                             "score": scores[s].astype(float)})
               for s in ("train", "val", "test")], ignore_index=True)}
    m = out["targets"][target]
    print(f" {arch}: train/val/test ROC-AUC {m['train']['roc_auc']:.3f}/"
          f"{m['val']['roc_auc']:.3f}/{m['test']['roc_auc']:.3f} (lookback {lookback}d)")
    return out


def _component_skill(res, target="y_stress_1d"):
    """A component's skill for meta-weighting: mean of val AUC and train CV AUC.

    Averaging the two independent estimates halves the variance of a val AUC read
    off ~8 positives; survival components use the val C-index (no CV available).
    """
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
    return (res.get("kind") != "regressor" and "predictions" in res
            and not str(res.get("name", "")).startswith("baseline_")
            and res.get("name") != "meta_pri")


def fit_meta_weights(results, target="y_stress_1d", floor=0.02, shrink=0.30):
    """Skill-proportional weights, floored and shrunk toward equal weights.

    w_i ∝ (1-shrink)·max(skill_i − 0.5, floor) + shrink·(1/n). The floor keeps
    weak-but-trained components alive and the shrinkage stops the 53-row val split
    from zeroing 11/14 models on AUC noise (a fitted stacker would overfit worse).
    """
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
    """PRI = 100 × weighted mean of the rank-normalized component columns.

    NaN components (e.g. LSTM warm-up days) are excluded per-row via a
    weight-renormalized mean.
    """
    cols = [c for c in weights if c in score_frame.columns]
    w = np.array([weights[c] for c in cols])
    vals = score_frame[cols].to_numpy(dtype=float)
    mask = ~np.isnan(vals)
    weighted = np.nansum(vals * w, axis=1)
    denom = (mask * w).sum(axis=1)
    return 100.0 * weighted / np.where(denom > 0, denom, np.nan)


def run_meta_pri(results, data, target="y_stress_1d"):
    """The meta-learner: blend every trained component score into the PRI series.

    Each component's full-timeline scores are rank-normalized against its TRAIN
    values, weighted by validation skill, and averaged to a 0-100 index. Evaluated
    like any classifier against the stress targets on test.
    """
    frame = None
    for res in results.values():
        if not _is_meta_component(res):
            continue
        if _component_skill(res, target) is None:
            continue
        preds = res["predictions"]
        ref = preds.loc[preds["split"] == "train", "score"]
        col = pd.DataFrame({
            TIME_COL: preds[TIME_COL], "split": preds["split"],
            res["name"]: rank_normalize(preds["score"], ref)})
        frame = col if frame is None else frame.merge(
            col, on=[TIME_COL, "split"], how="outer")
    frame = frame.sort_values(TIME_COL, kind="stable").reset_index(drop=True)

    weights = fit_meta_weights(results, target)
    frame["pri"] = compute_pri(frame, weights)

    y_true = pd.concat([pd.DataFrame({TIME_COL: data["times"][s], "split": s,
                                      "y_true": data["y"][target][s]})
                        for s in ("train", "val", "test")], ignore_index=True)
    frame = frame.merge(y_true, on=[TIME_COL, "split"], how="left")

    out = {"name": "meta_pri", "kind": "meta", "status": "trained",
           "proper_probs": False,
           "weights": weights, "n_components": len(weights), "targets": {}}
    for t in STRESS_TARGETS:
        yt = pd.concat([pd.DataFrame({TIME_COL: data["times"][s], "split": s,
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
