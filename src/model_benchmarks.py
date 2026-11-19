"""Modeling stage — read-side benchmarks over results (saved in model_results/ or in memory). No fitting here.

The backend imports this file, so the top-level imports stay numpy / pandas; sklearn and matplotlib
are imported inside the functions that need them.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import math

import numpy as np
import pandas as pd

from model_config import HORIZONS, STRESS_TARGETS, TARGET_BASE_COL, TIME_COL, CI_BLOCK_ROWS, RANDOM_STATE

# The realized outcome read next to the PRI: real USD of liquidation debt over the next 24h.
OUTCOME_COL = f"{TARGET_BASE_COL}_fwd_{HORIZONS[0]}"


def model_scorecard(result):
    """Per-model benchmark table (val + test) for one result dict.
    result -> rank metrics per horizon (classifier/anomaly/meta), error metrics, or C-index.
    Returns: a tidy DataFrame; skipped/failed models give a one-row status frame."""
    kind = result.get("kind", "model")
    if result.get("status") != "trained":
        return pd.DataFrame([{"model": result.get("name"), "kind": kind,
                              "status": result.get("status"),
                              "reason": result.get("reason", "")}])
    rows = []
    ci = result.get("test_auc_ci") or {}
    if kind in ("classifier", "anomaly", "meta"):
        cv = result.get("cv") or {}
        for t, per_split in result["targets"].items():
            for s in ("train", "val", "test"):
                if s not in per_split:
                    continue
                m = per_split[s]
                rows.append({"target": t, "split": s,
                             "roc_auc": round(m["roc_auc"], 3),
                             "pr_auc": round(m["pr_auc"], 3),
                             "f1": round(m["f1"], 3),
                             "brier": round(m["brier"], 3) if not math.isnan(m["brier"]) else None,
                             "n_pos": m["n_pos"], "n": m["n"],
                             "cv_auc": round(cv["mean"], 3) if (t == STRESS_TARGETS[0]
                                       and cv.get("mean") is not None
                                       and not math.isnan(cv["mean"])) else None,
                             "auc_ci95": (f"[{ci['lo']:.3f}, {ci['hi']:.3f}]"
                                          if s == "test" and t == STRESS_TARGETS[0] and ci
                                          else None)})
        return pd.DataFrame(rows)
    if kind == "regressor":
        for s in ("train", "val", "test"):
            if s not in result["metrics"]:
                continue
            m = result["metrics"][s]
            rows.append({"split": s, "rmse": round(m["rmse"], 3),
                         "mae": round(m["mae"], 3), "r2": round(m["r2"], 3),
                         "mae_usd": round(m["mae_usd"], 0)})
        return pd.DataFrame(rows)
    if kind == "survival":
        for s in ("train", "val", "test"):
            if s not in result["metrics"]:
                continue
            rows.append({"split": s, "c_index": round(result["metrics"][s]["c_index"], 3)})
        return pd.DataFrame(rows)
    return pd.DataFrame([{"model": result.get("name"), "kind": kind}])


def plot_history(result, target=None):
    """Training curve of a sequence model (LSTM / Transformer): train and val loss per epoch, the
    learning rate on the right axis, and the best epoch (the weights kept) and stop epoch marked.
    Shows overfitting (val loss turns up while train keeps falling) and whether a learning-rate drop
    came before or after training stopped.
    target: which horizon — defaults to the first one the result trained.
    Returns: the matplotlib Figure."""
    import matplotlib.pyplot as plt
    history = result.get("history") or {}
    if not history:
        raise ValueError(f"{result.get('name')}: no training history (only LSTM / Transformer have one)")
    target = target or next(iter(history))
    h = pd.DataFrame(history[target])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(h["epoch"], h["train_loss"], label="train loss")
    ax.plot(h["epoch"], h["val_loss"], label="val loss (early stopping)")
    ax.axvline(result["best_epoch"][target], color="green", linestyle="--", label="best epoch (kept)")
    ax.axvline(result["stop_epoch"][target], color="grey", linestyle=":", label="stop epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    lr_ax = ax.twinx()
    lr_ax.plot(h["epoch"], h["lr"], color="orange", alpha=0.6, label="learning rate")
    lr_ax.set_ylabel("learning rate")
    lines = ax.get_legend_handles_labels()[0] + lr_ax.get_legend_handles_labels()[0]
    ax.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=8)
    ax.set_title(f"{result.get('name')} — {target.rpartition('_')[2]}")
    fig.tight_layout()
    return fig


# =========================================================================== #
# Forests: AUC by tree count, importance plot
# =========================================================================== #

def forest_curve(result, data, target=STRESS_TARGETS[0], splits=("train", "val"), step=10):
    """A fitted forest's AUC as trees are added: the running mean of its first k trees' scores
    (a forest's probability is that mean), per split, plus out-of-bag (OOB) on train.
    Monitoring only: the tree count is picked on CV (model_evaluation.cv_forest_curve).
    Reading it: train far above the rest = memorising; a flat tail = more trees only cost time;
    OOB sits above val because 2h rows have their neighbours in the bag. Under class_weight="balanced"
    each tree draws positives about 3x as often, so a positive row is out of the bag in about 1 tree
    of 7 (a negative in 1 of 2); oob_coverage = rows with at least one out-of-bag tree.
    result: a runner result with its fitted models (one read back from disk has none).
    Returns: one row per k — n_trees, an AUC column per split, oob, oob_coverage."""
    from sklearn.metrics import roc_auc_score
    fitted = result.get("fitted") or {}
    if not fitted:
        raise ValueError(f"{result.get('name')}: no fitted models (read back from disk?): re-run its cell")
    if target not in fitted:
        raise ValueError(f"no {target} in this result: {sorted(fitted)}")
    model = fitted[target]
    if not hasattr(model, "estimators_samples_") or list(getattr(model, "classes_", [])) != [0, 1]:
        raise ValueError(f"{result.get('name')}: not a bagged 0/1 forest, no per-tree curve")
    cols = ((result.get("features_by_target") or {}).get(target) or result.get("features_used")
            or data["feature_cols"])
    xkey = "scaled" if result.get("scaled_input") else "raw"
    trees = model.estimators_
    n_trees = len(trees)
    ks = sorted({1, n_trees, *range(step, n_trees + 1, step)})
    curve = {"n_trees": ks}
    for s in splits:
        X = pd.DataFrame(data["X"][target][xkey][s], columns=data["feature_cols"])[cols].to_numpy()
        X32 = np.ascontiguousarray(X, dtype=np.float32)
        y = np.asarray(data["y"][target][s]).astype(int)
        P = np.vstack([t.predict_proba(X32, check_input=False)[:, 1] for t in trees])
        running = np.cumsum(P, axis=0) / np.arange(1, n_trees + 1)[:, None]
        if not np.allclose(running[-1], model.predict_proba(X)[:, 1], atol=1e-9):
            raise RuntimeError(f"{result.get('name')}: the mean of the trees does not match predict_proba")
        two = len(np.unique(y)) > 1
        curve[s] = [float(roc_auc_score(y, running[k - 1])) if two else float("nan") for k in ks]
        if s == "train":
            # OOB: a row is scored only by the trees that did not draw it (estimators_samples_
            # repeats each tree's own draw, class weighting and max_samples included)
            n = len(y)
            o_sum, o_cnt = np.zeros(n), np.zeros(n)
            oob, cover = [], []
            for j, idx in enumerate(model.estimators_samples_, 1):
                out_of_bag = np.bincount(idx, minlength=n) == 0
                o_sum += P[j - 1] * out_of_bag
                o_cnt += out_of_bag
                if j in ks:
                    seen = o_cnt > 0
                    ok = seen.any() and len(np.unique(y[seen])) > 1
                    oob.append(float(roc_auc_score(y[seen], o_sum[seen] / o_cnt[seen])) if ok else float("nan"))
                    cover.append(round(float(seen.mean()), 3))
            curve["oob"], curve["oob_coverage"] = oob, cover
    return pd.DataFrame(curve)


def plot_forest_curve(curve, title=None):
    """AUC by tree count (forest_curve or model_evaluation.cv_forest_curve).
    Returns: the matplotlib Figure."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    for col, style in (("train", "-"), ("oob", ":"), ("cv_auc", "-"), ("val", "-"), ("test", "--")):
        if col in curve.columns and curve[col].notna().any():
            ax.plot(curve["n_trees"], curve[col], style, label="CV (train folds)" if col == "cv_auc" else col)
    ax.set_xlabel("trees")
    ax.set_ylabel("ROC-AUC")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(title or "AUC by tree count")
    fig.tight_layout()
    return fig


def plot_importance(table, top=20, title=None):
    """Permutation importance (model_evaluation.cv_permutation_importance): mean AUC drop with its
    spread across folds. A bar below 0 = the model did better on unseen periods with it scrambled.
    Returns: the matplotlib Figure."""
    import matplotlib.pyplot as plt
    t = table.head(top).iloc[::-1]
    label = "group" if "group" in t.columns else t.columns[0]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(t))))
    ax.barh(t[label].astype(str), t["drop_mean"], xerr=t.get("drop_std"), color="steelblue", alpha=0.8)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("AUC drop when shuffled (CV test parts)")
    ax.set_title(title or "Permutation importance")
    fig.tight_layout()
    return fig


# =========================================================================== #
# Boosted models, linear coefficients, series diagnostics
# =========================================================================== #

def _fitted(result, target):
    fitted = result.get("fitted") or {}
    if target not in fitted:
        raise ValueError(f"{result.get('name')}: no fitted model for {target} (read back from disk? re-run its cell)")
    return fitted[target]


def _result_cols(result, data, target):
    return ((result.get("features_by_target") or {}).get(target) or result.get("features_used")
            or data["feature_cols"])


def boost_curve(result, data, target=STRESS_TARGETS[0], splits=("train", "val"), step=10):
    """A fitted boosted model's AUC as rounds are added (xgboost, catboost, lightgbm), per split.
    Monitoring only: the round count is picked on CV (model_evaluation.cv_boost_curve).
    Reading it: train still climbing while val is flat or falling = the extra rounds fit the train
    period only. Plot with plot_forest_curve.
    Returns: one row per k — n_trees and an AUC column per split."""
    from sklearn.metrics import roc_auc_score
    from model_evaluation import boost_rounds, staged_scores
    model, name = _fitted(result, target), result["name"]
    cols = _result_cols(result, data, target)
    xkey = "scaled" if result.get("scaled_input") else "raw"
    n = boost_rounds(name, model)
    ks = sorted({1, n, *range(step, n + 1, step)})
    curve = {"n_trees": ks}
    for s in splits:
        X = pd.DataFrame(data["X"][target][xkey][s], columns=data["feature_cols"])[cols].to_numpy()
        y = np.asarray(data["y"][target][s]).astype(int)
        two = len(np.unique(y)) > 1
        curve[s] = [float(roc_auc_score(y, sc)) if two else float("nan") for sc in staged_scores(name, model, X, ks)]
    return pd.DataFrame(curve)


def coef_table(result, target=STRESS_TARGETS[0]):
    """The fitted coefficients of a linear model (logistic, elastic net) or of Cox, one row per column.
    Inputs are standardised, so a coefficient is the change in log-odds (Cox: log-hazard) per one
    standard deviation of the column; ratio = exp(coef). Zero = the penalty dropped the column.
    Returns: column, coef, ratio — strongest first."""
    model = _fitted(result, target)
    cols = (result.get("features_by_target") or {}).get(target) or result.get("features_used")
    if hasattr(model, "coef_"):
        coef = np.ravel(model.coef_)
    elif hasattr(model, "params_"):                       # lifelines CoxPHFitter
        coef, cols = model.params_.to_numpy(), list(model.params_.index)
    else:
        raise ValueError(f"{result.get('name')}: no coefficients on this model")
    table = pd.DataFrame({"column": cols, "coef": coef, "ratio": np.exp(coef)})
    return table.loc[table["coef"].abs().sort_values(ascending=False, kind="stable").index].reset_index(drop=True)


def series_diagnostics(result):
    """What a series model (sarimax, hmm_regime) reports about its own fit, one row per horizon:
    convergence, iterations, log-likelihood (+ AIC / BIC and the coefficients for SARIMAX).
    Returns: DataFrame."""
    rows = []
    for t, d in (result.get("diagnostics") or {}).items():
        row = {"horizon": t.rpartition("_")[2], **{k: v for k, v in d.items() if not isinstance(v, (list, dict))}}
        row.update({f"param_{k}": v for k, v in (d.get("params") or {}).items()})
        rows.append(row)
    if not rows:
        raise ValueError(f"{result.get('name')}: no diagnostics (only sarimax and hmm_regime carry them)")
    return pd.DataFrame(rows)


def hmm_states(result, target=STRESS_TARGETS[0]):
    """The regimes an HMM found, lowest stress rate first: the stress rate of each state on train, the
    share of train rows it holds, the chance of staying in it from one row to the next and the dwell
    time that implies (rows of 2h). The states are unsupervised; the labels only name them.
    Returns: DataFrame, one row per state."""
    d = (result.get("diagnostics") or {}).get(target)
    if not d or "state_stress_rate" not in d:
        raise ValueError(f"{result.get('name')}: no HMM states for {target}")
    stay = np.array(d["stay_probability"], dtype=float)
    return pd.DataFrame({"state": range(len(stay)), "stress_rate": d["state_stress_rate"],
                         "share_of_train": d["state_share"], "stay_probability": stay,
                         "dwell_rows": 1.0 / np.maximum(1.0 - stay, 1e-9)})


def plot_loglik(result, target=STRESS_TARGETS[0]):
    """EM's log-likelihood per iteration for an HMM: it must rise and flatten. A curve still climbing
    at the last iteration = raise n_iter. Returns: the matplotlib Figure."""
    import matplotlib.pyplot as plt
    d = (result.get("diagnostics") or {}).get(target) or {}
    hist = d.get("loglik_history")
    if not hist:
        raise ValueError(f"{result.get('name')}: no log-likelihood history for {target}")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(range(1, len(hist) + 1), hist)
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("log-likelihood (train)")
    ax.set_title(f"{result.get('name')} — {target.rpartition('_')[2]} (converged: {d.get('converged')})")
    fig.tight_layout()
    return fig


# =========================================================================== #
# A tuning chain read side by side, and a paired test difference
# =========================================================================== #

def _pair(folds_a, folds_b):
    """Mean paired difference a - b over folds both scored, its SE and the folds a won."""
    a, b = np.asarray(folds_a, dtype=float), np.asarray(folds_b, dtype=float)
    if a.shape != b.shape:
        return float("nan"), float("nan"), None
    d = (a - b)[~(np.isnan(a) | np.isnan(b))]
    if not len(d):
        return float("nan"), float("nan"), None
    se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")
    return float(d.mean()), se, int((d > 0).sum())


def stage_table(stages, ref=None, test_for=()):
    """One row per stage of a tuning chain, from the results in memory: configs scored, 24h CV and
    its folds, the paired gain on the same folds against `ref` and against the stage before, the
    overfit gap (train - CV: train - val would also hold the regime shift), and val per horizon.
    Test appears only for the labels in test_for (finished models), so no stage is judged on it.
    stages: {label: result} in chain order; baselines (no CV) get blank CV columns.
    Returns: a DataFrame indexed by stage."""
    rows, prev_folds = [], None
    ref_folds = ((stages[ref].get("cv") or {}).get("folds") if ref is not None else None)
    for label, res in stages.items():
        cv = res.get("cv") or {}
        folds = cv.get("folds")
        row = {"stage": label, "n_trials": res.get("n_trials"),
               "n_features": len(res["features_used"]) if res.get("features_used") else None,
               "cv_auc": cv.get("mean"), "cv_se": cv.get("se")}
        for i, f in enumerate(folds or [], 1):
            row[f"fold_{i}"] = f
        if folds is not None and ref_folds is not None:
            row["d_vs_ref"], row["d_se"], row["wins_vs_ref"] = _pair(folds, ref_folds)
        if folds is not None and prev_folds is not None:
            row["d_vs_prev"], _, row["wins_vs_prev"] = _pair(folds, prev_folds)
        if folds is not None:
            prev_folds = folds
        tuning = res.get("tuning")
        for t, per_split in res.get("targets", {}).items():
            h = t.rpartition("_")[2]
            if h == HORIZONS[0]:
                row[f"train_{h}"] = per_split["train"]["roc_auc"]
                if cv.get("mean") is not None:
                    row["gap_train_cv"] = per_split["train"]["roc_auc"] - cv["mean"]
            elif isinstance(tuning, pd.DataFrame) and "horizon" in tuning.columns:
                picked = tuning[(tuning["horizon"] == h) & tuning["chosen"].astype(bool)]
                if len(picked):
                    row[f"cv_auc_{h}"] = float(picked["cv_auc"].iloc[0])
            row[f"val_{h}"] = per_split["val"]["roc_auc"]
            if label in test_for:
                row[f"test_{h}"] = per_split["test"]["roc_auc"]
        ci = res.get("test_auc_ci")
        if label in test_for and ci:
            row["test_ci_24h"] = f"[{ci['lo']:.3f}, {ci['hi']:.3f}]"
        row["changed"] = ", ".join(res.get("changed") or [])
        row["settings_hash"] = res.get("settings_hash")
        rows.append(row)
    return pd.DataFrame(rows).set_index("stage")


def paired_auc_ci(res_a, res_b, split="test", n_boot=1000, block=CI_BLOCK_ROWS, level=0.95,
                  seed=RANDOM_STATE):
    """Is A really better than B on one split? Both results' 24h scores (their predictions frames)
    are lined up on time and the SAME moving blocks of rows are resampled for both, so the shared
    easy and hard weeks cancel. Two separate CIs of about ±0.1 cannot answer this; the paired one can.
    Returns: {auc_a, auc_b, diff, lo, hi, share_a_better, n, n_boot_valid}."""
    from sklearn.metrics import roc_auc_score
    frames = []
    for res in (res_a, res_b):
        p = res.get("predictions")
        if not isinstance(p, pd.DataFrame):
            raise ValueError(f"{res.get('name')}: no predictions frame")
        frames.append(p[p["split"] == split][[TIME_COL, "y_true", "score"]])
    m = frames[0].merge(frames[1], on=TIME_COL, suffixes=("_a", "_b")).sort_values(TIME_COL)
    if not np.array_equal(m["y_true_a"].to_numpy(), m["y_true_b"].to_numpy()):
        raise ValueError("the two results do not share the same labels on these rows")
    y = m["y_true_a"].to_numpy().astype(int)
    a, b = m["score_a"].to_numpy(dtype=float), m["score_b"].to_numpy(dtype=float)
    n = len(y)
    rng = np.random.default_rng(seed)
    n_blocks = max(1, int(np.ceil(n / block)))
    diffs = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(1, n - block + 1), size=n_blocks)
        idx = np.concatenate([np.arange(st, min(st + block, n)) for st in starts])[:n]
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    alpha = (1 - level) / 2
    auc_a, auc_b = float(roc_auc_score(y, a)), float(roc_auc_score(y, b))
    return {"a": res_a.get("name"), "b": res_b.get("name"), "auc_a": auc_a, "auc_b": auc_b,
            "diff": auc_a - auc_b,
            "lo": float(np.quantile(diffs, alpha)) if diffs else float("nan"),
            "hi": float(np.quantile(diffs, 1 - alpha)) if diffs else float("nan"),
            "share_a_better": float(np.mean(np.asarray(diffs) > 0)) if diffs else float("nan"),
            "n": n, "n_boot_valid": len(diffs)}


def weight_concentration(weights):
    """How concentrated the PRI blend is.
    weights -> inverse Herfindahl (1 / sum wi^2) plus the largest single weight.
    Returns: dict. Low effective-N means the index rides on one or two models."""
    w = np.array([v for v in weights.values() if v > 0], dtype=float)
    if w.size == 0:
        return {"n_nonzero": 0, "effective_n": 0.0, "max_weight": 0.0}
    hhi = float(np.sum(w ** 2))
    return {"n_nonzero": int(w.size),
            "effective_n": round(1.0 / hhi, 2),
            "max_weight": round(float(w.max()), 3)}


def build_leaderboard(results, kind="classification", split="test"):
    """One row per model (x horizon for classification), sorted by the lead metric.
    results, kind: "classification" | "regression" | "survival".
    Returns: DataFrame; skipped models appear with their status so the roster stays visible."""
    rows = []
    if kind == "classification":
        for res in results.values():
            if res.get("kind") not in ("classifier", "anomaly", "meta"):
                continue
            name = res["name"]
            if res.get("status") != "trained":
                rows.append({"model": name, "status": res.get("status")})
                continue
            cv = res.get("cv") or {}
            for t, per_split in res["targets"].items():
                m = per_split[split]
                rows.append({"model": name, "target": t, "status": "trained",
                             "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"],
                             "f1": m["f1"], "brier": m["brier"],
                             "cv_auc_mean": cv.get("mean") if t == STRESS_TARGETS[0] else None,
                             "cv_auc_std": cv.get("std") if t == STRESS_TARGETS[0] else None})
        board = pd.DataFrame(rows)
        return board.sort_values(["target", "roc_auc"],
                                 ascending=[True, False]).reset_index(drop=True)
    if kind == "regression":
        for res in results.values():
            if res.get("kind") != "regressor":
                continue
            m = res["metrics"][split]
            rows.append({"model": res["name"], "rmse": m["rmse"], "mae": m["mae"],
                         "r2": m["r2"], "mae_usd": m["mae_usd"]})
        return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    if kind == "survival":
        for res in results.values():
            if res.get("kind") != "survival":
                continue
            name = res["name"]
            if res.get("status") != "trained":
                rows.append({"model": name, "status": res.get("status")})
                continue
            rows.append({"model": name, "status": "trained",
                         "c_index_val": res["metrics"]["val"]["c_index"],
                         "c_index_test": res["metrics"]["test"]["c_index"]})
        return pd.DataFrame(rows)
    raise ValueError(f"unknown leaderboard kind: {kind}")


def aggregate_feature_importances(results, top_n=15,
                                  models=("random_forest", "extra_trees",
                                          "xgboost", "lightgbm")):
    """Mean of sum-normalized native importances across the tree ensembles.
    results, optional model filter.
    Returns: DataFrame. hist_gradient_boosting and linear |coef| excluded by default."""
    per_model = {}
    for name in models:
        imp = results.get(f"{name}__classifier", results.get(name, {})).get("importances")
        if not imp:
            continue
        total = sum(imp.values()) or 1.0
        per_model[name] = {c: v / total for c, v in imp.items()}
    table = pd.DataFrame(per_model)
    table["mean_importance"] = table.mean(axis=1)
    return (table.sort_values("mean_importance", ascending=False)
            .head(top_n).reset_index(names="feature"))


def select_champion(results, target=STRESS_TARGETS[0], split="val", exclude=("meta_pri",)):
    """Best single model chosen on VAL (never test — selection bias).
    results, horizon -> searches trained classifier/anomaly components, baselines included.
    Returns: (name, val_metrics, test_metrics)."""
    best_name, best_val = None, -np.inf
    for res in results.values():
        if res.get("kind") not in ("classifier", "anomaly") or res.get("status") != "trained":
            continue
        if res["name"] in exclude:
            continue
        tgt = res.get("targets", {}).get(target)
        if not tgt or "val" not in tgt or math.isnan(tgt["val"]["roc_auc"]):
            continue
        if tgt["val"]["roc_auc"] > best_val:
            best_name, best_val = res["name"], tgt["val"]["roc_auc"]
    for res in results.values():
        if res.get("name") == best_name:
            tgt = res["targets"][target]
            return best_name, tgt["val"], tgt["test"]
    return None, None, None


def riskiest_days(pri_frame, outcome, k=10, outcome_col=OUTCOME_COL, time_col=TIME_COL):
    """Top-k calendar days by PRI, each shown at that day's highest-scoring row, with the realized
    outcome of the 24h after that row. One row per day, or one busy day would fill the table.
    pri_frame: time_col, split, score, y_true; outcome: DF_target_raw (time_col + outcome_col,
    already a forward sum over (t, t+24h], in USD).
    Returns: DataFrame [time_col, split, pri, y_true, next_day_<outcome_col>]."""
    f = pri_frame[pri_frame["score"].notna()].copy()
    best = f.groupby(f[time_col].astype(str).str[:10])["score"].idxmax()
    realized = f"next_day_{outcome_col}"
    merged = f.loc[best].merge(outcome[[time_col, outcome_col]].rename(
        columns={outcome_col: realized}), on=time_col, how="left")
    cols = [time_col, "split", "score", "y_true", realized]
    return (merged.sort_values("score", ascending=False)
            .head(k)[cols].rename(columns={"score": "pri"}).reset_index(drop=True))


def lead_time_backtest(pri_frame, stress_flags=None, alert_quantile=0.90, max_lead=7,
                       episode_gap="24h", eval_splits=("test",), reference_split="val",
                       time_col=TIME_COL):
    """How early the PRI flags real stress episodes, by TIME so it holds at any row grain.
    Alert cut = PRI quantile on reference_split (val: the PRI is ranked against val, so train is
    in-sample); events = episode starts in eval_splits (TEST only by default). An episode starts at a stress row with no stress row in the episode_gap before it, so a label
    that flickers inside one event is not counted as many. max_lead: look-back window in DAYS.
    stress_flags: in pri_frame's time order; defaults to its y_true column.
    Returns: (events_table, summary_dict) with lead days and the false-alert rate."""

    # a lead equal to max_lead usually means persistent alerts saturated the window —
    # check n_alert_days against the split length before quoting it
    f = pri_frame.sort_values(time_col, kind="stable").reset_index(drop=True)
    pri = f["score"].to_numpy(dtype=float)
    split = f["split"].to_numpy()
    flags = f["y_true"] if stress_flags is None else stress_flags
    stress = pd.Series(np.asarray(flags, dtype=float)).fillna(0).astype(int).to_numpy()
    stamps = pd.to_datetime(f[time_col].astype(str).str.replace(" UTC", "", regex=False))
    t = stamps.to_numpy()
    lead_span, gap = np.timedelta64(max_lead, "D"), pd.Timedelta(episode_gap).to_timedelta64()
    thr = float(np.nanquantile(pri[split == reference_split], alert_quantile))
    alert = pri >= thr

    def _before(i, span):
        """Row positions in [t_i - span, t_i)."""
        return np.arange(np.searchsorted(t, t[i] - span, side="left"), i)

    starts = [i for i in np.flatnonzero(stress == 1)
              if split[i] in eval_splits and stress[_before(i, gap)].sum() == 0]
    rows = []
    for i in starts:
        window = _before(i, lead_span)
        fired = window[alert[window]]
        hit = bool(len(fired))
        lead = round(float((t[i] - t[fired[0]]) / np.timedelta64(1, "D")), 2) if hit else None
        rows.append({time_col: f[time_col].iloc[i], "split": split[i],
                     "pri_on_day": round(pri[i], 1), "alerted_before": hit,
                     "lead_days": lead, "alert_same_day": bool(alert[i])})
    events = pd.DataFrame(rows)

    eval_mask = np.isin(split, eval_splits)
    alert_idx = np.flatnonzero(alert & eval_mask)
    ends = np.searchsorted(t, t[alert_idx] + lead_span, side="right")
    false_alerts = sum(1 for i, e in zip(alert_idx, ends) if stress[i:e].sum() == 0)
    leads = [r["lead_days"] for r in rows if r["lead_days"] is not None]
    summary = {
        "alert_threshold_pri": round(thr, 1),
        "n_stress_episodes": len(rows),
        "n_alerted_before": int(sum(r["alerted_before"] for r in rows)),
        "hit_rate": round(float(np.mean([r["alerted_before"] for r in rows])), 3)
        if rows else float("nan"),
        "median_lead_days": float(np.median(leads)) if leads else float("nan"),
        "mean_lead_days": round(float(np.mean(leads)), 2) if leads else float("nan"),
        "n_alert_days": int(stamps.iloc[alert_idx].dt.normalize().nunique()),
        "n_alert_rows": int(len(alert_idx)),
        "n_false_alerts": int(false_alerts),
        "false_alert_rate": round(false_alerts / len(alert_idx), 3)
        if len(alert_idx) else float("nan"),
    }
    return events, summary
