"""Modeling stage — read-side benchmarks over the saved model results.

Per-model scorecards, per-horizon leaderboards, aggregated tree-ensemble feature
importances, PRI weight concentration, the VAL-selected champion, riskiest-days
(next-day outcome), and the test-only lead-time backtest. No fitting here — pure
analysis of ``model_results/``. Orchestrated by notebooks/model_results.ipynb.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import math

import numpy as np
import pandas as pd

from model_config import STRESS_TARGETS, TARGET_BASE_COL, TIME_COL


def model_scorecard(result):
    """Per-model benchmark table (val + test) for one result dict.

    One tidy frame per model so a single ``display(model_scorecard(res))`` shows
    that model's benchmark in isolation — classifiers/anomaly/meta report the rank
    metrics per horizon, regressors the error metrics, survival the C-index. Skipped
    or failed models return a one-row status frame instead of raising.
    """
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
                             "cv_auc": round(cv["mean"], 3) if (t == "y_stress_1d"
                                       and cv.get("mean") is not None
                                       and not math.isnan(cv["mean"])) else None,
                             "auc_ci95": (f"[{ci['lo']:.3f}, {ci['hi']:.3f}]"
                                          if s == "test" and t == "y_stress_1d" and ci
                                          else None)})
        return pd.DataFrame(rows)
    if kind == "regressor":
        for s in ("train", "val", "test"):
            if s not in result["metrics"]:
                continue
            m = result["metrics"][s]
            rows.append({"split": s, "rmse_log": round(m["rmse"], 3),
                         "mae_log": round(m["mae"], 3), "r2": round(m["r2"], 3),
                         "mae_usd": round(m["mae_usd"], 0)})
        return pd.DataFrame(rows)
    if kind == "survival":
        for s in ("train", "val", "test"):
            if s not in result["metrics"]:
                continue
            rows.append({"split": s, "c_index": round(result["metrics"][s]["c_index"], 3)})
        return pd.DataFrame(rows)
    return pd.DataFrame([{"model": result.get("name"), "kind": kind}])


def weight_concentration(weights):
    """Diagnostic for how concentrated the PRI blend is.

    Reports the effective number of components (1 / Σwᵢ², the inverse Herfindahl
    index) and the single largest weight — a low effective-N warns the index is
    really driven by one or two models rather than the full zoo.
    """
    w = np.array([v for v in weights.values() if v > 0], dtype=float)
    if w.size == 0:
        return {"n_nonzero": 0, "effective_n": 0.0, "max_weight": 0.0}
    hhi = float(np.sum(w ** 2))
    return {"n_nonzero": int(w.size),
            "effective_n": round(1.0 / hhi, 2),
            "max_weight": round(float(w.max()), 3)}


def build_leaderboard(results, kind="classification", split="test"):
    """One row per model (× horizon for classification), sorted by the lead metric.

    kind="classification" covers classifier/anomaly/meta results; "regression"
    the regressors; "survival" the Cox C-index. Skipped models appear with their
    status so the roster stays visible.
    """
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
                             "cv_auc_mean": cv.get("mean") if t == "y_stress_1d" else None,
                             "cv_auc_std": cv.get("std") if t == "y_stress_1d" else None})
        board = pd.DataFrame(rows)
        return board.sort_values(["target", "roc_auc"],
                                 ascending=[True, False]).reset_index(drop=True)
    if kind == "regression":
        for res in results.values():
            if res.get("kind") != "regressor":
                continue
            m = res["metrics"][split]
            rows.append({"model": res["name"], "rmse_log": m["rmse"], "mae_log": m["mae"],
                         "r2": m["r2"], "mae_usd": m["mae_usd"]})
        return pd.DataFrame(rows).sort_values("rmse_log").reset_index(drop=True)
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

    ⚠️ hist_gradient_boosting has no native importances and linear |coef| lives on
    a different scale — both excluded by default.
    """
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


def select_champion(results, target="y_stress_1d", split="val", exclude=("meta_pri",)):
    """Best single model chosen on the VAL split (never on test — selection bias).

    Returns (name, val_metric_dict, test_metric_dict) among trained
    classifier/anomaly components, baselines included so a model must beat them.
    """
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


def riskiest_days(pri_frame, panel, k=10, target_col=TARGET_BASE_COL, time_col=TIME_COL):
    """Top-k days by PRI with the realized NEXT-day liquidation outcome alongside.

    PRI at day t predicts t+1, so the outcome column is target_col shifted back one
    day (next_day_{target_col}) — showing the same-day value would validate the
    wrong prediction.
    """
    outcome = panel[[time_col, target_col]].sort_values(time_col, kind="stable").copy()
    outcome[f"next_day_{target_col}"] = outcome[target_col].astype(float).shift(-1)
    merged = pri_frame.merge(
        outcome[[time_col, f"next_day_{target_col}"]], on=time_col, how="left")
    cols = [time_col, "split", "score", "y_true", f"next_day_{target_col}"]
    return (merged.sort_values("score", ascending=False)
            .head(k)[cols].rename(columns={"score": "pri"}).reset_index(drop=True))


def lead_time_backtest(pri_frame, stress_flags, alert_quantile=0.90, max_lead=7,
                       eval_splits=("test",), time_col=TIME_COL):
    """How early does the PRI flag real stress episodes? (plan.md lead-time analogue)

    Alert threshold = train-split PRI quantile. Events = stress-EPISODE starts (a
    stress day preceded by a calm day — consecutive stress days count once) inside
    eval_splits. For each event: was any of the preceding max_lead days an alert
    day, and how many days before the event did the first alert fire? Also reports
    the false-alert rate (alert days in eval_splits with no stress within max_lead).

    eval_splits defaults to TEST ONLY: val fit the meta weights and early-stopped
    the sequence models, so scoring it here would grade tuning data. A lead equal
    to max_lead usually means the window is saturated by persistent alerts —
    compare n_alert_days against the split length before quoting it.

    Returns (events_table, summary_dict).
    """
    f = pri_frame.sort_values(time_col, kind="stable").reset_index(drop=True)
    pri = f["score"].to_numpy(dtype=float)
    split = f["split"].to_numpy()
    stress = np.asarray(stress_flags).astype(int)
    thr = float(np.nanquantile(pri[split == "train"], alert_quantile))
    alert = pri >= thr

    starts = [i for i in range(len(stress))
              if stress[i] == 1 and (i == 0 or stress[i - 1] == 0)
              and split[i] in eval_splits]
    rows = []
    for i in starts:
        window = alert[max(0, i - max_lead):i]
        hit = bool(window.any())
        lead = int(len(window) - np.argmax(window)) if hit else None
        rows.append({time_col: f[time_col].iloc[i], "split": split[i],
                     "pri_on_day": round(pri[i], 1), "alerted_before": hit,
                     "lead_days": lead, "alert_same_day": bool(alert[i])})
    events = pd.DataFrame(rows)

    eval_mask = np.isin(split, eval_splits)
    alert_idx = np.flatnonzero(alert & eval_mask)
    false_alerts = sum(1 for i in alert_idx
                       if stress[i:min(len(stress), i + max_lead + 1)].sum() == 0)
    leads = [r["lead_days"] for r in rows if r["lead_days"] is not None]
    summary = {
        "alert_threshold_pri": round(thr, 1),
        "n_stress_episodes": len(rows),
        "n_alerted_before": int(sum(r["alerted_before"] for r in rows)),
        "hit_rate": round(float(np.mean([r["alerted_before"] for r in rows])), 3)
        if rows else float("nan"),
        "median_lead_days": float(np.median(leads)) if leads else float("nan"),
        "mean_lead_days": round(float(np.mean(leads)), 2) if leads else float("nan"),
        "n_alert_days": int(len(alert_idx)),
        "n_false_alerts": int(false_alerts),
        "false_alert_rate": round(false_alerts / len(alert_idx), 3)
        if len(alert_idx) else float("nan"),
    }
    return events, summary
