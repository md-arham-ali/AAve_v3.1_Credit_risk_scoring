"""Modeling stage — ensembles of the trained models, chosen on out-of-fold scores.

An out-of-fold (OOF) score comes from a model that never saw its row, and every OOF row is a
TRAIN row. So the blend method, its members and its weights are all learned on train only,
like every other choice in this stage: val and test take no part.
Runners return OOF scores with oof=True (res["oof"]).
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from model_config import HORIZONS, RANDOM_STATE, STRESS_TARGETS, TIME_COL, rank_normalize
from model_evaluation import evaluate_classification, bootstrap_auc_ci, paired_folds, _horizon

BLEND_METHODS = ("mean", "cv_weighted", "greedy", "stack")


# =========================================================================== #
# The OOF matrix
# =========================================================================== #

def oof_matrix(results, horizon=HORIZONS[0]):
    """One row per out-of-fold train row, one score column per model.
    results: {label: result dict with res["oof"]} (a bare OOF DataFrame also works).
    A model with no score for a row (a sequence model has no window for the first rows) stays NaN there.
    Returns: DataFrame time_bucket, fold, y_true, <label>...; models without this horizon are left out."""
    wide, left_out = None, []
    for label, res in results.items():
        oof = res if isinstance(res, pd.DataFrame) else res.get("oof")
        if not isinstance(oof, pd.DataFrame) or not (oof["horizon"] == horizon).any():
            left_out.append(label)
            continue
        part = oof.loc[oof["horizon"] == horizon, [TIME_COL, "fold", "y_true", "score"]].rename(columns={"score": label})
        wide = part if wide is None else wide.merge(part, on=[TIME_COL, "fold", "y_true"], how="outer")
    if wide is None:
        raise ValueError(f"no result holds out-of-fold scores for {horizon}: run the models with oof=True")
    if left_out:
        print(f" [{horizon}] no OOF scores, left out: {left_out}")
    return wide.sort_values([TIME_COL], kind="stable").reset_index(drop=True)


def _models(wide):
    return [c for c in wide.columns if c not in (TIME_COL, "fold", "y_true")]


def rank_by_fold(wide):
    """Each model's scores as ranks in [0, 1] inside each fold's test part. Models score on different
    scales (a probability, a hazard, a forecast); ranks put them on one, and reading no label keeps
    it honest. Returns: the same frame with ranked score columns (NaN stays NaN)."""
    out = wide.copy()
    cols = _models(wide)
    out[cols] = wide.groupby("fold")[cols].rank(pct=True)
    return out


def _blend(ranked, weights):
    """Weighted mean of the ranked columns, row by row; a missing component is left out of its row."""
    cols = list(weights)
    w = np.array([weights[c] for c in cols], dtype=float)
    vals = ranked[cols].to_numpy(dtype=float)
    mask = ~np.isnan(vals)
    denom = (mask * np.abs(w)).sum(axis=1)
    return np.nansum(vals * w, axis=1) / np.where(denom > 0, denom, np.nan)


def _auc(y, s):
    ok = ~np.isnan(s)
    return float(roc_auc_score(y[ok], s[ok])) if ok.any() and len(np.unique(y[ok])) > 1 else float("nan")


# =========================================================================== #
# Weights for each method, learned on a block of OOF rows
# =========================================================================== #

def blend_weights(ranked, method="mean", members=None, floor=0.02, greedy_steps=20, C=1.0):
    """The weights one blend method learns from a block of ranked OOF rows.
    mean         equal weights
    cv_weighted  weight ~ max(AUC - 0.5, floor): the PRI rule, with the AUC read on these rows
    greedy       forward selection with replacement (Caruana): add the member that lifts the blend's
                 AUC most, up to greedy_steps times; the weight is how often a member was added
    stack        logistic regression of the label on the ranked scores (missing = 0.5); weights can
                 be negative, and it has the most room to overfit
    Returns: {member: weight}."""
    members = members or _models(ranked)
    y = ranked["y_true"].to_numpy(dtype=float)
    if method == "mean":
        return {m: 1.0 / len(members) for m in members}
    if method == "cv_weighted":
        raw = {m: max(_auc(y, ranked[m].to_numpy(dtype=float)) - 0.5, floor) for m in members}
        total = sum(raw.values())
        return {m: v / total for m, v in raw.items()}
    if method == "greedy":
        counts, best_counts, best_auc = {m: 0 for m in members}, None, -1.0
        for _ in range(greedy_steps):
            trial = {m: _auc(y, _blend(ranked, {**{k: v for k, v in counts.items() if v}, m: counts[m] + 1}))
                     for m in members}
            pick = max(trial, key=lambda m: -1.0 if np.isnan(trial[m]) else trial[m])
            counts[pick] += 1
            if trial[pick] > best_auc + 1e-6:
                best_auc, best_counts = trial[pick], dict(counts)
        total = sum(best_counts.values())
        return {m: v / total for m, v in best_counts.items() if v}
    if method == "stack":
        X = ranked[members].fillna(0.5).to_numpy(dtype=float)
        model = LogisticRegression(C=C, max_iter=2000, random_state=RANDOM_STATE).fit(X, y.astype(int))
        return {m: float(c) for m, c in zip(members, model.coef_[0])}
    raise ValueError(f"method={method!r}: use one of {BLEND_METHODS}")


# =========================================================================== #
# Which blend? Scored fold by fold, weights from EARLIER folds only
# =========================================================================== #

def compare_blends(results, horizon=HORIZONS[0], members=None, methods=BLEND_METHODS):
    """Every blend method against the single models, on the out-of-fold rows of one horizon.
    A method that learns weights (cv_weighted, greedy, stack, and "best single", which learns which
    model to trust) learns them on the folds BEFORE the one it is scored on, so fold 1 has no score
    for those: compare on the paired columns (d_vs_ref, wins), not on a mean over different folds.
    The reference (first row) is the equal-weight mean, the blend that learns nothing.
    Returns: one row per blend and per single model — cv_auc, fold_k, n_members, paired against "mean"."""
    ranked = rank_by_fold(oof_matrix(results, horizon))
    members = members or _models(ranked)
    folds = sorted(ranked["fold"].unique())
    y = ranked["y_true"].to_numpy(dtype=float)
    is_fold = {k: (ranked["fold"] == k).to_numpy() for k in folds}

    def scored(weights_for):
        out = {}
        for k in folds:
            w = weights_for(k)
            out[f"fold_{k}"] = _auc(y[is_fold[k]], _blend(ranked[is_fold[k]], w)) if w else float("nan")
        return out

    earlier = lambda k: ranked[ranked["fold"] < k]
    rows = [{"blend": "mean", "n_members": len(members), **scored(lambda k: blend_weights(ranked, "mean", members))}]
    best = lambda k: max(members, key=lambda m: _auc(earlier(k)["y_true"].to_numpy(float), earlier(k)[m].to_numpy(float)))
    rows.append({"blend": "best single (so far)", "n_members": 1,
                 **scored(lambda k: {best(k): 1.0} if k > folds[0] else None)})
    for method in methods:
        if method == "mean":
            continue
        rows.append({"blend": method, "n_members": len(members),
                     **scored(lambda k, method=method: blend_weights(earlier(k), method, members) if k > folds[0] else None)})
    for m in members:
        rows.append({"blend": f"single: {m}", "n_members": 1, **scored(lambda k, m=m: {m: 1.0})})
    table = pd.DataFrame(rows)
    fold_cols = [c for c in table.columns if c.startswith("fold_")]
    table.insert(1, "cv_auc", table[fold_cols].mean(axis=1, skipna=True))
    table = paired_folds(table, ref=0)
    print(f" [{horizon}] {len(members)} member(s), {len(folds)} fold(s): mean blend {table['cv_auc'].iloc[0]:.3f}, "
          f"best single {table.loc[table['blend'].str.startswith('single'), 'cv_auc'].max():.3f}")
    return table


def score_correlation(results, horizon=HORIZONS[0]):
    """Spearman correlation between the models' out-of-fold scores. A blend gains from members that
    are right in different places: scores correlated near 1 say the same thing, so averaging them
    cannot add much. Returns: the correlation matrix."""
    wide = oof_matrix(results, horizon)
    return wide[_models(wide)].corr(method="spearman")


# =========================================================================== #
# The final blend on the full timeline
# =========================================================================== #

def run_ensemble(results, data, *, method="mean", members=None, target=STRESS_TARGETS[0], name="ensemble",
                 save=True):
    """Blend the members' full-timeline scores for `target` into one risk score.
    Weights: blend_weights(method) on ALL the out-of-fold rows of the target's horizon (train only).
    Scores: each member's predictions, rank-normalised against its own VAL scores (as the PRI does:
    val scores are out of sample, and no label is read), then the weighted mean.
    members: labels to use (default every result that has both OOF scores and predictions).
    Returns: result dict (kind "meta"), evaluated on the target like any classifier."""
    horizon = _horizon(target)
    usable = {k: r for k, r in results.items() if isinstance(r, dict) and isinstance(r.get("oof"), pd.DataFrame)
              and isinstance(r.get("predictions"), pd.DataFrame) and (r["oof"]["horizon"] == horizon).any()}
    members = list(members) if members else list(usable)
    missing = [m for m in members if m not in usable]
    if missing:
        raise ValueError(f"no OOF scores or predictions for {missing}: run those models with oof=True")
    ranked = rank_by_fold(oof_matrix({m: usable[m] for m in members}, horizon))
    weights = blend_weights(ranked, method, members)

    frame = None
    for m in weights:
        preds = usable[m]["predictions"]
        ref = preds.loc[preds["split"] == "val", "score"]
        col = pd.DataFrame({TIME_COL: preds[TIME_COL], "split": preds["split"],
                            m: rank_normalize(preds["score"], ref)})
        frame = col if frame is None else frame.merge(col, on=[TIME_COL, "split"], how="outer")
    frame = frame.sort_values(TIME_COL, kind="stable").reset_index(drop=True)
    frame["score"] = _blend(frame, weights)
    y_true = pd.concat([pd.DataFrame({TIME_COL: data["times"][target][s], "split": s, "y_true": data["y"][target][s]})
                        for s in ("train", "val", "test")], ignore_index=True)
    frame = frame.merge(y_true, on=[TIME_COL, "split"], how="left").dropna(subset=["score", "y_true"])

    oof_score = _blend(ranked, weights)
    oof_y = ranked["y_true"].to_numpy(dtype=float)
    cv_folds = [_auc(oof_y[(ranked["fold"] == k).to_numpy()], oof_score[(ranked["fold"] == k).to_numpy()])
                for k in sorted(ranked["fold"].unique())]
    part = lambda s: frame.loc[frame["split"] == s]
    out = {"name": name, "kind": "meta", "status": "trained", "proper_probs": False, "method": method,
           "weights": {k: float(v) for k, v in weights.items()}, "n_components": len(weights),
           "component_hashes": {m: usable[m].get("settings_hash") for m in weights},
           "targets": {target: {s: evaluate_classification(part(s)["y_true"], part(s)["score"], proper_probs=False)
                                for s in ("train", "val", "test")}},
           # in-sample for the weights (they were learned on these rows): compare_blends is the honest read
           "cv": {"mean": float(np.nanmean(cv_folds)), "std": float(np.nanstd(cv_folds)), "folds": cv_folds},
           "test_auc_ci": bootstrap_auc_ci(part("test")["y_true"], part("test")["score"]),
           "predictions": frame[[TIME_COL, "split", "y_true", "score"]].reset_index(drop=True),
           "trial": not save}
    m = out["targets"][target]
    shown = (f"val/test ROC-AUC {m['val']['roc_auc']:.3f}/{m['test']['roc_auc']:.3f}" if save else
             f"val ROC-AUC {m['val']['roc_auc']:.3f} (test hidden on a trial run)")
    print(f" {name} [{horizon}] {method}: {len(weights)} member(s) | {shown}")
    print(" weights:", {k: round(v, 3) for k, v in sorted(weights.items(), key=lambda kv: -abs(kv[1]))})
    return out
