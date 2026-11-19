"""Evidence plumbing for the tuning decisions: the same experiment run two ways and scored on the same periods.

Used by notebooks/model_tuning_evidence.ipynb. Nothing here saves a result or changes the split on disk.
- refit_inputs / load_data : the SAME split (boundaries, thresholds, folds) on a different column pool, in memory
- period_scores            : one model on four periods — the walk-forward folds and val (fit on all of train)
- score_table              : several period_scores side by side, paired against a reference
- calendar_columns / price_move_columns : extra columns for a check, known at t
- run_setup / setup_table / paired_ci    : a frozen setup through the public runners, and one setup against another
"""
import contextlib
import glob
import io
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from model_config import REG_TARGET, STRESS_TARGETS, TIME_COL
import model_split as msp
import model_evaluation as mev
import model_persistence as mps
import model_training as mtr
import model_series as mser
import model_ensemble as mens
import model_benchmarks as mbench
from model_zoo import CLASSIFIER_SPECS, REGRESSOR_SPECS, build_model, predict_scores, _xgb_overrides

PERIODS_NOTE = "fold_k = walk-forward folds on train; val = fit on all of train, scored on val; four = their mean"


# =========================================================================== #
# The same split on another column pool
# =========================================================================== #

def refit_inputs(meta, sources, priority=None, max_nan=0.5, max_corr=0.95):
    """A copy of the split definition with DF_model read from `sources` (a path, or a list of frames
    joined on time: model_split.load_input). Boundaries, embargo, stress thresholds and folds are kept
    as they are, so the rows and labels of every split are identical; only the column selection and
    the scaler are fitted again, on the same train rows model_split.ipynb uses (the strictest horizon's).
    Nothing is written: pass the result to load_data.
    Returns: the new meta dict (JSON-plain, like a reloaded split_meta.json)."""
    batch = list(meta["inputs"]["batch"])
    df_model = msp.load_input(sources, batch=batch)
    df_target = msp.load_input(meta["inputs"]["DF_target"], batch=batch)
    feat = msp.prepare_features(df_model, {"fill": meta["fill"], "non_features": meta["non_features"]})
    cols = msp.feature_columns(feat, meta["non_features"])
    bounds = {k: pd.Timestamp(v) for k, v in meta["boundaries"].items()}
    strictest = list(meta["embargo"])[-1]
    assign = msp.assign_split(df_target, bounds, strictest, warmup=meta["warmup"])
    train_stamps = set(msp.time_of(df_target)[(assign == "train").to_numpy()])
    train = feat[msp.time_of(feat).isin(train_stamps)]
    selection = msp.fit_column_selection(train, cols, max_nan=max_nan, max_corr=max_corr, priority=priority)
    new = {**meta,
           "inputs": {**meta["inputs"], "DF_model": sources, "DF_model_fingerprint": msp.frame_fingerprint(df_model)},
           "feature_cols": cols, "selection": selection, "scaler_params": msp.fit_scaler(train, selection["kept"])}
    return json.loads(json.dumps(msp._jsonable(new)))


def load_data(batch, meta):
    """model_persistence.load_model_data on an in-memory split definition (refit_inputs).
    The data dict carries evidence_only=True: its results are for comparison, never for model_results/."""
    data = mps.load_model_data(batch, meta=meta)
    data["evidence_only"] = True
    return data


def ranked(data, target=STRESS_TARGETS[0]):
    """Every column, strongest first, by its own |AUC - 0.5| on the train rows (rank_features "alone")."""
    return mtr.top_n_features(data, target, n=None, method="alone")


# =========================================================================== #
# One model on four periods
# =========================================================================== #

def _stamps(frame):
    return pd.DatetimeIndex(msp.time_of(frame))


def _with_extra(frame, extra):
    """frame + the extra columns, matched by time (extra: DataFrame indexed by UTC stamp, or None)."""
    if extra is None:
        return frame
    add = extra.reindex(_stamps(frame))
    out = frame.copy()
    for c in add.columns:
        if c in out.columns:
            raise ValueError(f"extra column {c!r} is already a feature")
        out[c] = add[c].to_numpy()
    return out


def recency_weights(stamps, half_life_days):
    """Sample weights that halve every half_life_days back from the newest fit row; mean 1."""
    age = np.asarray((stamps.max() - stamps).total_seconds(), dtype=float) / 86400.0
    w = np.power(0.5, age / float(half_life_days))
    return w / w.mean()


def period_scores(name, data, target=STRESS_TARGETS[0], params=None, cols=None, *, top_n=None, kind="classifier",
                  half_life_days=None, train_days=None, extra=None):
    """One model scored on four periods: each walk-forward fold of train, then val with the model fit on
    all of train. Test is never read. Everything fitted (scaler, ranking, model) reads the fit rows only.
    cols: feature subset (default every feature, plus every `extra` column).
    top_n: each period ranks `cols` on its own fit rows (column's own AUC) and keeps the top N.
    kind "regressor": fit the registry REGRESSOR on the continuous 24h score and rank the binary label by
        its prediction — the regression loss as a classifier's loss (24h target only).
    half_life_days: recency sample weights. train_days: fit on the last N days of the fit rows only.
    extra: more columns by time (calendar_columns, price_move_columns); blanks are filled with the fit median.
    Returns: {"folds": [auc per fold], "val": auc}."""
    spec = (CLASSIFIER_SPECS if kind == "classifier" else REGRESSOR_SPECS)[name]
    train_t = _with_extra(data["frames_t"][target]["train"], extra)
    val_t = _with_extra(data["frames_t"][target]["val"], extra)
    y_tr_all, y_val = data["y"][target]["train"], data["y"][target]["val"]
    fit_y = y_tr_all
    if kind == "regressor":
        fit_y = data["y"][REG_TARGET]["train"]
        if target != data["stress_targets"][0] or len(fit_y) != len(y_tr_all):
            raise ValueError("kind='regressor' needs the 24h target: the continuous score exists for it only")
    cols = list(cols) if cols is not None else list(data["feature_cols"]) + (list(extra.columns) if extra is not None else [])
    stamps = _stamps(train_t)
    parts = [(tr, train_t.iloc[te], y_tr_all[te]) for tr, te in mev._cv_folds(data, target)]
    parts.append((np.arange(len(train_t)), val_t, y_val))
    scores = []
    for fit_rows, test_frame, y_test in parts:
        if train_days:
            fit_rows = fit_rows[stamps[fit_rows] >= stamps[fit_rows].max() - pd.Timedelta(days=train_days)]
        fit_frame, y_fit = train_t.iloc[fit_rows], fit_y[fit_rows]
        if len(np.unique(y_tr_all[fit_rows])) < 2 or len(np.unique(y_test)) < 2:
            scores.append(float("nan"))
            continue
        sc = msp.fit_scaler(fit_frame, cols)
        use = cols
        if top_n and top_n < len(cols):
            raw = msp.apply_scaler(fit_frame, sc, cols, standardize=False).to_numpy()
            use = mev.rank_features(raw, y_tr_all[fit_rows], cols, method="alone")[:top_n]
        # a column blank on every fit row has no median to fill with: it becomes a constant 0 in both parts
        X_fit = np.nan_to_num(msp.apply_scaler(fit_frame, sc, use, standardize=spec["scaled"]).to_numpy(), nan=0.0)
        X_test = np.nan_to_num(msp.apply_scaler(test_frame, sc, use, standardize=spec["scaled"]).to_numpy(), nan=0.0)
        auto = _xgb_overrides(y_fit) if (name == "xgboost" and kind == "classifier") else {}
        model = build_model(name, kind, **{**auto, **(params or {})})
        kw = {"sample_weight": recency_weights(stamps[fit_rows], half_life_days)} if half_life_days else {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_fit, y_fit, **kw)
        s = model.predict(X_test) if kind == "regressor" else predict_scores(model, X_test, X_fit)
        scores.append(float(roc_auc_score(y_test, s)))
    return {"folds": scores[:-1], "val": scores[-1]}


def score_table(rows, ref=None):
    """{label: period_scores result} -> one row per label: cv (mean of folds), fold_k, val, four (mean of
    the four periods), and against `ref` (default the first label): d_four and better_in (periods won)."""
    labels = list(rows)
    ref = ref or labels[0]
    base = np.array(rows[ref]["folds"] + [rows[ref]["val"]], dtype=float)
    out = []
    for lab in labels:
        p = np.array(rows[lab]["folds"] + [rows[lab]["val"]], dtype=float)
        d = p - base
        out.append({"setting": lab, "cv": float(np.nanmean(p[:-1])),
                    **{f"fold_{i}": float(x) for i, x in enumerate(p[:-1], 1)}, "val": float(p[-1]),
                    "four": float(np.nanmean(p)), "d_four": float(np.nanmean(d)) if lab != ref else 0.0,
                    "better_in": f"{int(np.nansum(d > 0))}/{len(d)}" if lab != ref else "ref"})
    return pd.DataFrame(out).round(4)


# =========================================================================== #
# Extra columns for a check (known at t)
# =========================================================================== #

def calendar_columns(stamps, horizon="24h", step="2h"):
    """Calendar-only columns for the rows at `stamps`: the share of the next `horizon` that falls on a
    Saturday or Sunday (UTC), and the hour of the week as sin / cos. Pure clock: nothing to leak.
    Returns: DataFrame indexed by stamp."""
    t = pd.DatetimeIndex(stamps)
    steps = int(pd.Timedelta(horizon) / pd.Timedelta(step))
    ahead = np.mean([((t + pd.Timedelta(step) * (i + 1)).dayofweek >= 5) for i in range(steps)], axis=0)
    how = (t.dayofweek * 24 + t.hour).to_numpy()
    return pd.DataFrame({"cal_weekend_ahead": ahead, "cal_how_sin": np.sin(2 * np.pi * how / 168),
                         "cal_how_cos": np.cos(2 * np.pi * how / 168)}, index=t)


def price_move_columns(stamps, symbols=("WETH", "WBTC"), folder="query_result_data", step="2h"):
    """READ-ONLY diagnostic: price moves from the normalised oracle extracts (batch_*/oracle_price_*.csv),
    each closed at t — log return over 24h / 3d, the worst 2h return of the last 24h / 3d, realised
    volatility over 24h, and the drawdown from the trailing 7d high. Building such features for the
    pipeline is the data pipeline's work (bugs.md 10.1); this exists only to ask whether they would help.
    Returns: DataFrame indexed by stamp (blank until a window has rows)."""
    files = sorted(glob.glob(os.path.join(folder, "batch_*", "oracle_price_*.csv")))
    if not files:
        raise FileNotFoundError(f"no oracle extract under {folder}/batch_*/")
    px = pd.concat([pd.read_csv(f, usecols=["time_bucket", "symbol", "avg_price_usd"]) for f in files])
    px["t"] = pd.to_datetime(px["time_bucket"].astype(str).str.replace(" UTC", "", regex=False), utc=True)
    t = pd.DatetimeIndex(stamps)
    grid = pd.date_range(px["t"].min(), px["t"].max(), freq=step, tz="UTC")
    per = lambda span: int(pd.Timedelta(span) / pd.Timedelta(step))
    out = {}
    for sym in symbols:
        p = px[px["symbol"] == sym].groupby("t")["avg_price_usd"].mean().reindex(grid).ffill()
        lp = np.log(p)
        r = lp.diff()
        out[f"px_{sym}_ret_24h"] = lp - lp.shift(per("24h"))
        out[f"px_{sym}_ret_3d"] = lp - lp.shift(per("3d"))
        out[f"px_{sym}_minret_24h"] = r.rolling(per("24h"), min_periods=6).min()
        out[f"px_{sym}_minret_3d"] = r.rolling(per("3d"), min_periods=18).min()
        out[f"px_{sym}_vol_24h"] = r.rolling(per("24h"), min_periods=6).std()
        out[f"px_{sym}_dd_7d"] = lp - lp.rolling(per("7d"), min_periods=12).max()
    return pd.DataFrame(out, index=grid).reindex(t)


def columns_alone(data, columns=None, extra=None, target=STRESS_TARGETS[0], splits=("val",)):
    """Each column as the score, no model: its AUC on every fold's test part and on each of `splits`.
    Sides are kept as they are (under 0.5 = the column ranks stress low in that period).
    Returns: one row per column — fold_k, then one column per split."""
    train_t = _with_extra(data["frames_t"][target]["train"], extra)
    y = data["y"][target]["train"]
    cols = list(columns) if columns is not None else (list(extra.columns) if extra is not None else list(data["feature_cols"]))
    auc = lambda yy, x: (float(roc_auc_score(yy, np.nan_to_num(x, nan=np.nanmedian(x))))
                         if len(np.unique(yy)) > 1 and not np.isnan(x).all() else float("nan"))
    rows = []
    for c in cols:
        row = {"column": c}
        for i, (_, te) in enumerate(mev._cv_folds(data, target), 1):
            row[f"fold_{i}"] = auc(y[te], train_t[c].to_numpy(dtype=float)[te])
        for s in splits:
            frame = _with_extra(data["frames_t"][target][s], extra)
            row[s] = auc(data["y"][target][s], frame[c].to_numpy(dtype=float))
        rows.append(row)
    return pd.DataFrame(rows).round(3)


# =========================================================================== #
# A frozen setup through the public runners, and one setup against another
# =========================================================================== #

def run_member(name, data, spec):
    """One model of a frozen setup, through its own public runner, as a trial (save=False) with OOF scores.
    spec: {"24h": {"cols": [...], "params": {...}}, "3d": ..., "7d": ...} — a horizon left out is not run.
    Sequence models stop on the tail of train (val takes no part in their fit), or, on a span_data view, on its held val
    tail (every train + val row is then a fit row). Returns: the result dict."""
    horizons = [h for h in ("24h", "3d", "7d") if h in spec]
    by_h = dict(zip(("24h", "3d", "7d"), data["stress_targets"]))
    targets = [by_h[h] for h in horizons]
    feats = {h: list(spec[h]["cols"]) for h in horizons}
    fixed = {h: {k: (tuple(v) if k in ("order", "seasonal_order") else v) for k, v in spec[h].get("params", {}).items()}
             for h in horizons}
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name in CLASSIFIER_SPECS:
            return mtr.run_classifier(name, data, targets=targets, fixed=fixed, features=feats, tune=False,
                                      save=False, oof=True)
        if name in ("lstm", "transformer"):
            # a config read back from a result carries stop="train_tail" already: it is passed below, once
            fixed = {h: {k: v for k, v in fixed[h].items() if k != "stop"} for h in horizons}
            first = fixed[horizons[0]]
            common = {k: v for k, v in first.items() if all(fixed[h].get(k) == v for h in horizons)}
            per = {h: {k: v for k, v in fixed[h].items() if k not in common} for h in horizons}
            runner = mtr.run_lstm if name == "lstm" else mtr.run_transformer
            # on the official data a net stops on the tail of train (val takes no part in its fit). On a span_data view
            # val's merged rows ARE the tail of train, so "train_tail" would stop on them and the net would not be fitted
            # on val at all: there it stops on the held val tail ("val"), and every train + val row is a fit row
            stop = "val" if data.get("span") else "train_tail"
            return runner(data, targets=targets, per_horizon={h: d for h, d in per.items() if d} or None,
                          features=feats, save=False, oof=True, stop=stop, **common)
        if name == "cox_ph":
            return mtr.run_cox(data, target=targets[0], features=feats[horizons[0]], save=False, oof=True,
                               **fixed[horizons[0]])
        runner = {"sarimax": mser.run_sarimax, "hmm_regime": mser.run_hmm}[name]
        parts = [runner(data, targets=[by_h[h]], features=feats[h], save=False, oof=True, **fixed[h]) for h in horizons]
        res = parts[0]
        for r in parts[1:]:
            res["targets"].update(r["targets"])
            res["oof"] = pd.concat([res["oof"], r["oof"]], ignore_index=True)
        return res


def run_setup(data, specs, label=""):
    """Every member of a frozen setup (run_member), then the equal-weight 24h ensemble of them.
    Returns: {member: result, ..., "ensemble": result}."""
    results = {}
    for name, spec in specs.items():
        results[name] = run_member(name, data, spec)
        print(f" {label} {name}: done")
    with contextlib.redirect_stdout(io.StringIO()):
        results["ensemble"] = mens.run_ensemble(results, data, method="mean")
    return results


def _metrics(res, target):
    return res["targets"][target] if target in res.get("targets", {}) else res.get("metrics")


def setup_table(results, data, horizon="24h"):
    """One row per member (and the ensemble): cv (mean fold AUC of its out-of-fold scores), val, test."""
    target = dict(zip(("24h", "3d", "7d"), data["stress_targets"]))[horizon]
    rows = []
    for name, res in results.items():
        m = _metrics(res, target)
        if not m:
            continue
        cv = float("nan")
        oof = res.get("oof")
        if isinstance(oof, pd.DataFrame):
            o = oof[oof["horizon"] == horizon]
            f = [roc_auc_score(g["y_true"], g["score"]) for _, g in o.groupby("fold") if g["y_true"].nunique() > 1]
            cv = float(np.mean(f)) if f else float("nan")
        rows.append({"model": name, "cv": cv, "val": m["val"]["roc_auc"], "test": m["test"]["roc_auc"]})
    return pd.DataFrame(rows).round(4)


def paired_ci(res_a, res_b, splits=("test",), **kw):
    """model_benchmarks.paired_auc_ci of A against B over one split or several taken together
    (("val", "test") = the whole out-of-sample span): the same moving blocks are resampled for both.
    Returns: paired_auc_ci's dict."""
    def over(res):
        p = res["predictions"]
        p = p[p["split"].isin(splits)].copy()
        p["split"] = "span"
        return {"name": res.get("name"), "predictions": p}
    return mbench.paired_auc_ci(over(res_a), over(res_b), split="span", **kw)


def stamps(data, target=STRESS_TARGETS[0], splits=("train", "val", "test")):
    """Every row stamp of the given splits of one target, in time order."""
    return pd.DatetimeIndex(np.concatenate([_stamps(data["frames_t"][target][s]).to_numpy() for s in splits]), tz="UTC")


def past_target_columns(data, lags=("0h", "24h")):
    """The target's own past as columns, for a check only (features and targets stay apart in DF_model):
    the continuous 24h liquidation score of the window that CLOSED at t (stamped t - 24h), and of earlier
    closed windows (lag further back). Read from data["label_history"], so nothing after t is used.
    Returns: DataFrame indexed by stamp."""
    hist = data["label_history"][REG_TARGET]
    labels, h = hist["labels"], hist["horizon"]
    t = stamps(data)
    return pd.DataFrame({f"past_liq_24h_lag{lag}": labels.reindex(t - h - pd.Timedelta(lag)).to_numpy() for lag in lags},
                        index=t)


def rank_agreement(table, by=("cv", "val", "test")):
    """Do the periods rank the models the same way? Spearman correlation between each pair of the score
    columns of a setup_table (ensemble row left out). 1 = same order, 0 = unrelated, below 0 = reversed."""
    t = table[table["model"] != "ensemble"]
    rows = [{"pair": f"{a} vs {b}", "spearman": float(t[a].corr(t[b], method="spearman")), "models": len(t)}
            for i, a in enumerate(by) for b in by[i + 1:]]
    return pd.DataFrame(rows).round(3)


# =========================================================================== #
# Round two (2026-11-16): the train + val protocol, and a ledger for test reads
# =========================================================================== #

def _cat(a, b):
    """Two blocks of one kind joined in order: DataFrames, arrays or lists."""
    if isinstance(a, pd.DataFrame):
        return pd.concat([a, b], ignore_index=True)
    if isinstance(a, np.ndarray):
        return np.concatenate([a, b])
    return list(a) + list(b)


def _take(block, keep):
    """The rows of a block where `keep` is True (DataFrame, array or list)."""
    if isinstance(block, pd.DataFrame):
        return block.loc[keep].reset_index(drop=True)
    if isinstance(block, np.ndarray):
        return block[keep]
    return [v for v, k in zip(block, keep) if k]


def span_data(data, hold=0.15, n_folds=5):
    """The same data with val folded into train, so every public runner fits on train + val and its
    walk-forward folds reach into val. Nothing is written and the split on disk is untouched.
    Per target: the first (1 - hold) of val's rows join train; the last `hold` of val stays as `val`,
    never fitted on, so stop="val" and the F1 threshold still have rows that no fit has seen. Fit rows
    whose label window (t, t + h] reaches the held tail are dropped (the fold rule: an embargo of one
    horizon). Test is exactly as it was, so a result on this data and the official result score the SAME test
    rows and can be paired. The scaler stays the one fitted on train alone (a leak-free choice).
    hold: share of val kept out (0.15 = the tail the sequence models stop on). n_folds: walk-forward folds on
    the new train block; 5 puts the last fold on the final stretch of val.
    Returns: a data dict like load_data's, flagged evidence_only, with data["span"] = what was done."""
    out = {**data, "n_folds": int(n_folds), "evidence_only": True,
           "splits": {}, "frames_t": {}, "times": {}, "X": {}, "y": {}}
    notes = {}
    for t in data["frames_t"]:
        h = data["label_history"][t]["horizon"]
        st_tr, st_va = _stamps(data["frames_t"][t]["train"]), _stamps(data["frames_t"][t]["val"])
        cut = int(round(len(st_va) * (1 - hold)))
        if not 0 < cut < len(st_va):
            raise ValueError(f"hold={hold}: the held tail of {_horizon_name(t)} would be empty or all of val")
        stamps = st_tr.append(st_va[:cut])
        keep = np.asarray(stamps + h < st_va[cut])                  # label windows end before the held tail starts
        keep_tr, keep_va = keep[:len(st_tr)], keep[len(st_tr):]

        def merge(block):
            return {"train": _cat(_take(block["train"], keep_tr), _take(_head(block["val"], cut), keep_va)),
                    "val": _tail(block["val"], cut), "test": block["test"]}
        out["splits"][t] = merge(data["splits"][t])
        out["frames_t"][t] = merge(data["frames_t"][t])
        out["times"][t] = merge(data["times"][t])
        out["y"][t] = merge(data["y"][t])
        out["X"][t] = {kind: merge(data["X"][t][kind]) for kind in ("raw", "scaled")}
        notes[t] = {"train": int(keep.sum()), "val_held": len(st_va) - cut, "dropped_by_embargo": int((~keep).sum())}
    out["span"] = {"hold": hold, "n_folds": int(n_folds), "rows": notes}
    first = data["stress_targets"][0]
    print(f" span data: {_horizon_name(first)} train {notes[first]['train']} (train + {1 - hold:.0%} of val), "
          f"val held out {notes[first]['val_held']}, test {len(data['frames_t'][first]['test'])}, "
          f"{n_folds} walk-forward folds")
    return out


def _head(block, n):
    return block.iloc[:n] if isinstance(block, pd.DataFrame) else block[:n]


def _tail(block, n):
    return block.iloc[n:].reset_index(drop=True) if isinstance(block, pd.DataFrame) else block[n:]


def _horizon_name(target):
    return str(target).rpartition("_")[2]


def span_folds(span, target=STRESS_TARGETS[0]):
    """Where the walk-forward folds of a span_data bundle sit in time: rows, last train stamp, test period
    and positives per fold (model_split.fold_summary). Use it to see that the last fold lies in val."""
    frame = span["frames_t"][target]["train"]
    folds = mev._cv_folds(span, target)
    return msp.fold_summary(msp.time_of(frame), folds, span["y"][target]["train"])


def fold_scores(res, horizon="24h"):
    """AUC of every fold of a result's out-of-fold scores (oof=True runs): {fold: auc}. A single-class fold is NaN."""
    o = res["oof"]
    o = o[o["horizon"] == horizon]
    return {int(k): (float(roc_auc_score(g["y_true"], g["score"])) if g["y_true"].nunique() > 1 else float("nan"))
            for k, g in o.groupby("fold")}


def compare_folds(results, ref=None, horizon="24h", min_gain=0.01):
    """Results of one protocol side by side on their folds, paired against `ref` (default the first):
    fold_k, mean, d_mean (mean paired change), d_ex_last (the same without the last fold: on a span view that
    fold lies in val, which some setups were picked on), wins (folds won), passes. A row passes when d_mean >= min_gain
    and it wins in at least two thirds of the folds (3 of 3 -> 2, 5 of 5 -> 4: the RF showcase rule,
    Descisions.md 2026-11-11); the reference never passes. results: {label: a result run with oof=True}."""
    labels = list(results)
    ref = ref or labels[0]
    base = fold_scores(results[ref], horizon)
    rows = []
    for lab in labels:
        f = fold_scores(results[lab], horizon)
        ks = sorted(f)
        d = np.array([f[k] - base.get(k, float("nan")) for k in ks])
        wins, need = int(np.nansum(d > 0)), int(np.ceil(2 * len(ks) / 3))
        d_mean = float(np.nanmean(d)) if lab != ref else 0.0
        rows.append({"setting": lab, **{f"fold_{k}": f[k] for k in ks}, "mean": float(np.nanmean(list(f.values()))),
                     "d_mean": d_mean, "d_ex_last": float(np.nanmean(d[:-1])) if lab != ref and len(d) > 1 else 0.0,
                     "wins": wins if lab != ref else 0,
                     "passes": bool(lab != ref and d_mean >= min_gain and wins >= need)})
    return pd.DataFrame(rows).round(4)


def same_spec(a, b):
    """Two frozen specs ({"24h": {"cols": [...], "params": {...}}}) are the same setup: compared as JSON, so a
    tuple and the list it came back as from a file are equal."""
    dump = lambda x: json.dumps(x, sort_keys=True, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return dump(a) == dump(b)


def freeze_choice(tables, ref="official"):
    """One setup per model, chosen by the folds alone (test is not read until the choice is made): the
    passing row with the largest d_mean in that model's compare_folds table, else the reference.
    tables: {model: compare_folds table}. Returns: {model: label}."""
    out = {}
    for model, t in tables.items():
        ok = t[t["passes"]]
        out[model] = str(ok.loc[ok["d_mean"].idxmax(), "setting"]) if len(ok) else ref
    return out


def read_test(label, results, data, ledger, horizon="24h"):
    """The test column of setup_table for one FROZEN setup, once. `ledger` is a dict the notebook keeps for the
    session: a label that is already in it is refused, so a setup cannot be re-read after a tweak (the rule in
    Descisions.md 2026-11-16 D-1). Copy each read into Descisions.md. Returns: setup_table's rows (cv, val, test)."""
    if label in ledger:
        raise RuntimeError(f"test was already read for {label!r} at {ledger[label]}: a frozen setup is read once. "
                           f"A changed setup needs a new label, and the change is a new read in Descisions.md")
    ledger[label] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    print(f" test read #{len(ledger)} for {label!r}: write it into Descisions.md")
    return setup_table(results, data, horizon=horizon)


ROUND_TWO_FILE = os.path.join("model_results", "round2_specs.json")      # model_results loaders skip a file with no "name"


def save_round_two(specs, path=ROUND_TWO_FILE):
    """The round-two candidates the training notebook froze, {model: {"24h": {"cols": [...], "params": {...}}}},
    written as JSON for the evidence notebook (tuples come back as lists: run_member turns SARIMAX's orders back).
    Returns: the path."""
    with open(path, "w") as fh:
        json.dump(specs, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f" wrote {len(specs)} round-two candidate(s) -> {path}")
    return path


def load_round_two(path=ROUND_TWO_FILE):
    """The candidates save_round_two wrote ({} when the training notebook has not been run to that cell)."""
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def round_two_summary(res_official, res_frozen, data_official, data_span, horizon="24h", target=0.72):
    """Official (train-only fit) against frozen (train + val fit), one row per model and the ensemble: cv on each
    protocol's own folds, val (the official's whole val; the frozen one's held tail), test, and the paired test
    difference with its CI (the two fits score the SAME test rows). `reaches` is a plain reading of test >= target:
    it says nothing about whether a score was picked on test (it was not: the setup is frozen on folds)."""
    off = setup_table(res_official, data_official, horizon).set_index("model")
    fro = setup_table(res_frozen, data_span, horizon).set_index("model")
    rows = []
    for m in fro.index:
        ci = paired_ci(res_frozen[m], res_official[m], splits=("test",))
        rows.append({"model": m, "cv official": off.loc[m, "cv"], "val official": off.loc[m, "val"],
                     "test official": off.loc[m, "test"], "cv frozen": fro.loc[m, "cv"],
                     "val tail frozen": fro.loc[m, "val"], "test frozen": fro.loc[m, "test"],
                     "d test": ci["diff"], "ci lo": ci["lo"], "ci hi": ci["hi"],
                     "reaches": bool(fro.loc[m, "test"] >= target)})
    return pd.DataFrame(rows).round(4)


# =========================================================================== #
# Round three (2026-11-17): stages read on the four periods, val as the target
# =========================================================================== #

def four_periods(results, data, ref=None, horizon="24h", min_val=0.02, min_cv=0.0, min_wins=3, gate="all"):
    """Results of one stage side by side on the four periods (the walk-forward folds of their OOF scores, and the
    official val), paired against `ref` (default the first label). No test column.
    gate "all": a row passes at d_val >= min_val, d_cv >= min_cv and a win in >= min_wins of the periods
    (Descisions.md 2026-11-17 D-1). gate "val": d_val >= min_val only (the column stage, where CV is optimistic).
    results: {label: a result run with oof=True}. Returns: one row per label."""
    target = dict(zip(("24h", "3d", "7d"), data["stress_targets"]))[horizon]
    labels = list(results)
    ref = ref or labels[0]

    def periods(res):
        f = fold_scores(res, horizon)
        return [f[k] for k in sorted(f)], float(_metrics(res, target)["val"]["roc_auc"])
    base_f, base_v = periods(results[ref])
    rows = []
    for lab in labels:
        f, v = periods(results[lab])
        d = np.array(f + [v]) - np.array(base_f + [base_v]) if len(f) == len(base_f) else np.full(len(f) + 1, np.nan)
        cv, d_cv, d_val = float(np.nanmean(f)), float(np.nanmean(d[:-1])), float(d[-1])
        wins = int(np.nansum(d > 0))
        ok = d_val >= min_val and (gate == "val" or (d_cv >= min_cv and wins >= min_wins))
        rows.append({"setting": lab, **{f"fold_{i}": x for i, x in enumerate(f, 1)}, "cv": cv, "val": v,
                     "d_cv": d_cv if lab != ref else 0.0, "d_val": d_val if lab != ref else 0.0,
                     "wins": f"{wins}/{len(d)}" if lab != ref else "ref", "passes": bool(lab != ref and ok)})
    return pd.DataFrame(rows).round(4)


def _row_spec(base, row, horizon="24h"):
    """A stage row applied to a base spec: "cols" replaces the columns; "params" replaces every setting; "set" changes
    some (a None value removes the setting, so the runner's default applies)."""
    b = base[horizon]
    params = dict(row["params"]) if "params" in row else dict(b.get("params", {}))
    for k, v in row.get("set", {}).items():
        if v is None:
            params.pop(k, None)
        else:
            params[k] = v
    return {horizon: {"cols": list(row.get("cols", b["cols"])), "params": params}}


def _row_label(row):
    if "label" in row:
        return str(row["label"])
    parts = [f"{k}={v}" for k, v in row.get("set", {}).items()]
    if "cols" in row:
        parts.insert(0, f"{len(row['cols'])} cols")
    return ", ".join(parts) or "row"


def stage(name, data, base, rows, *, gate="all", infold_pool=None, infold_tol=0.01, min_val=None, title="", horizon="24h",
          after=None):
    """One front of one model: the base spec ({"24h": {"cols": [...], "params": {...}}}) and every row applied to it,
    each run through the public runner (run_member: trial, save=False, OOF scores), read on the four periods.
    rows: [{"label": ..., "cols": [...], "set": {...}} or {"params": {...}}] — see _row_spec.
    gate: "all" (the full rule) or "val" (the column stage). min_val: the val bar (default 0.02).
    infold_pool: classifiers only — the column pool for an honest CV of a top-N row: the same params on
        lab.period_scores(top_n=n), each fold ranking the pool on its own rows. A row gives its n as "n" (default its
        column count). Adds cv_infold and infold_ok (not more than infold_tol below the base's in-fold CV: 0.01, the
        noise bar of a fold mean); under gate "val" a row also needs infold_ok to pass.
    after: the previous stage of the same model (best_of applied): its pick's result is reused as this base's, so
        the base is not fitted twice. Its spec must equal `base`.
    Returns: {"name", "specs", "results", "table", "pick": None} (best_of fills "pick")."""
    target = dict(zip(("24h", "3d", "7d"), data["stress_targets"]))[horizon]
    specs = {"base": base}
    for r in rows:
        lab = _row_label(r)
        while lab in specs:
            lab += "'"
        specs[lab] = _row_spec(base, r, horizon)
    ns = {"base": len(base[horizon]["cols"]), **{l: r.get("n", len(specs[l][horizon]["cols"]))
                                                   for l, r in zip(list(specs)[1:], rows)}}
    print(f" {name} {title}: {len(specs) - (after is not None)} run(s)")
    results = {}
    if after is not None:
        prev = after["pick"] or "base"
        if not same_spec(after["specs"][prev], base):
            raise ValueError("after=: the previous stage's pick is not this stage's base")
        results["base"] = after["results"][prev]
    for lab, spec in specs.items():
        if lab in results:
            continue
        t0 = pd.Timestamp.now()
        results[lab] = run_member(name, data, spec)
        print(f"   {lab}: {(pd.Timestamp.now() - t0).total_seconds():.0f}s")
    kw = {"min_val": min_val} if min_val is not None else {}
    table = four_periods(results, data, ref="base", horizon=horizon, gate=gate, **kw)
    if infold_pool is not None and name in CLASSIFIER_SPECS:
        inf = {lab: float(np.nanmean(period_scores(name, data, target, spec[horizon]["params"], cols=infold_pool,
                                                   top_n=ns[lab])["folds"]))
               for lab, spec in specs.items()}
        table["cv_infold"] = table["setting"].map(inf).round(4)
        table["infold_ok"] = table["cv_infold"] >= inf["base"] - infold_tol
        if gate == "val":
            table["passes"] = table["passes"] & table["infold_ok"]
    return {"name": name, "specs": specs, "results": results, "table": table, "pick": None}


def best_of(st):
    """The greedy step, by rule (no hand edit): the passing row of a stage with the largest d_val, else the base.
    Stores the label in st["pick"]. Returns: the spec to use as the next stage's base."""
    t = st["table"]
    ok = t[t["passes"]]
    st["pick"] = str(ok.loc[ok["d_val"].idxmax(), "setting"]) if len(ok) else "base"
    print(f" {st['name']}: pick -> {st['pick']}")
    return st["specs"][st["pick"]]


def next_stage(stages, data, rows, title="", **kw):
    """The next front of a model after its column stage: run on the current pick (frozen_spec), reusing its result,
    and apply best_of. Skipped when the column stage passed nothing: on one column the later fronts are inert
    (Descisions.md 2026-11-17 D-1). stages: the model's list of stage dicts, appended to.
    Returns: the stage table, or a one-row frame saying why it was skipped."""
    if stages[0]["pick"] == "base":
        msg = "columns + capacity passed nothing: the model stays on one column, where this front is inert"
        print(f" {stages[0]['name']} {title}: skipped ({msg})")
        return pd.DataFrame({"skipped": [msg]})
    st = stage(stages[0]["name"], data, frozen_spec(stages), rows, title=title, after=stages[-1], **kw)
    stages.append(st)
    best_of(st)
    return st["table"]


def rows_tried(stages):
    """How many configs a model's round three ran (each stage's rows; a reused base is not counted twice)."""
    return sum(len(s["specs"]) - (i > 0) for i, s in enumerate(stages))


def chain_report(stages, data, horizon="24h"):
    """A model's round three end to end: the first stage's base (the official config) against the last pick, on the
    four periods, with the paired val CI (the same moving blocks of val resampled for both). No test.
    stages: the stage dicts in the order they ran (best_of applied to each). Returns: (table, ci dict)."""
    first, last = stages[0], stages[-1]
    official, frozen = first["results"]["base"], last["results"][last["pick"] or "base"]
    table = four_periods({"official": official, "round three": frozen}, data, horizon=horizon)
    ci = paired_ci(frozen, official, splits=("val",))
    print(f" {first['name']}: picks {[s['pick'] for s in stages]}; val {ci['auc_b']:.3f} -> {ci['auc_a']:.3f}, "
          f"paired d {ci['diff']:+.3f} [{ci['lo']:+.3f}, {ci['hi']:+.3f}]")
    return table, ci


def frozen_spec(stages):
    """The spec a model ends round three with: the last stage's pick (its base when nothing passed)."""
    last = stages[-1]
    return last["specs"][last["pick"] or "base"]


ROUND_THREE_FILE = os.path.join("model_results", "round3_specs.json")


def save_round_three(stages_by_model, folder=os.path.join("model_results", "round3"), data=None, horizon="24h"):
    """Write round three to disk, from the stage dicts still in the kernel (nothing is re-run):
    <folder>/<model>__stage<k>.csv  every stage table (fold_k, cv, val, d_cv, d_val, wins, passes, ...), with a `picked` column
    <folder>/summary.csv           per model: picks, configs tried, CV / val official vs pick, paired val CI (when `data` is given)
    <folder>/round3_specs.json     the spec each model ends with (also read by lab.load_round_two(path))
    stages_by_model: {model: [stage dict, ...]} — a model whose last stage has no pick (a stage that stopped) keeps its last
    completed pick. No test column is written. Returns: the folder."""
    os.makedirs(folder, exist_ok=True)
    specs, rows = {}, []
    for model, stages in stages_by_model.items():
        done = [s for s in stages if s.get("pick") is not None]
        for k, st in enumerate(done, 1):
            t = st["table"].copy()
            t["picked"] = t["setting"] == st["pick"]
            t.to_csv(os.path.join(folder, f"{model}__stage{k}.csv"), index=False)
        if not done:
            continue
        specs[model] = frozen_spec(done)
        row = {"model": model, "picks": " | ".join(s["pick"] for s in done), "configs tried": rows_tried(done),
               "columns": len(specs[model][horizon]["cols"])}
        if data is not None:
            with contextlib.redirect_stdout(io.StringIO()):
                t, ci = chain_report(done, data, horizon)
            row.update({"cv official": t.loc[0, "cv"], "cv pick": t.loc[1, "cv"], "val official": t.loc[0, "val"],
                        "val pick": t.loc[1, "val"], "paired d val": ci["diff"], "ci lo": ci["lo"], "ci hi": ci["hi"]})
        rows.append(row)
    pd.DataFrame(rows).round(4).to_csv(os.path.join(folder, "summary.csv"), index=False)
    save_round_two(specs, path=os.path.join(folder, "round3_specs.json"))
    save_round_two(specs, path=ROUND_THREE_FILE)
    print(f" round three saved -> {folder}/ ({len(specs)} model(s), {sum(len(s) for s in stages_by_model.values())} stage table(s))")
    return folder
