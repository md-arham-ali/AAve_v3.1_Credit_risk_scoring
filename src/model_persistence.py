"""Modeling stage — result persistence, run manifest, and the training-data bundle loader."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hashlib
import json

import numpy as np
import pandas as pd
from pathlib import Path

from model_config import (TIME_COL, STRESS_TARGETS, REG_TARGET, TARGET_BASE_COL,
                          MODEL_RESULTS_DIR, MODEL_DATA_DIR, RESULTS_SCHEMA)
import model_split as msp
import model_dataset as mds
import model_scaling as msc


def _json_safe(obj):
    """Recursively convert numpy scalars/arrays for json.dump."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _run_key(meta):
    """What a set of results belongs to: the exact split definition + the results schema.
    Any change to the data, boundaries, feature selection or thresholds changes split_meta.json,
    and so the key. Returns: a short hex string."""
    text = json.dumps(meta, sort_keys=True) + f"|schema={RESULTS_SCHEMA}"
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _manifest_key(folder):
    """The run key run_manifest.json declares for this results folder, or None."""
    path = Path(folder) / "run_manifest.json"
    return json.load(open(path)).get("run_key") if path.exists() else None


def _stem(result):
    """name__kind: the file stem of one result. Keyed so classifier/regressor pairs never collide."""
    return f"{result['name']}__{result.get('kind', 'model')}"


def save_model_result(result, out_dir=MODEL_RESULTS_DIR):
    """Persist one model's result: metrics JSON + full-timeline predictions CSV + tuning table CSV
    (+ the out-of-fold scores CSV when the runner was called with oof=True).
    result dict + out_dir -> files keyed name__kind, stamped with the current manifest's run key.
    The tuning CSV is the source of truth for the search; MLflow only gets a copy.
    A trial run (save=False on the runner) writes nothing, so the result PRI reads stays the official one."""
    stem = _stem(result)
    if result.get("trial"):
        print(f" {stem}: trial run (save=False), nothing saved")
        return
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    preds, tuning, oof = result.get("predictions"), result.get("tuning"), result.get("oof")
    # fitted models are objects, not JSON: they go through save_fitted, and only for chosen models
    meta = {k: v for k, v in result.items()
            if k not in ("predictions", "tuning", "fitted") and not isinstance(v, pd.DataFrame)}
    meta["run_key"] = _manifest_key(folder)
    if meta["run_key"] is None:
        print(f" ⚠️ {stem}: no run key in {folder}/run_manifest.json — call write_run_manifest "
              f"first, or this result will not be read back")
    with open(folder / f"{stem}.json", "w") as fh:
        json.dump(_json_safe(meta), fh, indent=1)
    if isinstance(preds, pd.DataFrame):
        preds.to_csv(folder / f"predictions_{stem}.csv", index=False)
    if isinstance(tuning, pd.DataFrame):
        tuning.to_csv(folder / f"tuning_{stem}.csv", index=False)
    if isinstance(oof, pd.DataFrame):
        oof.to_csv(folder / f"oof_{stem}.csv", index=False)
    print(f" saved {stem} → {folder}/")


def save_fitted(result, data, out_dir=MODEL_RESULTS_DIR):
    """Save the fitted model(s) of a CHOSEN result: the exact train fit behind its reported scores.
    One folder per model, model_results/fitted/<name>__<kind>/:
        <horizon>.joblib   sklearn, xgboost, lightgbm, IsolationForest, Cox (joblib = pickle)
        <horizon>.pt       LSTM / Transformer state_dict, rebuilt with model_training._build_net
                           from params_by_target and n_features
        bundle.json        everything needed to score new rows the same way: feature_cols and
                           scaler_params (for model_split.apply_scaler), x_key, features_used,
                           thresholds_f1, params_by_target, run_key, settings_hash, library versions.
                           Per-horizon features: features_by_target[target] is the column list of
                           files[target]; features_used is then only cv_target's list
    data: the bundle the result was trained on (load_model_data).
    Refuses a trial run (save=False) and a result read back from disk (it holds no fitted model).
    Returns: the folder, or None for a trial run."""
    stem = _stem(result)
    if result.get("trial"):
        print(f" {stem}: trial run (save=False), fitted model not saved")
        return None
    fitted = result.get("fitted")
    if not fitted:
        raise ValueError(f"{stem}: no fitted model in this result. save_fitted needs the result the "
                         f"runner returned in this session, not one read back from disk.")
    import joblib
    folder = Path(out_dir) / "fitted" / stem
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    for target, model in fitted.items():
        label = target.rpartition("_")[2] if target != REG_TARGET else target
        if hasattr(model, "state_dict"):                  # torch network
            import torch
            path = folder / f"{label}.pt"
            torch.save(model.state_dict(), path)
        elif isinstance(model, list) and model and all(hasattr(m, "state_dict") for m in model):
            import torch                                  # n_seeds > 1: one state_dict per seed, in order
            path = folder / f"{label}.pt"
            torch.save([m.state_dict() for m in model], path)
        else:
            path = folder / f"{label}.joblib"
            joblib.dump(model, path)
        files[target] = path.name

    import sklearn
    import xgboost
    import lightgbm
    x_key = ("raw" if result["name"] == "isolation_forest"
             else "scaled" if result.get("scaled_input", True) else "raw")
    bundle = {"name": result["name"], "kind": result.get("kind"), "files": files,
              "run_key": data["run_key"], "settings_hash": result.get("settings_hash"),
              "x_key": x_key, "feature_cols": data["feature_cols"],
              "features_used": result.get("features_used"), "n_features": len(data["feature_cols"]),
              # set when features were chosen per horizon: each <horizon> file reads its own list
              "features_by_target": result.get("features_by_target"),
              "scaler_params": data["scaler_params"], "thresholds_f1": result.get("thresholds_f1"),
              "params_by_target": result.get("params_by_target"), "lookback": result.get("lookback"),
              "versions": {"python": sys.version.split()[0], "sklearn": sklearn.__version__,
                           "xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__}}
    with open(folder / "bundle.json", "w") as fh:
        json.dump(_json_safe(bundle), fh, indent=1)
    print(f" saved fitted {stem} ({len(files)} model file(s)) → {folder}/")
    return folder


def flat_settings(result, max_len=500):
    """The result's used settings as one flat {name: value} dict, the shape MLflow params need.
    More than one horizon: names get the horizon first ("7d.max_depth"). Nested dicts are joined
    with a dot ("optimizer_params.weight_decay"). Values longer than max_len characters are cut.
    Returns: {str: value}; changed names are in result["changed"]."""
    by_target = result.get("params_by_target") or {"": result.get("params") or {}}
    multi = len(by_target) > 1
    out = {}

    def _put(prefix, value):
        if isinstance(value, dict) and value:
            for k, v in value.items():
                _put(f"{prefix}.{k}", v)
            return
        text = value if isinstance(value, (int, float, bool)) or value is None else str(value)
        out[prefix] = text[:max_len] if isinstance(text, str) else text

    for target, settings in by_target.items():
        label = target.rpartition("_")[2] if target != REG_TARGET else target
        for k, v in settings.items():
            _put(f"{label}.{k}" if multi else k, v)
    return out


def write_run_manifest(data, out_dir=MODEL_RESULTS_DIR, extra=None):
    """Write run provenance beside the results: run key, batch, timestamp, lib versions, split
    shapes. Only results carrying this run key are read back (load_model_results).
    out_dir + run metadata -> manifest JSON.
    Returns: the manifest dict."""

    # model_results/ is only trustworthy when this manifest matches the run that made it.
    import datetime
    import sklearn
    import xgboost
    import lightgbm
    default_t = data["stress_targets"][0]
    manifest = {
        "run_key": data["run_key"],
        "results_schema": RESULTS_SCHEMA,
        "batch": data["batch"],
        "run_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "versions": {"sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
                     "lightgbm": lightgbm.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "split_rows": {s: len(data["splits"][default_t][s]) for s in ("train", "val", "test")},
        "n_features": len(data["feature_cols"]),
        "dropped": data["selection"]["dropped"],
        "stress_targets": list(data["stress_targets"]),
        "stress_thresholds": data["thresholds"],
        "embargo": data["embargo"],
        "cv_folds": data["n_folds"],
    }
    if extra:
        manifest.update(extra)
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "run_manifest.json", "w") as fh:
        json.dump(_json_safe(manifest), fh, indent=1)
    print(f" wrote run_manifest.json → {folder}/")
    return manifest


def load_model_results(out_dir=MODEL_RESULTS_DIR, current_only=True):
    """Read saved results back: {name__kind: result_dict}, predictions attached.
    current_only: keep only results whose run key matches run_manifest.json, so a file from an
    older run or dataset can never reach the PRI blend, a leaderboard or the dashboard. What was
    skipped is printed. False reads everything (forensics, backfills)."""
    folder = Path(out_dir)
    key = _manifest_key(folder)
    out, ignored = {}, []
    for path in sorted(folder.glob("*.json")):
        res = json.load(open(path))
        if "name" not in res:                     # run_manifest.json etc.
            continue
        if current_only and (key is None or res.get("run_key") != key):
            ignored.append(path.stem)
            continue
        pred_path = folder / f"predictions_{path.stem}.csv"
        if pred_path.exists():
            res["predictions"] = pd.read_csv(pred_path)
        out[path.stem] = res
    if ignored:
        print(f" ignored {len(ignored)} result file(s) not from the current run: "
              f"{', '.join(ignored[:6])}{' …' if len(ignored) > 6 else ''}")
    return out


def _target_bundle(features, target_frame, target_col, assign_lab, scaler_params, cols,
                   time_col=TIME_COL):
    """Join features to ONE target column, split its rows by assign_lab, build X/y/times.
    target_frame: holds target_col (DF_target itself, or msp.binarize_stress's DF_stress).
    assign_lab: msp.assign_split's labels, in target_frame's row order.
    Returns: {splits, frames_t, X: {raw, scaled}, y, times}, each {train, val, test}, each in
    TIME order within its split."""
    joined = mds.join_for_training(features, target_frame, target_col, time_col=time_col)
    lab_by_time = pd.Series(assign_lab.to_numpy(), index=msp.time_of(target_frame, time_col))
    row_lab = lab_by_time.reindex(msp.time_of(joined, time_col)).to_numpy()

    out = {"splits": {}, "frames_t": {}, "X": {"raw": {}, "scaled": {}}, "y": {}, "times": {}}
    for s in ("train", "val", "test"):
        part = joined[row_lab == s].reset_index(drop=True)
        out["splits"][s] = part
        out["frames_t"][s] = part[[time_col, *cols]]
        out["X"]["raw"][s] = msp.apply_scaler(part, scaler_params, cols, standardize=False).to_numpy()
        out["X"]["scaled"][s] = msp.apply_scaler(part, scaler_params, cols, standardize=True).to_numpy()
        out["times"][s] = part[time_col].tolist()
        out["y"][s] = part[target_col].to_numpy()
    return out


def load_split_meta(batch=None):
    """The split definition JSON (model_split.py) for `batch` (None = its own recorded span).
    Standalone from load_model_data so a notebook can read embargo/thresholds/etc. for MLflow
    params without paying for the full reload + rejoin.
    Returns: the parsed split_meta.json dict."""
    import transform as tf
    run = tf.use_output_batch(batch)
    folder = Path(run.folder(MODEL_DATA_DIR)) / "split"
    return json.load(open(folder / "split_meta.json"))


def _label_history(frame, column, horizon):
    """Every labelled row of one target, in time order, whatever its split: what a persistence
    baseline may read. Returns: {labels: Series indexed by UTC stamp, horizon: Timedelta}."""
    y = pd.Series(frame[column].to_numpy(dtype=float), index=pd.DatetimeIndex(msp.time_of(frame)))
    return {"labels": y.dropna().sort_index(), "horizon": pd.Timedelta(horizon)}


def _target_scaling_config(meta):
    """target_scaling_config.json saved beside the DF_target this split read.
    meta["inputs"]["DF_target"]: a CSV path, or [folder, table] resolved in the split's batch.
    Raises when it is missing: without it a score cannot be read back as USD."""
    import transform as tf
    src = meta["inputs"]["DF_target"]
    folder = (Path(src).parent if isinstance(src, str)
              else Path(src[0]) / tf.batch_dir_name(tuple(meta["inputs"]["batch"])))
    path = folder / "target_scaling_config.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — re-run the targets section of model_dataset")
    return json.load(open(path))


def load_model_data(batch=None, meta=None):
    """Reload DF_model / DF_target per the fitted split (model_split.py), rejoin one target
    at a time, and scale — the new per-horizon split keeps features and targets apart, so
    nothing pre-joined is read off disk.
    Each stress horizon this split actually has (data["stress_targets"] — the split's own
    HORIZONS, which can be a subset of model_config.STRESS_TARGETS; this run has no "15d",
    say, if it was fit with HOLDOUT="7d") gets its OWN train/val/test rows (its own embargo)
    — only run_classifier spans every horizon in `targets`; every other runner (baselines,
    isolation forest, Cox, LSTM, transformer, PRI, regressors) uses stress_targets[0] /
    REG_TARGET, both the shortest horizon, so they share one split.
    batch: the split's batch span (None = whatever meta recorded when the split was fit).
    meta: a split definition already in memory, instead of the one on disk (model_lab.refit_inputs:
        the same split on another column pool, for a comparison). Default None = read split_meta.json.
    DF_target has no same-day/raw column for any target base — only its forward-looking
    columns — so a "today's value" baseline isn't available; persistence reads the latest label
    already known at t instead, one full horizon back (model_training._persistence).
    Returns: {feature_cols, scaler_params, selection, thresholds, stress_targets, to_usd, label_history,
    run_key, batch, n_folds, embargo, splits, frames_t, X, y, times} — to_usd: {REG_TARGET: score array -> USD array}; splits/frames_t/X/y/times keyed first by target name (every
    stress_targets entry + REG_TARGET), each holding its own {train, val, test}."""
    meta = load_split_meta(batch) if meta is None else meta
    # a LIST of the two boundary months, not a tuple: resolve_batch treats a tuple as one
    # literal span (fails unless that exact folder exists); a list month-matches each
    # boundary independently, which works whether the source dir has one merged batch
    # folder or the two original sub-batches (model_split's CombinedRun span round-trips
    # through this the same way).
    meta_batch = list(meta["inputs"]["batch"])

    DF_model = msp.load_input(meta["inputs"]["DF_model"], batch=meta_batch)
    DF_target = msp.load_input(meta["inputs"]["DF_target"], batch=meta_batch)
    features = msp.prepare_features(DF_model, meta)
    cols = meta["selection"]["kept"]
    params = meta["scaler_params"]

    horizons = list(meta["embargo"])
    # save_split_meta's _jsonable stringifies every pd.Timestamp — back to Timestamps,
    # or assign_split's own arithmetic against them (t < bounds["start"] + Timedelta(...)) breaks
    bounds = {k: pd.Timestamp(v) for k, v in meta["boundaries"].items()}
    assign = {h: msp.assign_split(DF_target, bounds, h, warmup=meta["warmup"]) for h in horizons}
    thresholds = meta["stress"]["thresholds"]
    DF_stress = msp.binarize_stress(DF_target, thresholds)

    data = {"run_key": _run_key(meta), "batch": meta_batch, "n_folds": meta["folds"]["n"],
            "embargo": meta["embargo"],
            "feature_cols": cols, "scaler_params": params, "selection": meta["selection"],
            "thresholds": thresholds, "stress_targets": [], "label_history": {},
            "splits": {}, "frames_t": {},
            "times": {}, "X": {}, "y": {}}
    for fwd_col, th in thresholds.items():
        t = msp.stress_name(fwd_col)
        if t not in STRESS_TARGETS:
            continue
        data["stress_targets"].append(t)
        h = fwd_col.rpartition("_fwd_")[2]
        bundle = _target_bundle(features, DF_stress, t, assign[h], params, cols)
        data["splits"][t], data["frames_t"][t], data["times"][t] = (
            bundle["splits"], bundle["frames_t"], bundle["times"])
        data["X"][t] = bundle["X"]
        data["y"][t] = {s: bundle["y"][s].astype(int) for s in bundle["y"]}
        data["label_history"][t] = _label_history(DF_stress, t, h)

    reg_h = horizons[0]        # shortest horizon — "next period", same spirit as the old daily REG_TARGET
    reg_col = f"{TARGET_BASE_COL}_fwd_{reg_h}"
    bundle = _target_bundle(features, DF_target, reg_col, assign[reg_h], params, cols)
    data["splits"][REG_TARGET], data["frames_t"][REG_TARGET], data["times"][REG_TARGET] = (
        bundle["splits"], bundle["frames_t"], bundle["times"])
    data["X"][REG_TARGET] = bundle["X"]
    # the score as it is: DF_target is already scaled 0-100, a log on top would scale it twice
    data["y"][REG_TARGET] = {s: bundle["y"][s].astype(float) for s in bundle["y"]}
    data["label_history"][REG_TARGET] = _label_history(DF_target, reg_col, reg_h)
    data["to_usd"] = {REG_TARGET: msc.target_unscaler(DF_target, reg_col, _target_scaling_config(meta))}

    default_t = data["stress_targets"][0] if data["stress_targets"] else STRESS_TARGETS[0]
    rows = {s: len(data["splits"][default_t][s]) for s in ("train", "val", "test")}
    print(f" loaded {len(data['stress_targets'])} stress horizon(s) + regressor ({reg_h}), "
          f"{len(cols)} features | default ({default_t}) split rows {rows}")
    return data
