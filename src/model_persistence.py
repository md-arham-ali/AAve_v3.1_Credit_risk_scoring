"""Modeling stage — result persistence, run manifest, and the training-data bundle loader."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json

import numpy as np
import pandas as pd
from pathlib import Path

from model_config import (TIME_COL, STRESS_TARGETS, REG_TARGET, TARGET_BASE_COL,
                          MODEL_RESULTS_DIR, EMBARGO)
from model_split import apply_feature_transform, apply_scaler


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


def save_model_result(result, out_dir=MODEL_RESULTS_DIR):
    """Persist one model's result: metrics JSON + full-timeline predictions CSV.
    result dict + out_dir -> two files keyed name__kind.
    Returns: the written paths. Keyed so classifier/regressor pairs never collide."""
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{result['name']}__{result.get('kind', 'model')}"
    preds = result.get("predictions")
    meta = {k: v for k, v in result.items()
            if k != "predictions" and not isinstance(v, pd.DataFrame)}
    with open(folder / f"{stem}.json", "w") as fh:
        json.dump(_json_safe(meta), fh, indent=1)
    if isinstance(preds, pd.DataFrame):
        preds.to_csv(folder / f"predictions_{stem}.csv", index=False)
    print(f" saved {stem} → {folder}/")


def write_run_manifest(data, out_dir=MODEL_RESULTS_DIR, extra=None):
    """Write run provenance beside the results: timestamp, lib versions, split shapes.
    out_dir + run metadata -> manifest JSON.
    Returns: the manifest path."""

    # model_results/ is only trustworthy when this manifest matches the run that made it.
    import datetime
    import sklearn
    import xgboost
    import lightgbm
    manifest = {
        "run_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "versions": {"sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
                     "lightgbm": lightgbm.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "split_rows": {s: len(data["splits"][s]) for s in ("train", "val", "test")},
        "n_features": len(data["feature_cols"]),
        "transform": {"n_log1p": len(data["transform"]["log1p_cols"]),
                      "dropped": data["transform"]["dropped"]},
        "thresholds": data["thresholds"],
        "embargo_days": EMBARGO,
    }
    if extra:
        manifest.update(extra)
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "run_manifest.json", "w") as fh:
        json.dump(_json_safe(manifest), fh, indent=1)
    print(f" wrote run_manifest.json → {folder}/")
    return manifest


def load_model_results(out_dir=MODEL_RESULTS_DIR):
    """Read every saved result back: {name__kind: result_dict}, predictions attached."""
    folder = Path(out_dir)
    out = {}
    for path in sorted(folder.glob("*.json")):
        res = json.load(open(path))
        if "name" not in res:                     # run_manifest.json etc.
            continue
        pred_path = folder / f"predictions_{path.stem}.csv"
        if pred_path.exists():
            res["predictions"] = pd.read_csv(pred_path)
        out[path.stem] = res
    return out


def load_model_data(data_dir="transformed_data"):
    """Load the split CSVs + meta into the bundle every runner consumes.
    Applies the persisted train-fit transform (log1p heavy tails, drop near-constants), then scales.
    Returns: {splits, frames_t, feature_cols, thresholds, scaler_params, transform, X: {raw, scaled}, y, times, stress}."""
    folder = Path(data_dir)
    meta = json.load(open(folder / "model_split_meta.json"))
    transform = meta["feature_transform"]
    cols = [c for c in meta["feature_cols"] if c not in set(transform["dropped"])]
    params = meta["scaler_params"]
    thresholds = {int(k): float(v) for k, v in meta["thresholds"].items()}
    splits = {s: pd.read_csv(folder / f"DF_model_{s}.csv") for s in ("train", "val", "test")}
    frames_t = {s: apply_feature_transform(part, transform) for s, part in splits.items()}

    data = {"splits": splits, "frames_t": frames_t, "feature_cols": cols,
            "thresholds": thresholds, "scaler_params": params, "transform": transform,
            "X": {"raw": {}, "scaled": {}}, "y": {}, "times": {}, "stress": {}}
    for s, part in frames_t.items():
        data["X"]["raw"][s] = apply_scaler(part, params, cols, standardize=False).to_numpy()
        data["X"]["scaled"][s] = apply_scaler(part, params, cols, standardize=True).to_numpy()
        data["times"][s] = part[TIME_COL].tolist()
        data["stress"][s] = (splits[s][TARGET_BASE_COL].astype(float)
                             > thresholds[1]).to_numpy().astype(int)
    for t in list(STRESS_TARGETS):
        data["y"][t] = {s: splits[s][t].to_numpy(dtype=int) for s in splits}
    data["y"][REG_TARGET] = {s: splits[s][REG_TARGET].to_numpy(dtype=float) for s in splits}
    print(f" loaded splits {[len(splits[s]) for s in ('train', 'val', 'test')]} rows, "
          f"{len(cols)} features ({len(transform['log1p_cols'])} log1p, "
          f"{len(transform['dropped'])} dropped)")
    return data
