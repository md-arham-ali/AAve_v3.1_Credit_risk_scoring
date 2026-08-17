"""Backfill preserved model_results_* folders into MLflow as historical runs.

One-off. Each preserved folder is a complete pipeline run whose results would
otherwise only exist as files — this replays them into MLflow so past split
configurations are comparable in the UI alongside future runs.

Every run is tagged `source=backfill` plus the folder it came from, so backfilled
history is never mistaken for a live run.

Run inside the credit-risk-model container (it has both mlflow and src/ on the
path), with the tracking server already up:

    docker compose up -d mlflow
    docker compose run --rm credit-risk-model python /app/scripts/backfill_mlflow.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "/app/src")

import mlflow_tracking as mt
from model_persistence import load_model_results


# folder -> the tags that describe what that run WAS. Recorded explicitly rather
# than parsed from the name, because the names are informal and one of them
# understates what actually happened (see the embargo note below).
BACKFILL_RUNS = {
    "model_results_20260721_70-15-15": {
        "split": "70/15/15",
        "embargo_days": "7",
        "note": "pre-domain-split config, 52 features",
    },
    "model_results_20260729_55-25-25": {
        "split": "55/25/25",
        "embargo_days": "1",
        # The folder's own model_split_meta recorded 7 because the split notebook
        # wrote cfg.EMBARGO instead of the value it actually passed. The real gap
        # was 1 day, so labels near the boundary overlapped val and every val
        # metric in this run is inflated. Tagged so the UI cannot mislead.
        "note": "embargo actually 1 despite meta saying 7 — val metrics inflated",
    },
}


def backfill(folder, tags, repo_root=Path("/app")):
    path = repo_root / folder
    if not path.is_dir():
        print(f" SKIP {folder} — not found")
        return 0

    results = load_model_results(out_dir=str(path))
    if not results:
        print(f" SKIP {folder} — no result JSONs")
        return 0

    manifest_path = path / "run_manifest.json"
    manifest = json.load(open(manifest_path)) if manifest_path.exists() else {}

    # The runs predate split_fractions being written into the meta, so synthesise
    # the params from the manifest + the explicit tags above.
    split_meta = {
        "embargo_days": tags.get("embargo_days"),
        "split_fractions": tags.get("split"),
        "feature_cols": [None] * manifest.get("n_features", 0),
        "thresholds": manifest.get("thresholds", {}),
    }

    ids = mt.log_all_results(
        results,
        split_meta=split_meta,
        extra_tags={
            "source": "backfill",
            "source_folder": folder,
            "run_at_utc": manifest.get("run_at_utc", "unknown"),
            **{k: str(v) for k, v in tags.items()},
        },
    )
    print(f" {folder}: {len(ids)} runs logged"
          f"  ({manifest.get('run_at_utc', '?')[:19]}, {manifest.get('split_rows')})")
    return len(ids)


if __name__ == "__main__":
    total = sum(backfill(folder, tags) for folder, tags in BACKFILL_RUNS.items())
    print(f"\nbackfill complete — {total} historical runs in MLflow")
