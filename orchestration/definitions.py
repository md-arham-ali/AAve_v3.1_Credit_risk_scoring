"""Dagster orchestration — one asset per notebook, run on the HOST via papermill.

Lives outside src/ on purpose: every Dockerfile does `COPY src/ /app/src/`, so
anything in src/ is baked into all five images. This file drives the pipeline
from the host and has no business inside the containers it may later drive.

Run it:

    .venv/bin/dagster dev -f orchestration/definitions.py

then open http://localhost:3000 and click Materialize on any asset.

Each asset shells out to papermill exactly the way docker/*/run.sh already does,
just with host paths and the aave-main kernel. Executed copies land in runs/ with
a UTC stamp, so a failed run leaves an inspectable notebook behind.

NOTE: extraction (process.ipynb + normalize.ipynb) is deliberately NOT here yet —
the Dune query ids 402/404 and need fixing first. The chain therefore starts from
query_result_data/ as it already exists on disk.
"""

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dagster import Definitions, asset

# orchestration/definitions.py -> repo root is one level up
ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
RUNS = ROOT / "runs"

# aave-main, not python3: the python3 kernelspec's argv[0] is the bare string
# "python", so which interpreter it picks depends on PATH at launch. aave-main
# points at an absolute path into this repo's .venv.
KERNEL = "aave-main"


def run_notebook(context, name, params=None):
    """Execute one notebook with papermill. Non-zero exit fails the asset.

    sys.executable -m papermill (not the bare `papermill` command) so the
    notebook always runs under the same interpreter Dagster itself is running
    under — no PATH ambiguity, and no dependence on the .venv/bin wrapper
    scripts, whose shebangs have broken once before after a directory rename.

    params — optional {name: value} passed through as papermill `-p` overrides.
    Unused today; wired up so the commented config_schema example at the bottom
    of this file is real rather than aspirational. Requires the target notebook
    to have a cell tagged `parameters` — without that tag papermill accepts the
    values and silently ignores them.
    """
    RUNS.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    source = NOTEBOOKS / f"{name}.ipynb"
    output = RUNS / f"{name}_{stamp}.ipynb"

    if not source.exists():
        raise FileNotFoundError(f"notebook not found: {source}")

    cmd = [sys.executable, "-m", "papermill",
           str(source), str(output),
           "-k", KERNEL,
           "--cwd", str(ROOT)]
    for key, value in (params or {}).items():
        cmd += ["-p", str(key), str(value)]

    context.log.info(f"papermill {source.name} -> runs/{output.name}"
                     + (f"  params={params}" if params else ""))
    subprocess.run(
        cmd,
        cwd=ROOT,
        check=True,          # notebook error -> asset turns red, downstream is skipped
    )
    context.log.info(f"{name} complete")
    return output


# --------------------------------------------------------------------------- #
# STAGE — validate + transform
# --------------------------------------------------------------------------- #
# Order matters and is not arbitrary: validation.ipynb checks the frames that
# transform.ipynb and feature_addition.ipynb build, so it has to run last or it
# would be checking the PREVIOUS run's output. That ordering was already encoded
# in docker/validate_transform/run.sh; here it is encoded as asset deps instead.

@asset
def transformed_panel(context):
    """query_result_data/ -> transformed_data/ (scaled, valued, assembled panels)."""
    run_notebook(context, "transform")


@asset(deps=[transformed_panel])
def feature_panel(context):
    """+ ~70 derived ratio / growth / risk features onto the panel."""
    run_notebook(context, "feature_addition")


@asset(deps=[feature_panel])
def validated_panel(context):
    """GE suites on the raw tables + tier checks on the frames just built."""
    run_notebook(context, "validation")

# --------------------------------------------------------------------------- #
# STAGE — credit-risk prep
# --------------------------------------------------------------------------- #

@asset(deps=[validated_panel])
def feature_evidence(context):
    """Evidence pass over the panel — quality/skew profile, co-movement clusters,
    tail-risk tiers. Writes no files; its findings justify the log1p + selection
    choices the next two notebooks make. In the graph purely for ordering."""
    run_notebook(context, "model_features")


@asset(deps=[feature_evidence])
def model_dataset(context):
    """2h panel -> daily (ratios rebuilt as ratio-of-sums), joined with the 24h
    liq/user frames, restricted to FEATURE_COLS, forward targets added
    -> transformed_data/DF_model_dataset_24h.csv"""
    run_notebook(context, "model_dataset")


@asset(deps=[model_dataset])
def credit_risk_split(context):
    """Chronological 55/25/25 with a 7d embargo, train-only stress thresholds +
    feature transform + scaler
    -> DF_model_{train,val,test}.csv + model_split_meta.json"""
    run_notebook(context, "model_split")


# --------------------------------------------------------------------------- #
# STAGE — credit-risk model
# --------------------------------------------------------------------------- #

@asset(deps=[credit_risk_split])
def trained_models(context):
    """The full zoo — 2 baselines + 11 classifiers + IsolationForest + Cox PH +
    LSTM + Transformer + 9 regressors, each tuned on embargoed walk-forward CV
    over TRAIN only, then blended into the PRI. Slowest asset by far.
    -> ~60 files in model_results/ + pri_timeseries.csv + run_manifest.json
    Also logs one MLflow run per model (needs the server up on :5001)."""
    run_notebook(context, "model_training")


@asset(deps=[trained_models])
def model_report(context):
    """Read-side only, fits nothing. Staged leaderboards, then ONE cross-family
    comparison with bootstrap CIs. Always reflects the run that just finished."""
    run_notebook(context, "model_results")


# --------------------------------------------------------------------------- #
# REFERENCE — the other @asset options, for later
# --------------------------------------------------------------------------- #
# Everything above uses only `deps`. The four options below are the ones worth
# knowing about; none are needed yet, so they are left commented rather than
# guessed at. Shown together on one asset so the shapes are clear:
#
# from dagster import MaterializeResult          # needed for per-run metadata
#
# @asset(
#     deps=[model_dataset],
#
#     # 1. description — free text shown in the UI's asset detail pane. Without
#     #    it the UI falls back to the function docstring, which is why the
#     #    assets above carry real docstrings instead.
#     description="Chronological split with a 7-day embargo.",
#
#     # 2. group_name — clusters assets visually in the graph view. With 9 assets
#     #    in one chain it buys little; it earns its keep once a second domain
#     #    (liquidity_risk) adds a parallel branch.
#     group_name="credit_risk_prep",
#
#     # 3. metadata — STATIC key/values attached to the asset definition itself.
#     #    Good for pointers that never change.
#     metadata={"notebook": "model_split.ipynb", "owner": "credit_risk"},
#
#     # 4. config_schema — typed, run-time config. Dagster renders a form in the
#     #    UI's "Materialize with config" dialog and REJECTS a wrong type before
#     #    anything runs. This is the clean way to vary a run without editing the
#     #    notebook — e.g. re-running the split at a different embargo.
#     config_schema={"embargo": int},
# )
# def credit_risk_split(context):
#     embargo = context.op_config["embargo"]        # how you read config back
#
#     # papermill parameter injection: the notebook needs ONE cell tagged
#     # `parameters` (Jupyter: View > Cell Toolbar > Tags). Papermill inserts an
#     # override cell directly beneath it. Without that tag the values are
#     # accepted silently and never applied — the failure mode is a run that
#     # looks fine and ignored your parameter.
#     run_notebook(context, "model_split", params={"EMBARGO": embargo})
#
#     # 3b. per-RUN metadata — differs every materialization, so it is returned,
#     #     not declared. Shows up on the timeline for that run.
#     return MaterializeResult(metadata={"embargo_days": embargo})


# --------------------------------------------------------------------------- #
# Every asset must be listed here or Dagster will not see it.
# --------------------------------------------------------------------------- #
defs = Definitions(
    assets=[
        # validate + transform
        transformed_panel,
        feature_panel,
        validated_panel,
        # credit-risk prep
        feature_evidence,
        model_dataset,
        credit_risk_split,
        # credit-risk model
        trained_models,
        model_report,
    ]
)
