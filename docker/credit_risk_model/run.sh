#!/bin/sh
# credit-risk-model: train the zoo, blend the PRI, then read the results back.
#
# 1. model_training   two baselines + 11 classifiers + IsolationForest + Cox PH +
#                     LSTM + Temporal Transformer + 9 regressors, each tuned on
#                     embargoed walk-forward CV over TRAIN only, then blended into
#                     the PRI. Writes ~60 files to model_results/ plus
#                     pri_timeseries.csv and run_manifest.json.
# 2. model_results     read-side only, fits nothing. Staged leaderboards, then ONE
#                     cross-family comparison at the end with bootstrap CIs.
#                     Overwrites nothing — it reads whatever step 1 just wrote, so
#                     it always reflects the current run, never an older one.
#
# set -e matters here: model_results loads model_results/*.json, so if training
# fails partway it must not run and present a half-populated leaderboard as if it
# were complete.
set -e

RUNS_DIR=/app/runs
mkdir -p "$RUNS_DIR"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

echo "=== 1/2  model_training.ipynb — 20+ models, tuned, + PRI meta-blend ==="
echo "         (slowest stage: per-model grid search over 5 embargoed CV folds,"
echo "          plus up to 200 epochs each for the LSTM and Transformer)"
papermill /app/notebooks/model_training.ipynb \
          "$RUNS_DIR/model_training_${STAMP}.ipynb" \
          -k python3 --cwd /app

echo "=== 2/2  model_results.ipynb — staged leaderboards + final comparison ==="
papermill /app/notebooks/model_results.ipynb \
          "$RUNS_DIR/model_results_${STAMP}.ipynb" \
          -k python3 --cwd /app

echo "=== credit-risk-model complete — executed notebooks in runs/,"
echo "    metrics + predictions in model_results/ ==="
