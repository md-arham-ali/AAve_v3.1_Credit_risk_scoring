#!/bin/sh
# credit-risk-prep: turn the shared panel into a split, leakage-guarded modelling table.
#
# 1. model_features   evidence pass — per-column quality/skew profile, co-movement
#                     clusters, tail-risk tiers, balanced row regimes. Writes no
#                     files; its findings justify the log1p + selection choices
#                     the next two notebooks make.
# 2. model_dataset    2h panel -> daily (ratios rebuilt as ratio-of-sums), joined
#                     with the 24h liq/user frames, restricted to FEATURE_COLS,
#                     forward targets added -> DF_model_dataset_24h.csv
# 3. model_split      chronological 70/15/15 with a 7d embargo, train-only stress
#                     thresholds + feature transform + scaler
#                     -> DF_model_{train,val,test}.csv + model_split_meta.json
set -e

RUNS_DIR=/app/runs
mkdir -p "$RUNS_DIR"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

echo "=== 1/3  model_features.ipynb — feature quality + structure evidence ==="
papermill /app/notebooks/model_features.ipynb \
          "$RUNS_DIR/model_features_${STAMP}.ipynb" \
          -k python3 --cwd /app

echo "=== 2/3  model_dataset.ipynb — daily credit-risk table + forward targets ==="
papermill /app/notebooks/model_dataset.ipynb \
          "$RUNS_DIR/model_dataset_${STAMP}.ipynb" \
          -k python3 --cwd /app

echo "=== 3/3  model_split.ipynb — embargoed split, train-only transform/scaler ==="
papermill /app/notebooks/model_split.ipynb \
          "$RUNS_DIR/model_split_${STAMP}.ipynb" \
          -k python3 --cwd /app

echo "=== credit-risk-prep complete — executed notebooks in runs/ ==="
