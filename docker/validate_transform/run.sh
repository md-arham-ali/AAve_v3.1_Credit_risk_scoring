#!/bin/sh
# Validation + transformation stage.
#
# ORDER MATTERS. validation.ipynb checks raw tables AND the transformed frames
# (transformed_data/DF_common_final.csv, DF_common_1.csv) in one notebook, and
# those frames are produced by transform.ipynb. Running validation first would
# check last run's frames — or crash on a clean checkout. So:
#
#   1. transform          query_result_data/  -> transformed_data/
#   2. feature_addition    transformed_data/  -> transformed_data/ (+ ~70 features)
#   3. validation          checks raw tables AND the frames just built
set -e

RUNS_DIR=/app/runs
mkdir -p "$RUNS_DIR"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

echo "=== 1/3  transform.ipynb — scale, value, aggregate, assemble panels ==="
papermill /app/notebooks/transform.ipynb \
          "$RUNS_DIR/transform_${STAMP}.ipynb" \
          -k python3 --cwd /app

echo "=== 2/3  feature_addition.ipynb — derived ratio / growth / risk features ==="
papermill /app/notebooks/feature_addition.ipynb \
          "$RUNS_DIR/feature_addition_${STAMP}.ipynb" \
          -k python3 --cwd /app

echo "=== 3/3  validation.ipynb — GE suites on raw tables + tier checks on frames ==="
papermill /app/notebooks/validation.ipynb \
          "$RUNS_DIR/validation_${STAMP}.ipynb" \
          -k python3 --cwd /app

echo "=== validate_transform stage complete — executed notebooks in runs/ ==="
