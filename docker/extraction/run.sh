#!/bin/sh
# Extraction stage: fetch Dune tables, then normalize raw units in place.
#
# -e   stop at the first failing command (so a failed fetch doesn't silently
#      fall through into normalization on incomplete data)
# -k python3
#      force the kernel. normalize.ipynb's saved kernelspec is
#      ".venv (3.14.4.final.0)" — a name that only existed on the original
#      laptop. Without this override papermill aborts with "kernel not found".
set -e

RUNS_DIR=/app/runs
mkdir -p "$RUNS_DIR"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

echo "=== 1/2  process.ipynb — fetch Dune query results ==="
papermill /app/notebooks/process.ipynb \
          "$RUNS_DIR/process_${STAMP}.ipynb" \
          -k python3 --cwd /app

echo "=== 2/2  normalize.ipynb — raw units -> real units (in place, idempotent) ==="
papermill /app/notebooks/normalize.ipynb \
          "$RUNS_DIR/normalize_${STAMP}.ipynb" \
          -k python3 --cwd /app

echo "=== extraction stage complete — executed notebooks in runs/ ==="
