#!/bin/bash
# Start the Dagster UI + orchestrator LOCALLY.
#
#   ./scripts/dagster_dev.sh
#
# Why this script exists at all: `dagster dev` with DAGSTER_HOME unset silently
# creates a throwaway .tmp_dagster_home_<random>/ in the working directory and
# DELETES it on exit. Runs still get written to SQLite in there — they are just
# thrown away when you Ctrl+C, so the materialization timeline (the main reason
# assets beat plain tasks) never accumulates. Exporting the variable by hand
# works but has to be redone in every new terminal, and exporting it AFTER the
# server is already running does nothing — a process reads its environment once,
# at startup. This script removes both failure modes.
#
# Runs in the FOREGROUND. This terminal becomes the server; leave the tab open.
# Ctrl+C stops it.
#
# Storage is SQLite and needs no configuration — it is Dagster's default. The
# only thing that was ever missing is a DAGSTER_HOME that outlives the process.

set -euo pipefail

PORT=3000
DEFS="orchestration/definitions.py"

# Resolve the repo root from this script's own location, so the script works from
# any working directory (same trick as scripts/mlflow_local.sh).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Absolute path required — Dagster rejects a relative DAGSTER_HOME outright.
export DAGSTER_HOME="$ROOT/.dagster_home"
PY="$ROOT/.venv/bin/python"

# --- preflight -------------------------------------------------------------- #

if [ ! -x "$PY" ]; then
  echo "ERROR: $PY not found."
  echo "       Create the venv:  python3.12 -m venv .venv"
  echo "       Then install:     .venv/bin/python -m pip install -r requirements.txt"
  exit 1
fi

if ! "$PY" -c "import dagster" 2>/dev/null; then
  echo "ERROR: dagster is not installed in .venv"
  echo "       Fix:  $PY -m pip install dagster==1.13.17 dagster-webserver==1.13.17"
  echo "       (use 'python -m pip', not '.venv/bin/pip' — that wrapper's shebang"
  echo "        still points at the old pre-rename venv path)"
  exit 1
fi

if [ ! -f "$ROOT/$DEFS" ]; then
  echo "ERROR: $DEFS not found under $ROOT"
  exit 1
fi

if lsof -ti "tcp:$PORT" >/dev/null 2>&1; then
  echo "ERROR: port $PORT is already in use by PID(s): $(lsof -ti tcp:$PORT | tr '\n' ' ')"
  echo "       Another dagster dev is probably already up — check http://localhost:$PORT"
  echo "       To take it over:  kill \$(lsof -ti tcp:$PORT)"
  exit 1
fi

mkdir -p "$DAGSTER_HOME"

# A leftover temp home means a previous session ran without DAGSTER_HOME set.
# Only mention it — it holds runs that are already orphaned, and deleting other
# people's directories is not this script's job.
for tmp in "$ROOT"/.tmp_dagster_home_*; do
  [ -d "$tmp" ] || continue
  echo "· note: leftover temp instance $(basename "$tmp") — from a run with"
  echo "        DAGSTER_HOME unset. Its history is orphaned; safe to delete."
done

# --- report ----------------------------------------------------------------- #

RUNS_DB="$DAGSTER_HOME/history/runs.db"
if [ -f "$RUNS_DB" ]; then
  COUNT=$(sqlite3 "$RUNS_DB" "SELECT COUNT(*) FROM runs;" 2>/dev/null || echo "?")
  echo "· history  : $RUNS_DB  ($COUNT existing runs)"
else
  echo "· history  : $RUNS_DB  (new — created on first materialization)"
fi
echo "· defs     : $DEFS"
echo "· notebooks: executed copies land in $ROOT/runs/"
echo ""
echo "  UI      http://localhost:$PORT"
echo "  stop    Ctrl+C"
echo ""
echo "  Before materializing trained_models, start MLflow in another tab:"
echo "    ./scripts/mlflow_local.sh"
echo ""

# --- run -------------------------------------------------------------------- #
# `$PY -m dagster` rather than the .venv/bin/dagster wrapper: same interpreter
# guarantee the assets rely on (they shell out via sys.executable), and immune to
# the wrapper-shebang breakage this repo has already hit once.
exec "$PY" -m dagster dev -f "$DEFS" --port "$PORT"
