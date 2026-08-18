#!/bin/bash
# Start the MLflow tracking server LOCALLY (no Docker), in the FOREGROUND.
#   ./scripts/mlflow_local.sh     — Ctrl+C stops it; tracking only works while it runs.
# Serves the SAME ./mlflow_data/ SQLite db the Docker `mlflow` service bind-mounts,
# so switching is a change of process, not a migration. Refuses to run alongside
# that service — SQLite tolerates exactly one writer.

set -euo pipefail

PORT=5001
EXPERIMENT_HINT="credit_risk"

# Repo root from the script's own location, so it works from any working directory.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DIR="$ROOT/mlflow_data"
DB="$DATA_DIR/mlflow.db"
ARTIFACTS="$DATA_DIR/artifacts"
PY="$ROOT/.venv/bin/python"

# --- preflight -------------------------------------------------------------- #

if [ ! -x "$PY" ]; then
  echo "ERROR: $PY not found."
  echo "       Create the venv:  python3.12 -m venv .venv"
  echo "       Then install:     .venv/bin/python -m pip install -r requirements.txt"
  exit 1
fi

if ! "$PY" -c "import mlflow" 2>/dev/null; then
  echo "ERROR: mlflow is not installed in .venv"
  echo "       Fix:  $PY -m pip install mlflow==2.22.0"
  echo "       (use 'python -m pip', not '.venv/bin/pip' — that wrapper's shebang"
  echo "        still points at the old pre-rename venv path)"
  exit 1
fi

# Stop the Docker service if it is up — the two-writer guard.
if command -v docker >/dev/null 2>&1 \
   && docker compose ps --services --filter status=running 2>/dev/null | grep -qx mlflow; then
  echo "· Docker mlflow service is running — stopping it first"
  echo "  (both write $DB, and SQLite allows only one writer)"
  docker compose stop mlflow
fi

# Anything else already holding the port?
if lsof -ti "tcp:$PORT" >/dev/null 2>&1; then
  echo "ERROR: port $PORT is already in use by PID(s): $(lsof -ti tcp:$PORT | tr '\n' ' ')"
  echo "       Another MLflow server is probably already up — check http://localhost:$PORT"
  echo "       To take it over:  kill \$(lsof -ti tcp:$PORT)"
  exit 1
fi

mkdir -p "$ARTIFACTS"

# --- report ----------------------------------------------------------------- #

if [ -f "$DB" ]; then
  SIZE=$(du -h "$DB" | cut -f1)
  RUNS=$(sqlite3 "$DB" "SELECT COUNT(*) FROM runs;" 2>/dev/null || echo "?")
  echo "· database : $DB  ($SIZE, $RUNS existing runs)"
else
  echo "· database : $DB  (new — will be created)"
fi
echo "· artifacts: $ARTIFACTS"
echo ""
echo "  UI            http://localhost:$PORT   (open the '$EXPERIMENT_HINT' experiment,"
echo "                                          not 'Default' — Default is empty)"
echo "  notebook uses MLFLOW_TRACKING_URI=http://localhost:$PORT"
echo "  stop          Ctrl+C"
echo ""

# --- run -------------------------------------------------------------------- #
# --host 127.0.0.1 : this machine only. The Docker service needs 0.0.0.0 because
#                    there "this machine" means inside the container; here that
#                    would expose the server to your whole network for no reason.
exec "$PY" -m mlflow server \
  --backend-store-uri "sqlite:///$DB" \
  --default-artifact-root "$ARTIFACTS" \
  --host 127.0.0.1 \
  --port "$PORT"
