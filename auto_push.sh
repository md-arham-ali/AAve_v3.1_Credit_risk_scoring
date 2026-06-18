#!/bin/bash
# Daily auto-commit-and-push for the aave_v3.1 repo.
# Commits any changes and pushes to origin/main. Logs to .autopush.log.

REPO="/Users/ali/Documents/python/aave_v3.1"
LOG="$REPO/.autopush.log"

{
  echo "===== $(date) ====="
  # Make sure git is on PATH whether installed via Homebrew (Apple Silicon /
  # Intel) or system.
  export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

  cd "$REPO" || { echo "Repo not found: $REPO"; exit 1; }

  if [[ -n "$(git status --porcelain)" ]]; then
    git add -A
    git commit -m "Auto-commit $(date +%F)"
    git push origin main && echo "Pushed successfully." || echo "Push FAILED."
  else
    echo "No changes to commit."
  fi
} >> "$LOG" 2>&1
