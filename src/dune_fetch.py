"""Fetch the latest stored result table of a Dune query and save it as a versioned CSV.

Libraries imported by this module:
    requests        -> the Dune HTTP API call
    pandas          -> result table + CSV writing
    python-dotenv   -> read DUNE_API_KEY from .env
    os, pathlib, datetime, sys -> standard library
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv

DUNE_API_BASE = "https://api.dune.com/api/v1"
ENV_KEY_NAME = "DUNE_API_KEY"
DEFAULT_SAVE_DIR = "query_result_data"


def load_api_key(dotenv_path=None):
    """Read DUNE_API_KEY from the .env file into the environment and return it.

    Call this once per session (the notebook's first cell does) to make the key
    available to every later call. The key itself is never printed.
    """
    load_dotenv(dotenv_path)
    api_key = os.getenv(ENV_KEY_NAME)
    if not api_key:
        raise RuntimeError(
            f"{ENV_KEY_NAME} is missing. Add a line `{ENV_KEY_NAME}=your_key` to your .env file."
        )
    return api_key


def fetch_query_table(query_id, api_key=None, save_dir=DEFAULT_SAVE_DIR, page_limit=100000, save=True):
    """Fetch a Dune query's latest result as a DataFrame and save it to a CSV.

    This reads the query's most recent stored results — it does NOT re-run the
    query, so it spends no execution credits. Large results are paged through fully.

    Args:
        query_id:   the numeric Dune query id (provided manually in the notebook).
        api_key:    optional; falls back to load_api_key() / the environment.
        save_dir:   folder for the CSV (created if missing).
        page_limit: rows fetched per API page.
        save:       set False to skip writing the CSV.

    Returns:
        (dataframe, csv_path)  -- csv_path is None when save=False.
    """
    api_key = api_key or load_api_key()
    headers = {"X-Dune-Api-Key": api_key}
    url = f"{DUNE_API_BASE}/query/{query_id}/results"

    rows = []
    columns = None
    offset = 0
    while True:
        response = requests.get(
            url,
            headers=headers,
            params={"limit": page_limit, "offset": offset},
            timeout=200,
        )
        response.raise_for_status()
        payload = response.json()

        result = payload.get("result")
        if result is None:
            state = payload.get("state", "unknown")
            raise RuntimeError(
                f"No stored results for query {query_id} (state={state}). "
                "Run the query on Dune at least once first."
            )

        rows.extend(result.get("rows", []))
        columns = (result.get("metadata") or {}).get("column_names", columns)

        next_offset = payload.get("next_offset")
        if next_offset is None:        # no more pages
            break
        offset = next_offset

    df = pd.DataFrame(rows)
    if columns:                        # preserve Dune's original column order
        df = df.reindex(columns=columns)

    csv_path = None
    if save:
        folder = Path(save_dir)
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        csv_path = folder / f"query_result_data_{query_id}_{stamp}.csv"
        df.to_csv(csv_path, index=False)

    return df, csv_path


if __name__ == "__main__":
    # Run as a script:  python dune_fetch.py <query_id>
    qid = int(sys.argv[1])
    frame, path = fetch_query_table(qid)
    print(f"Fetched {len(frame)} rows -> {path}")
