"""Pull a Dune query's latest stored result table and save it as a versioned CSV."""

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
    """Read DUNE_API_KEY from .env into the environment and return it.
    Returns: the key string. Never printed. Call once per session."""
    load_dotenv(dotenv_path)
    api_key = os.getenv(ENV_KEY_NAME)
    if not api_key:
        raise RuntimeError(
            f"{ENV_KEY_NAME} is missing. Add a line `{ENV_KEY_NAME}=your_key` to your .env file."
        )
    return api_key


def fetch_query_table(query_id, api_key=None, save_dir=DEFAULT_SAVE_DIR, page_limit=100000,
                      save=True, table_name=None):
    """Fetch a Dune query's latest stored result (no re-run, no credits spent).
    query_id/api_key/save_dir/page_limit/save/table_name -> paged fetch + CSV write.
    Returns: (DataFrame, csv_path); csv_path is None when save=False."""

    # NOTE: table_name is cosmetic — transform._NAME_RE reads the id out of the tail.
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
        prefix = table_name or DEFAULT_SAVE_DIR
        csv_path = folder / f"{prefix}_{query_id}_{stamp}.csv"
        df.to_csv(csv_path, index=False)

    return df, csv_path


if __name__ == "__main__":
    # Run as a script:  python dune_fetch.py <query_id>
    qid = int(sys.argv[1])
    frame, path = fetch_query_table(qid)
    print(f"Fetched {len(frame)} rows -> {path}")
