"""Fetch Dune query results as versioned CSVs, by either of two paths.

fetch_query_table() reads the stored result — free, but fixed to the window the query was
last run with. run_query_table() re-runs it for a window you choose: costs credits, and
needs {{start_date}}/{{end_date}} — plus {{bucket_hours}} when you set the grain — already
in the SQL on Dune. A parameter the query does not declare is NOT rejected: Dune ignores
it, runs the width hardcoded in that SQL, and still bills, so an unparameterised query
looks like it accepted the value. Keys resolve per account group. Long-form notes on all
of this: explanations_notebooks.md.
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv

DUNE_API_BASE = "https://api.dune.com/api/v1"
ENV_KEY_NAME = "DUNE_API_KEY"
DEFAULT_SAVE_DIR = "query_result_data"
DEFAULT_MANIFEST_DIR = "query_result_data/_runs"

# None means "send no performance field at all", which is the portable default: Dune then
# applies the account's own default tier (documented as medium). Naming a tier explicitly
# is an entitlement the account may not have — DUNE_API_KEY_1 answers an explicit
# "medium" with 400 Invalid performance tier while DUNE_API_KEY_2 accepts the same string,
# so a hardcoded tier makes the client work on one account and not another.
# Set "small"/"medium"/"large" per call only when you actually want to pin one.
DEFAULT_PERFORMANCE = None
VALID_PERFORMANCE_TIERS = ("small", "medium", "large")

# Bucket widths the panel may be built on — the divisors of 24, and nothing else.
# The SQL bucket formula, floor(hour(t) / N) * N, restarts at every midnight, while query
# 04's grid CTE strides continuously from start_date. Those two agree only when N divides
# 24. At N=5 the formula's last bucket of each day is short and the grid drifts to 01:00,
# 06:00, ... — reserve_config then keys on timestamps no other table produces and joins to
# nothing, with no error raised anywhere. Rejecting the width here is the only guard.
VALID_BUCKET_HOURS = (1, 2, 3, 4, 6, 8, 12, 24)

# Export is billed per MB, so one careless 25 MB pull costs a fifth of a free month's
# quota (the 402s on 2026-07-25). Raise deliberately, per call, once you know the size.
DEFAULT_MAX_RESULT_MB = 25.0

# Past 2**53 a float can no longer represent every integer. Used only to decide what to
# REPORT — nothing here rounds, rejects or repairs a value on the strength of it.
FLOAT_EXACT_LIMIT = 2 ** 53

# Given _json_exact() parses with parse_float=Decimal, the parsed type proves where the
# value came from: int means a whole-number literal, exact at any width; Decimal or float
# means it carried a '.' or exponent, which only a DOUBLE in the query produces. So
# Decimal belongs with float, not int — calling it exact would claim a confidence the
# data does not have.
_EXACT_TYPES = (int,)
_INEXACT_TYPES = (float, Decimal)

# Which .env variable holds each account's key, most-specific name first. Group 1 falls
# back to the legacy unnumbered DUNE_API_KEY so older setups keep working.
DEFAULT_KEY_ENV_NAMES = {
    1: ("DUNE_API_KEY_1", ENV_KEY_NAME),
    2: ("DUNE_API_KEY_2",),
    3: ("DUNE_API_KEY_3",),
}

# Terminal execution states. QUERY_STATE_PENDING / _EXECUTING mean "keep polling".
_DONE_STATES = {
    "QUERY_STATE_COMPLETED",
    "QUERY_STATE_FAILED",
    "QUERY_STATE_CANCELLED",
    "QUERY_STATE_EXPIRED",
}


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


def _headers(api_key=None):
    """Auth header dict, resolving the key from .env when not passed explicitly."""
    return {"X-Dune-Api-Key": api_key or load_api_key()}


def _raise_for_status(response, what):
    """raise_for_status(), but keep the explanation Dune actually sent.
    response/what -> nothing on success; requests.HTTPError carrying the body otherwise.

    requests puts only the status line in the exception and drops the response body — and
    the body is precisely where Dune says WHY. Every opaque failure this project has hit
    (401 rejected key, 402 quota, 400 on execute) was diagnosable from that body and was
    not diagnosable from the status code alone.

    Hints are keyed off what Dune actually said, never off the status code alone. An
    earlier version asserted that a 400 on /execute meant the query was unparameterised;
    that is wrong twice over — Dune does NOT error on parameters a query lacks (it runs
    the hardcoded window and bills, see bugs.md 8.2), and the hint masked the real
    "Invalid performance tier" answer it was printed next to."""
    if response.ok:
        return

    detail = ""
    try:
        payload = response.json()
    except ValueError:
        detail = (response.text or "").strip()[:400]
    else:
        if isinstance(payload, dict):
            error = payload.get("error", payload)
            detail = error.get("message", str(error)) if isinstance(error, dict) else str(error)
        else:
            detail = str(payload)[:400]

    hint = ""
    lowered = detail.lower()
    if "valid date" in lowered or "doesn't have a valid" in lowered:
        hint = ("  (that parameter is typed *Date* on Dune, which only accepts "
                "'YYYY-MM-DD HH:MM:SS'. Set its Type to *Text* in the query editor — the "
                "SQL writes DATE '{{x}}' / TIMESTAMP '{{x}} 00:00:00' and supplies the "
                "wrapper itself, so it needs the bare day this client sends)")
    elif "performance" in lowered or "tier" in lowered:
        hint = ("  (this account cannot select that engine tier — pass performance=None to "
                "let Dune use the account default)")
    elif response.status_code == 401:
        hint = "  (key rejected — wrong account's key for this query id, or a stale key)"
    elif response.status_code == 402:
        hint = "  (credit quota exhausted on this account — not a code problem)"
    raise requests.HTTPError(
        f"{response.status_code} {response.reason} on {what}: {detail or '(empty body)'}{hint}",
        response=response,
    )


def _json_exact(response):
    """Parse a Dune JSON body with parse_float=Decimal, so the client adds no rounding.
    Integer literals already survive as Python int; this protects every other literal.
    Does NOT recover precision Dune's engine discarded — that needs a fix in the SQL."""
    return json.loads(response.text, parse_float=Decimal)


def _paginate(url, headers, page_limit, extra_params=None, missing_hint=""):
    """Walk Dune's offset pagination and collect every row.
    Returns: (rows, column_names). Dune silently caps page_limit server-side (~30k) and
    hands back the offset to use next, so the loop trusts next_offset over its own math."""
    rows, columns, offset = [], None, 0
    while True:
        params = {"limit": page_limit, "offset": offset}
        if extra_params:
            params.update(extra_params)
        response = requests.get(url, headers=headers, params=params, timeout=200)
        _raise_for_status(response, f"GET {url}")
        payload = _json_exact(response)

        result = payload.get("result")
        if result is None:
            state = payload.get("state", "unknown")
            raise RuntimeError(
                f"No result payload (state={state}) from {url}. {missing_hint}".strip()
            )

        rows.extend(result.get("rows", []))
        columns = (result.get("metadata") or {}).get("column_names", columns)

        next_offset = payload.get("next_offset")
        if next_offset is None:        # no more pages
            break
        offset = next_offset

    return rows, columns


def _to_frame(rows, columns):
    """Build the DataFrame, preserving Dune's original column order."""
    df = pd.DataFrame(rows)
    if columns:
        df = df.reindex(columns=columns)
    return df


def audit_precision(df, limit=FLOAT_EXACT_LIMIT):
    """Find columns carrying values too large for a float to represent exactly.
    df -> {column: {"exact", "lossy", "max_digits", "sample"}}, only for columns holding
    a value past `limit`. An empty dict means nothing is at risk.

    exact and lossy are opposite findings and must never be summed: exact is a big JSON
    integer that needs no fix, lossy came from a DOUBLE whose low digits Dune already
    dropped, so only a SQL change can recover it."""
    report = {}
    for column in df.columns:
        series = df[column]
        if series.dtype.kind in "iub":          # int / unsigned / bool — exact by width
            continue

        exact = lossy = 0
        biggest = None
        for value in series.to_numpy():
            # bool is an int subclass, so it has to be rejected before the exact check.
            if value is None or isinstance(value, bool):
                continue
            if isinstance(value, _EXACT_TYPES):
                bucket = "exact"
            elif isinstance(value, _INEXACT_TYPES):
                if value != value:              # NaN (true for float and Decimal alike)
                    continue
                bucket = "lossy"
            else:                               # str, datetime, address — not numeric
                continue

            magnitude = abs(value)
            if magnitude <= limit:
                continue
            if bucket == "exact":
                exact += 1
            else:
                lossy += 1
            if biggest is None or magnitude > biggest:
                biggest = magnitude

        if exact or lossy:
            report[column] = {
                "exact": exact,
                "lossy": lossy,
                "max_digits": len(str(int(biggest))),
                "sample": str(biggest),
            }
    return report


def _print_precision_report(report, table_name, indent="  "):
    """Print an audit_precision() result as an actionable note. Silent when clean.
    Exact columns are named too, or a table of legitimate big integers would look
    identical to one that was never checked."""
    if not report:
        return

    exact = sorted(c for c, r in report.items() if r["exact"] and not r["lossy"])
    lossy = sorted(c for c, r in report.items() if r["lossy"])

    if exact:
        print(f"{indent}exact big integers, no action needed: {', '.join(exact)}")
    if lossy:
        print(f"{indent}PRECISION LOSS — arrived as DOUBLE, low digits already discarded "
              "by Dune:")
        for column in lossy:
            entry = report[column]
            print(f"{indent}  {column}: {entry['lossy']} value(s), up to "
                  f"{entry['max_digits']} digits (largest {entry['sample']})")
        print(f"{indent}  fix in the SQL for {table_name}, not here: emit the raw integer "
              f"as CAST({lossy[0]} AS VARCHAR) and do any AVG/SUM/division in Python.")


def _save_csv(df, query_id, save_dir, table_name, start_date=None, end_date=None):
    """Write the versioned CSV and return its path.
    Naming is load-bearing — transform._parse_name reads the tail of the stem:

        <table>_<id>_<start>_<end>.csv   an execute run, both bounds known
        <table>_<id>_asof_<end>.csv      an execute run with an upper bound only
                                         (reserve_config is a cumulative as-of snapshot)
        <table>_<id>_<YYYYMMDDTHHMMSSZ>.csv
                                         a "stored" fetch, which is never told a window,
                                         so the fetch time is all there is to record

    The window is in the stem on purpose: the filename now states what the file CONTAINS
    rather than when it was pulled. That was previously refused because a lexicographic
    sorted(glob(...))[-1] ranks '2025-...' below '20260822T...', so a window-named file
    would have lost to an old stamped one. Both resolvers were made window-aware first —
    transform.latest_paths() and transform.newest_matching() — so the ordering is now
    explicit rather than a side effect of string comparison.

    Note this makes a re-fetch of the SAME window overwrite in place instead of piling up
    another timestamped copy. That is the intended behaviour; the run manifest still keeps
    one dated record per execution.

    The bucket width is deliberately NOT in the name. transform._parse_name anchors all
    three of its patterns to the END of the stem, so any extra suffix makes the file
    invisible to latest_paths() rather than merely misnamed. Consequence: re-running one
    window at a different bucket_hours overwrites the previous CSV in place, and the run
    manifest is the only record of which width produced a given file."""
    folder = Path(save_dir)
    folder.mkdir(parents=True, exist_ok=True)
    prefix = table_name or DEFAULT_SAVE_DIR

    if start_date and end_date:
        suffix = f"{start_date}_{end_date}"
    elif end_date:
        suffix = f"asof_{end_date}"
    else:
        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    csv_path = folder / f"{prefix}_{query_id}_{suffix}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def fetch_query_table(query_id, api_key=None, save_dir=DEFAULT_SAVE_DIR, page_limit=100000,
                      save=True, table_name=None, audit=True):
    """Fetch a Dune query's latest stored result (no re-run, no execution credits).
    query_id/save_dir/page_limit/save/table_name -> paged fetch + CSV write.
    Returns: (DataFrame, csv_path); csv_path is None when save=False."""

    # audit is on by default because once in a CSV a DOUBLE column looks exactly like an
    # exact one. table_name is cosmetic — transform._NAME_RE reads the id out of the tail.
    headers = _headers(api_key)
    url = f"{DUNE_API_BASE}/query/{query_id}/results"
    rows, columns = _paginate(
        url, headers, page_limit,
        missing_hint=f"Query {query_id} has no stored result — run it on Dune at least "
                     "once first, or use run_query_table() to run it from here.",
    )
    df = _to_frame(rows, columns)
    if audit:
        _print_precision_report(audit_precision(df), table_name or query_id)
    csv_path = _save_csv(df, query_id, save_dir, table_name) if save else None
    return df, csv_path


# ---------------------------------------------------------------------------
# Parameterised execution path
# ---------------------------------------------------------------------------


def execute_query(query_id, query_parameters=None, api_key=None,
                  performance=DEFAULT_PERFORMANCE):
    """Trigger a fresh run of a Dune query with parameter values. Costs credits.
    query_parameters: flat {name: value} matching the {{placeholders}} in the SQL.
    performance: None omits the field so Dune picks the account's default tier; a string
    pins one, and must be one this account is entitled to.
    Returns: execution_id. Dune's REST field is `query_parameters`, never `params`."""
    if performance is not None and performance not in VALID_PERFORMANCE_TIERS:
        raise ValueError(
            f"performance must be None or one of {VALID_PERFORMANCE_TIERS}, "
            f"got {performance!r}. None omits the field and lets Dune choose."
        )

    body = {}
    if performance is not None:
        body["performance"] = performance
    if query_parameters:
        body["query_parameters"] = query_parameters

    response = requests.post(
        f"{DUNE_API_BASE}/query/{query_id}/execute",
        headers={**_headers(api_key), "Content-Type": "application/json"},
        json=body,
        timeout=60,
    )
    _raise_for_status(response, f"POST /query/{query_id}/execute")
    return response.json()["execution_id"]


def execution_status(execution_id, api_key=None):
    """Poll one execution's status. This endpoint is free — it consumes no credits.
    Returns: the raw status payload, incl. state and execution_cost_credits."""
    response = requests.get(
        f"{DUNE_API_BASE}/execution/{execution_id}/status",
        headers=_headers(api_key),
        timeout=60,
    )
    _raise_for_status(response, f"GET /execution/{execution_id}/status")
    return response.json()


def wait_for_execution(execution_id, api_key=None, poll_seconds=5, timeout_seconds=1800,
                       verbose=True):
    """Block until an execution reaches a terminal state.
    poll_seconds/timeout_seconds -> repeated free status calls.
    Returns: final status payload. Raises on FAILED/CANCELLED/EXPIRED or timeout — note a
    timeout does NOT cancel the execution, it keeps running and keeps billing.

    timeout_seconds is a POLLING budget, nothing more. It cannot extend how long Dune is
    willing to run a query: the engine has its own server-side cap set by the account's
    plan, and when that fires the execution comes back QUERY_STATE_FAILED with Dune's own
    "timed out after N minutes" message. Raising timeout_seconds has no effect on that —
    the two limits are unrelated, and only the smaller one is ever observed."""
    deadline = time.monotonic() + timeout_seconds
    last_state = None
    while True:
        status = execution_status(execution_id, api_key)
        state = status.get("state")
        if verbose and state != last_state:
            print(f"        {execution_id} -> {state}")
            last_state = state

        if state in _DONE_STATES:
            if state != "QUERY_STATE_COMPLETED":
                error = status.get("error") or {}
                message = error.get("message", "no error message")
                # Keyed off what Dune actually said, like _raise_for_status. Worth spelling
                # out because the two timeouts read identically in a traceback and only one
                # of them is ours: a server-side kill arrives as FAILED (this branch), while
                # our own budget expiring raises TimeoutError further down.
                hint = ""
                if "timed out" in message.lower():
                    hint = (
                        f"  (Dune's own execution cap, set by the account plan — NOT this "
                        f"client. timeout_seconds={timeout_seconds} is only how long we are "
                        "willing to poll and was never reached, so raising it changes "
                        "nothing. What does help: a bigger engine (performance='large'), a "
                        "narrower window, or confirming the query still declares "
                        "{{start_date}}/{{end_date}} on Dune — a parameter dropped while "
                        "editing is ignored rather than rejected, so the query silently runs "
                        "its full hardcoded range and blows the cap)"
                    )
                raise RuntimeError(
                    f"Execution {execution_id} ended as {state}: {message}{hint}"
                )
            return status

        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Execution {execution_id} still {state} after {timeout_seconds}s. "
                "It is STILL RUNNING and still billing — do not re-execute. Collect it "
                f"later with fetch_execution_table('{execution_id}')."
            )
        time.sleep(poll_seconds)


def probe_execution_size(execution_id, api_key=None):
    """Cheaply ask how big a finished result is before downloading all of it.
    Pulls a single row so the response metadata reports the totals.
    Returns: (total_rows, total_bytes); either may be None if Dune omits it."""
    response = requests.get(
        f"{DUNE_API_BASE}/execution/{execution_id}/results",
        headers=_headers(api_key),
        params={"limit": 1, "offset": 0},
        timeout=60,
    )
    _raise_for_status(response, f"GET /execution/{execution_id}/results (size probe)")
    metadata = ((response.json().get("result") or {}).get("metadata")) or {}
    return metadata.get("total_row_count"), metadata.get("total_result_set_bytes")


def fetch_execution_table(execution_id, query_id=None, api_key=None,
                          save_dir=DEFAULT_SAVE_DIR, page_limit=100000, save=True,
                          table_name=None, audit=True, start_date=None, end_date=None):
    """Download the results of a specific execution_id (results live 90 days).
    Use this to collect an execution that timed out in wait_for_execution — re-reading
    one costs export credits only, never a second run. Returns: (DataFrame, csv_path).

    start_date/end_date only name the file (see _save_csv) — they do NOT filter anything,
    since the execution already ran with whatever window it was given. Pass the window
    that execution actually used, or the filename will misdescribe its own contents."""
    if save and query_id is None:
        raise ValueError(
            "query_id is required when save=True: the filename must carry "
            "_<query_id>_<window-or-stamp> for transform._parse_name to see the file at "
            "all. An execution_id is not numeric and would be silently skipped downstream."
        )
    headers = _headers(api_key)
    url = f"{DUNE_API_BASE}/execution/{execution_id}/results"
    rows, columns = _paginate(
        url, headers, page_limit,
        missing_hint=f"Execution {execution_id} has no results — it may have expired "
                     "(results live 90 days) or never completed.",
    )
    df = _to_frame(rows, columns)
    if audit:
        _print_precision_report(audit_precision(df), table_name or query_id)
    csv_path = (_save_csv(df, query_id, save_dir, table_name, start_date, end_date)
                if save else None)
    return df, csv_path


def _validate_bucket_hours(bucket_hours):
    """Coerce a bucket width to int and reject any width 24 is not a multiple of.
    bucket_hours -> int. Accepts "6" as well as 6, since a width can arrive from an env
    var or a widget; Dune substitutes the parameter textually either way."""
    try:
        width = int(bucket_hours)
        # int() floors, so 2.5 would arrive as a silent 2 — the exact class of quiet
        # width mismatch this function exists to prevent. Reject it instead.
        if width != float(bucket_hours):
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(
            f"bucket_hours must be a whole number of hours, got {bucket_hours!r}"
        ) from None
    if width not in VALID_BUCKET_HOURS:
        raise ValueError(
            f"bucket_hours={width} is not one of {VALID_BUCKET_HOURS}. 24 must be a "
            "multiple of the width, or query 04's grid drifts out of step with the bucket "
            "formula and reserve_config silently stops joining to the other tables."
        )
    return width


def run_query_table(query_id, start_date=None, end_date=None, table_name=None,
                    api_key=None, save_dir=DEFAULT_SAVE_DIR, page_limit=100000, save=True,
                    performance=DEFAULT_PERFORMANCE, extra_params=None,
                    start_param="start_date", end_param="end_date",
                    max_result_mb=DEFAULT_MAX_RESULT_MB, poll_seconds=5,
                    timeout_seconds=1800, dry_run=False, verbose=True,
                    bucket_hours=None, bucket_param="bucket_hours"):
    """Re-run a parameterised Dune query for one date window and save the result.
    query_id + start_date/end_date -> execute -> poll -> size check -> paged download.
    Returns: (DataFrame, csv_path, meta); on dry_run (None, None, meta), nothing billed.

    bucket_hours sets the panel grain and is sent only when not None — pass None for any
    query that does not yet declare {{bucket_hours}} on Dune, since sending it there is
    accepted, ignored, and billed at the SQL's hardcoded width. Validated against
    VALID_BUCKET_HOURS before anything is executed, so a bad width costs no credits."""

    # Dates pass through verbatim as 'YYYY-MM-DD' — the SQL supplies the DATE '...'
    # wrapper — and end_date is EXCLUSIVE, so April 2025 is 2025-04-01 -> 2025-05-01.
    params = dict(extra_params or {})
    if start_date is not None:
        params[start_param] = start_date
    if end_date is not None:
        params[end_param] = end_date

    # Validated before the execute call, not after: an invalid width must cost nothing.
    # Sent only when asked for — see the docstring on why a stray value is worse than a
    # missing one here.
    if bucket_hours is not None:
        params[bucket_param] = _validate_bucket_hours(bucket_hours)

    meta = {
        "query_id": query_id,
        "table_name": table_name,
        "query_parameters": params,
        "performance": performance,
        "requested_at": datetime.now(timezone.utc).isoformat(),
    }

    if dry_run:
        meta["dry_run"] = True
        if verbose:
            preview = {}
            if performance is not None:
                preview["performance"] = performance
            if params:
                preview["query_parameters"] = params
            print(f"        DRY RUN — would POST /query/{query_id}/execute with "
                  f"{json.dumps(preview) or '{}'}")
        return None, None, meta

    execution_id = execute_query(query_id, params, api_key, performance)
    meta["execution_id"] = execution_id

    status = wait_for_execution(execution_id, api_key, poll_seconds, timeout_seconds, verbose)
    meta["execution_cost_credits"] = status.get("execution_cost_credits")
    meta["execution_started_at"] = status.get("execution_started_at")
    meta["execution_ended_at"] = status.get("execution_ended_at")

    total_rows, total_bytes = probe_execution_size(execution_id, api_key)
    total_mb = round(total_bytes / 1_048_576, 3) if total_bytes else None
    meta["total_row_count"] = total_rows
    meta["total_result_mb"] = total_mb

    if verbose:
        print(f"        result: {total_rows} rows, {total_mb} MB, "
              f"{meta['execution_cost_credits']} credits to run")

    # Aborts after the run but before the download: compute is already paid for by now,
    # so what this guard actually saves is the larger, more repeatable export bill.
    if max_result_mb is not None and total_mb is not None and total_mb > max_result_mb:
        raise RuntimeError(
            f"Result is {total_mb} MB, over the {max_result_mb} MB guard for "
            f"{table_name or query_id}. The run is already paid for; the download is not. "
            f"Narrow the window, or re-collect with "
            f"fetch_execution_table('{execution_id}', {query_id}, table_name={table_name!r}) "
            f"— or call again with max_result_mb={max_result_mb * 4:g}."
        )

    df, csv_path = fetch_execution_table(
        execution_id, query_id=query_id, api_key=api_key, save_dir=save_dir,
        page_limit=page_limit, save=save, table_name=table_name, audit=False,
        start_date=start_date, end_date=end_date,
    )
    meta["rows_downloaded"] = len(df)
    meta["csv_path"] = str(csv_path) if csv_path else None

    # audited here, not inside fetch_execution_table, so the finding reaches the manifest
    # rather than only a print that scrolls away
    meta["precision_audit"] = audit_precision(df)
    if verbose:
        _print_precision_report(meta["precision_audit"], table_name or query_id,
                                indent="        ")
    return df, csv_path, meta


def write_run_manifest(metas, manifest_dir=DEFAULT_MANIFEST_DIR):
    """Record what window each CSV in a run covers, plus execution_id and credit cost.
    The filename can't carry the window without breaking normalize.ipynb's sorted-glob.
    Written to a subdirectory — transform.latest_paths globs *.csv non-recursively."""
    folder = Path(manifest_dir)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = folder / f"run_{stamp}.json"
    path.write_text(json.dumps(metas, indent=2, default=str))
    return path


# ---------------------------------------------------------------------------
# Multi-account layer: one key per group of queries
# ---------------------------------------------------------------------------


def _env_assignment_state(name, env_lines):
    """Where does `name` stand in a .env file? -> 'live' | 'commented' | 'absent'.
    Matched per line against a real assignment, never by substring: `"DUNE_API_KEY" in
    text` is also true for DUNE_API_KEY_3, and a commented-out example would read live."""
    assign = re.compile(rf"^(?P<hash>\s*#\s*)?(export\s+)?{re.escape(name)}\s*=")
    state = "absent"
    for line in env_lines:
        match = assign.match(line)
        if not match:
            continue
        if match.group("hash") is None:
            return "live"                  # an uncommented assignment wins outright
        state = "commented"
    return state


def load_api_keys(key_env_names=None, dotenv_path=None, require=True, verbose=True):
    """Resolve one Dune API key per account group out of .env.
    key_env_names: {group: (name, fallback, ...)}; the first name that is set wins.
    Returns: (api_keys, key_source) — {group: key} and {group: env_var_name}.
    Unlike load_api_key(), several keys coexist here and some may legitimately be absent."""
    key_env_names = key_env_names or DEFAULT_KEY_ENV_NAMES
    env_path = Path(dotenv_path) if dotenv_path else Path.cwd() / ".env"

    # Explicit path, not bare load_dotenv() — the no-arg form walks caller stack frames,
    # which is fragile under papermill/Dagster. A missing .env is fine: the extraction
    # container injects the keys as environment variables instead.
    load_dotenv(env_path)

    api_keys, key_source = {}, {}
    for group, names in key_env_names.items():
        for name in names:
            value = os.getenv(name)
            if value:
                api_keys[group], key_source[group] = value, name
                break

    # "absent", "commented out" and "present but unparseable" need three different fixes,
    # and dotenv drops an unparseable line with only an easily-missed stderr warning —
    # hence the detail below. Key material is never printed, not even its length.
    if verbose:
        env_lines = env_path.read_text().splitlines() if env_path.exists() else []
        for group, names in key_env_names.items():
            if group in api_keys:
                print(f"  key {group}: loaded from {key_source[group]}")
                continue
            states = {n: _env_assignment_state(n, env_lines) for n in names}
            live = [n for n, st in states.items() if st == "live"]
            commented = [n for n, st in states.items() if st == "commented"]
            if live:
                print(f"  key {group}: {live[0]} is assigned in {env_path.name} but did "
                      "NOT parse — dotenv skipped the line. Rewrite it as")
                print(f"                  {live[0]}=yourkeyhere")
                print("                no quotes, and nothing after the value unless it "
                      "starts with '#'. A quoted value followed by any bare character "
                      "(a stray '.', say) silently drops the whole variable.")
            elif commented:
                print(f"  key {group}: {commented[0]} is COMMENTED OUT in "
                      f"{env_path.name} — uncomment it.")
            else:
                print(f"  key {group}: absent — add {names[0]}=... to {env_path.name}")

    if require and not api_keys:
        raise RuntimeError(
            f"No Dune keys loaded from {env_path}. A key reported as 'did NOT parse' or "
            "'commented out' above is a formatting problem, not a missing key."
        )
    return api_keys, key_source


def run_group(group, query_groups, api_keys, tables=None, key_source=None,
              mode="stored", start_date=None, end_date=None, param_overrides=None,
              performance=DEFAULT_PERFORMANCE, max_result_mb=DEFAULT_MAX_RESULT_MB,
              poll_seconds=5, timeout_seconds=1800, dry_run=False,
              save_dir=DEFAULT_SAVE_DIR, page_limit=100000, show=None, verbose=True,
              bucket_hours=None):
    """Fetch or execute every query owned by one account group, using that group's key.
    group looks up query_groups[group] and api_keys[group] together, so a key cannot be
    paired with another account's queries. mode: "stored" (fetch) or "execute" (re-run).
    Returns: {"tables": {name: df}, "metas": [...], "failures": {name: why}}.

    bucket_hours goes to every table in the group; opt one out with
    param_overrides={"<table>": {"bucket_hours": None}} until its SQL on Dune declares the
    parameter. Mixed widths across tables produce a panel whose (time_bucket, asset) keys
    do not line up, which no downstream stage can detect, so the opt-outs are printed."""

    if mode not in {"stored", "execute"}:
        raise ValueError(f"mode must be 'stored' or 'execute', got {mode!r}")
    if group not in query_groups:
        raise KeyError(f"No query group {group}. Known groups: {sorted(query_groups)}")
    if group not in api_keys:
        raise RuntimeError(
            f"No API key for group {group}. Add "
            f"{DEFAULT_KEY_ENV_NAMES.get(group, ('DUNE_API_KEY_?',))[0]}=... to .env "
            "and re-run load_api_keys()."
        )

    api_key = api_keys[group]
    registry = query_groups[group]
    param_overrides = param_overrides or {}

    selected = list(registry) if tables is None else list(tables)
    unknown = [t for t in selected if t not in registry]
    if unknown:
        raise KeyError(
            f"not in group {group}: {unknown}. Group {group} has: {sorted(registry)}"
        )

    if verbose:
        source = (key_source or {}).get(group, "?")
        print(f"group {group} ({source})  mode={mode}  {len(selected)} quer(y/ies)")
        if mode == "execute":
            print(f"  window {start_date} -> {end_date} (end exclusive)  dry_run={dry_run}")
            if bucket_hours is not None:
                # Named explicitly because a table running at the wrong width still
                # succeeds, still bills, and looks identical in the output.
                opted_out = [t for t in selected
                             if param_overrides.get(t, {}).get("bucket_hours", "") is None]
                sent = [t for t in selected if t not in opted_out]
                print(f"  bucket_hours={bucket_hours} -> "
                      f"{', '.join(sent) if sent else '(none)'}")
                if opted_out:
                    print(f"    NOT sent to {', '.join(opted_out)} — each runs at the "
                          "width hardcoded in its own SQL")
        print()

    out = {"tables": {}, "metas": [], "failures": {}}

    for table_name in selected:
        query_id = registry[table_name]
        if verbose:
            print(f"{table_name} ({query_id})")
        try:
            if mode == "execute":
                kwargs = dict(
                    start_date=start_date,
                    end_date=end_date,
                    bucket_hours=bucket_hours,
                    performance=performance,
                    max_result_mb=max_result_mb,
                    timeout_seconds=timeout_seconds,
                    poll_seconds=poll_seconds,
                    save_dir=save_dir,
                    page_limit=page_limit,
                    dry_run=dry_run,
                    verbose=verbose,
                )
                # e.g. {"reserve_config": {"start_date": None}} for an as-of snapshot
                # that takes only an upper bound — a None value is simply not sent
                kwargs.update(param_overrides.get(table_name, {}))
                df, csv_path, meta = run_query_table(
                    query_id, table_name=table_name, api_key=api_key, **kwargs
                )
                out["metas"].append(meta)
                if df is None:             # dry run — nothing executed, nothing to show
                    continue
            else:
                df, csv_path = fetch_query_table(
                    query_id, api_key=api_key, save_dir=save_dir,
                    page_limit=page_limit, table_name=table_name,
                )
        # one dead id must not abort the rest — collecting failures keeps 402 (quota
        # exhausted) distinguishable from 404 (wrong key for this group)
        except Exception as exc:
            out["failures"][table_name] = f"{query_id}: {type(exc).__name__}: {exc}"
            if verbose:
                print(f"  FAILED  {type(exc).__name__}: {exc}\n")
            continue

        if verbose:
            print(f"  ok      {len(df)} rows -> {csv_path}\n")
        if show is not None:          # e.g. IPython display — passed in, never imported here
            show(df)
        out["tables"][table_name] = df

    if verbose:
        if mode == "execute" and dry_run:
            print(f"\ngroup {group}: DRY RUN — {len(selected)} request(s) shown, none "
                  "sent, nothing billed. Set dry_run=False to run them.")
        else:
            print(f"\ngroup {group}: {len(out['tables'])}/{len(selected)} tables "
                  f"({mode} mode)")

    if out["metas"] and not dry_run:
        # window + credit cost go in the manifest, not the filename (see _save_csv)
        spent = sum(m.get("execution_cost_credits") or 0 for m in out["metas"])
        manifest = write_run_manifest(out["metas"])
        if verbose:
            print(f"  execution credits spent on account {group}: {spent}")
            print(f"  manifest: {manifest}")

    if verbose and out["failures"]:
        print("  failures:")
        for name, why in out["failures"].items():
            print(f"    {name}: {why}")

    return out


def make_runner(query_groups, api_keys, key_source=None, show=None, **defaults):
    """Bind a registry, its keys and default settings into a one-argument runner.
    Keeps call sites at `run(2, ["flashloan"])`, every setting still overridable per call.
    Returns: run(...) -> the run_group() dict, also kept on run.results[group]."""
    results = {}

    def run(group, tables=None, **overrides):
        # bound values go in first so a per-call override wins instead of colliding
        settings = {"show": show, "key_source": key_source, **defaults, **overrides}
        out = run_group(group, query_groups, api_keys, tables=tables, **settings)
        results[group] = out
        return out

    run.results = results
    return run


if __name__ == "__main__":
    # Stored result:  python dune_fetch.py <query_id>
    # Fresh window :  python dune_fetch.py <query_id> <start_date> <end_date> [bucket_hours]
    qid = int(sys.argv[1])
    if len(sys.argv) >= 4:
        frame, path, run_meta = run_query_table(
            qid, sys.argv[2], sys.argv[3],
            bucket_hours=sys.argv[4] if len(sys.argv) >= 5 else None,
        )
        print(f"Ran {qid} [{sys.argv[2]} -> {sys.argv[3]}]: "
              f"{len(frame)} rows, {run_meta['execution_cost_credits']} credits -> {path}")
    else:
        frame, path = fetch_query_table(qid)
        print(f"Fetched {len(frame)} rows -> {path}")
