"""Transformation helpers for the Aave V3.1 Dune result tables.

READS the versioned CSVs in query_result_data/ and never writes back there. A fetch adds
a file named for the DATE WINDOW it contains (`_<start>_<end>` / `_asof_<end>`), or for
the fetch time when the window is unknowable — a "stored" fetch is never told one.

Since two extraction windows now sit side by side in that folder, "which file is current"
is the wrong question — "which files belong together" is the right one. Resolution is
therefore BATCHED and driven by CONTENT: month_span() reads the first and last few rows of
a file and reports the (year, month) it starts and ends in, and batches() groups every
extract that agrees on that span. The filename is only a tiebreak between two files
claiming the same span. Nothing downstream should sort a glob.
"""

import csv
import os
import re
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd

# Make sibling src/ modules importable however this file is loaded (notebook, script, or import).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_validation import TABLE_LABELS, table_name_from_path

# Folder holding the raw, versioned extracts. Read-only from this stage.
SOURCE_DIR = "query_result_data"

# The column every time-series extract is keyed on, and how many rows are sampled from each
# end of a file to establish the window it holds. Five is enough to survive a couple of
# out-of-order rows at an edge and still costs two small reads, not a full parse.
TIME_COL = "time_bucket"
EDGE_ROWS = 5

# Three versioned-suffix forms, all ending the stem, all carrying the query id:
#   _<id>_<start>_<end>     window-named  — what dune_fetch writes for an execute run
#   _<id>_asof_<end>        window-named  — as-of snapshot, upper bound only (reserve_config)
#   _<id>_<YYYYMMDDTHHMMSSZ>  fetch-stamped — legacy, and still what a "stored" fetch writes
#                                             because that path is never told a window
# Order matters when matching: the range form must be tried before the stamp form, since
# a bare `_(\d+)_` prefix could otherwise claim part of a date.
# The id is required to be >= 6 digits, which is NOT cosmetic: a table name ending in a
# number is otherwise indistinguishable from `<table>_<id>`, and `DF_common_1_<start>_<end>`
# parsed as table "DF_common" + id "1" — the frame came back under the wrong label. Every
# real Dune id here is 7 digits (7702138 … 8595542), so 6 is a safe floor that still rejects
# a trailing `_1` and a bare year.
_RANGE_RE = re.compile(r"_(\d{6,})_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})$")
_ASOF_RE = re.compile(r"_(\d{6,})_asof_(\d{4}-\d{2}-\d{2})$")
_STAMP_RE = re.compile(r"_(\d{6,})_(\d{8}T\d{6}Z)$")
# A fourth form with no query id at all: `<table>_<start>_<end>.csv`, which is what
# save_batch() writes for a TRANSFORMED frame — those are built here, not fetched, so they
# have no Dune id to carry. Tried last, after every id-bearing form has had its chance.
_RANGE_NOID_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})$")

# Kept as an alias: other code (and older notes) refer to transform._NAME_RE.
_NAME_RE = _STAMP_RE

# Window-named files outrank fetch-stamped ones. A stamp says only when a pull happened;
# a window says what the file actually contains, which is the thing downstream needs.
_TIER_WINDOW, _TIER_STAMP = 1, 0


def _parse_name(path):
    """Return (query_id, sort_key) for a versioned CSV path, or (None, None).

    sort_key is (tier, end, start) and is meant for ordering, not display:
      tier   window-named beats fetch-stamped (see _TIER_WINDOW)
      end    latest end date wins — the file covering the most recent data
      start  only a tiebreak; the earliest start wins, so that between two files ending on
             the same day the wider one is chosen.
    For a stamped file the stamp stands in for `end`, which keeps stamped files ordered
    among themselves exactly as before. This is now a tiebreak WITHIN a batch, not the
    thing that decides which window is current — see month_span()."""
    stem = Path(path).stem

    match = _RANGE_RE.search(stem)
    if match:
        return match.group(1), (_TIER_WINDOW, match.group(3), match.group(2))

    match = _ASOF_RE.search(stem)
    if match:
        # No lower bound at all, so it is the widest possible window for that end date.
        return match.group(1), (_TIER_WINDOW, match.group(2), "")

    match = _STAMP_RE.search(stem)
    if match:
        return match.group(1), (_TIER_STAMP, match.group(2), "")

    match = _RANGE_NOID_RE.search(stem)
    if match:
        # No id to return, but the window still orders these against each other.
        return None, (_TIER_WINDOW, match.group(2), match.group(1))

    return None, None


# --- naming: which table a file IS ---
# data_validation.TABLE_LABELS is the registry, and an id missing from it is SILENT: the file
# loads under "query_<id>", a name nothing downstream asks for, while the real name keeps
# resolving to some other window's file. That comment in data_validation records it biting
# three times; 8595542 (last year's oracle) is the fourth. dune_fetch already writes
# "<table>_<id>_<window>.csv", so the table name is sitting in the filename — recovering it
# from there beats surfacing an opaque id. The registry still wins wherever it speaks, so the
# deliberate "*_legacy" renames stand.

def label_for(path):
    """Table label for a versioned CSV path.
    Returns: (label, source) — source is 'registry' when TABLE_LABELS knows the query id,
    'filename' when the name had to be recovered from the stem. Anchored on the id parsed by
    _parse_name, so a file whose name carries no id keeps the old query_<id> behaviour."""
    stem = Path(path).stem
    qid, _ = _parse_name(path)
    if qid and qid in TABLE_LABELS:
        return TABLE_LABELS[qid], "registry"
    if qid:
        name = stem.split(f"_{qid}_")[0]
        if name and name != stem:
            return name, "filename"
    match = _RANGE_NOID_RE.search(stem)
    if match:
        return stem[: match.start()], "filename"    # transformed export: window, no id
    return table_name_from_path(path), "registry"


# --- what a file CONTAINS: the month span read out of its own rows ---
# A filename is a claim; the rows are the fact, and the two disagree in the folder today —
# reserve_config's `_asof_2026-03-31` name carries no start at all, yet its rows start
# 2025-04-01, which is what puts it in the same batch as everything else from that window.
# Only the first and last EDGE_ROWS rows are read (the tail by seeking backwards from the
# end), so establishing the span of a 24 MB extract costs two small reads, not a parse.

def _split_row(raw):
    """Split one raw CSV line into fields (quote-aware, unlike raw.split(','))."""
    text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
    return next(csv.reader([text.rstrip("\r\n")]), [])


def _tail_lines(handle, n_rows, block=65536):
    """Last n_rows non-empty lines of an open binary file, read backwards in blocks.
    Keeps reading until more than n_rows newlines are in hand, which is what guarantees
    the lines finally returned are complete rather than sliced mid-row."""
    handle.seek(0, os.SEEK_END)
    pos = handle.tell()
    data = b""
    while pos > 0 and data.count(b"\n") <= n_rows:
        step = min(block, pos)
        pos -= step
        handle.seek(pos)
        data = handle.read(step) + data
    return [line for line in data.splitlines() if line.strip()][-n_rows:]


def read_edge_values(source, column=TIME_COL, n_rows=EDGE_ROWS):
    """First and last `n_rows` values of one column, without reading the middle of the file.
    source: a CSV path or an in-memory DataFrame; column; n_rows.
    Returns: (head_values, tail_values) — both empty when the column is absent."""
    if isinstance(source, pd.DataFrame):
        if column not in source.columns:
            return [], []
        return list(source[column].head(n_rows)), list(source[column].tail(n_rows))

    with open(Path(source), "rb") as handle:
        header_raw = handle.readline()
        fields = _split_row(header_raw)
        if column not in fields:
            return [], []
        index = fields.index(column)

        head_raw = []
        for _ in range(n_rows):
            line = handle.readline()
            if not line:
                break
            head_raw.append(line)
        tail_raw = _tail_lines(handle, n_rows)

    header = header_raw.decode("utf-8", "replace").rstrip("\r\n")

    def value_of(line):
        row = _split_row(line)
        return row[index] if len(row) > index else None

    head = [value_of(line) for line in head_raw]
    # A file shorter than the block read hands its header back in the tail — drop it.
    tail = [value_of(line) for line in tail_raw
            if line.decode("utf-8", "replace").rstrip("\r\n") != header]
    return head, tail


def _edge_stamps(source, column=TIME_COL, n_rows=EDGE_ROWS):
    """Parsed UTC timestamps for `source`, unparseable values dropped.
    Both time_bucket spellings in the folder land here — '… 00:00:00.000 UTC' and a bare
    '… 00:00:00' — so the suffix is stripped before parsing rather than trusted to it.

    A FILE is sampled at both edges (cheap, and the extracts arrive ORDER BY time_bucket);
    an in-memory FRAME is read in full, because it is already in memory, nothing guarantees
    a frame was left sorted, and a frame is what save_batch() bakes a date into — a name is
    permanent in a way a lookup is not."""
    if isinstance(source, pd.DataFrame):
        column_values = source[column] if column in source.columns else []
        values = [str(v) for v in column_values if v is not None and str(v) != "nan"]
    else:
        head, tail = read_edge_values(source, column=column, n_rows=n_rows)
        values = [str(v) for v in head + tail if v is not None and str(v) != "nan"]
    if not values:
        return pd.Series([], dtype="datetime64[ns, UTC]")
    cleaned = pd.Series(values, dtype="object").str.replace(" UTC", "", regex=False)
    return pd.to_datetime(cleaned, utc=True, format="mixed", errors="coerce").dropna()


def month_key(value):
    """'YYYY-MM' for one timestamp-ish value, or None when it cannot be parsed."""
    stamp = pd.to_datetime(str(value).replace(" UTC", ""), utc=True, errors="coerce")
    return None if pd.isna(stamp) else stamp.strftime("%Y-%m")


def month_span(source, column=TIME_COL, n_rows=EDGE_ROWS):
    """The (start_month, end_month) a table covers, as 'YYYY-MM' — YEAR AND MONTH ONLY.
    source: CSV path or DataFrame; column; n_rows.
    Returns: ('2025-04', '2026-03'), or (None, None) when there is no parseable time column.

    Deliberately month-resolution: the extracts in one window disagree on the day (one ends
    2026-03-30 22:00, another 2026-03-31 22:00) purely because of when the last event of
    each kind landed. Matching on the day would split one batch into three; matching on the
    month groups exactly what belongs together. Min/max over both edges, not head-then-tail,
    so a file that is not ordered by time still reports the right span."""
    stamps = _edge_stamps(source, column=column, n_rows=n_rows)
    if stamps.empty:
        return (None, None)
    return (stamps.min().strftime("%Y-%m"), stamps.max().strftime("%Y-%m"))


def spans_match(*sources, column=TIME_COL, n_rows=EDGE_ROWS):
    """True when every source covers the same month span — the generic table matcher.
    sources: any mix of CSV paths and DataFrames.
    Returns: bool; False if any source has no readable time column (unknown never matches)."""
    spans = {month_span(s, column=column, n_rows=n_rows) for s in sources}
    return len(spans) == 1 and (None, None) not in spans


def span_label(span):
    """Display form of a span key: ('2025-04','2026-03') -> '2025-04 → 2026-03'."""
    start, end = span
    return f"{start} → {end}" if start else "static"


# --- batches: every extract that agrees on a month span belongs to one ---

def _coverage_records(source_dir=SOURCE_DIR, pattern="*.csv", column=TIME_COL,
                      n_rows=EDGE_ROWS):
    """One record per CSV in the folder: label, path, edge timestamps, month span, name key."""
    records = []
    for path in sorted(Path(source_dir).glob(pattern)):
        stamps = _edge_stamps(path, column=column, n_rows=n_rows)
        first = stamps.min() if not stamps.empty else None
        last = stamps.max() if not stamps.empty else None
        label, label_from = label_for(path)
        records.append({
            "label": label,
            "label_from": label_from,
            "file": path.name,
            "path": path,
            "first_time": first,
            "last_time": last,
            "start_month": first.strftime("%Y-%m") if first is not None else None,
            "end_month": last.strftime("%Y-%m") if last is not None else None,
            "name_key": _parse_name(path)[1],
        })
    return records


def _pick(records):
    """One file per (span, label) when several claim the same span.
    Falls back to the filename rule of the pre-batch resolver: window-named beats
    fetch-stamped, then latest end, then earliest start (the widest file wins a tie)."""
    keyed = [r for r in records if r["name_key"]]
    if not keyed:
        return sorted(records, key=lambda r: r["file"])[-1]["path"]
    top = max(r["name_key"][:2] for r in keyed)
    tied = [r for r in keyed if r["name_key"][:2] == top]
    return min(tied, key=lambda r: r["name_key"][2])["path"]


def table_coverage(source_dir=SOURCE_DIR, pattern="*.csv", column=TIME_COL,
                   n_rows=EDGE_ROWS):
    """What every file in the folder actually contains, one row per file.
    Returns: DataFrame [label, label_from, file, first_time, last_time, start_month,
    end_month], sorted by span then label. Static tables (no time column) sort last with a
    null span. Read `label_from`: 'filename' means TABLE_LABELS has no entry for that query
    id and the name was recovered from the file — the registry needs the id added."""
    frame = pd.DataFrame(_coverage_records(source_dir, pattern, column, n_rows))
    if frame.empty:
        return frame
    return (frame.drop(columns=["path", "name_key"])
                 .sort_values(["start_month", "end_month", "label"], na_position="last")
                 .reset_index(drop=True))


def batches(source_dir=SOURCE_DIR, pattern="*.csv", column=TIME_COL, n_rows=EDGE_ROWS):
    """Group the extracts by the month span they cover.
    Returns: {(start_month, end_month): {label: Path}}, spans in ascending order — so
    list(batches())[-1] is the newest window. Static tables are excluded (static_paths())."""
    grouped = {}
    for record in _coverage_records(source_dir, pattern, column, n_rows):
        span = (record["start_month"], record["end_month"])
        if span == (None, None):
            continue
        grouped.setdefault(span, {}).setdefault(record["label"], []).append(record)
    return {
        span: {label: _pick(records) for label, records in sorted(grouped[span].items())}
        for span in sorted(grouped)
    }


def static_paths(source_dir=SOURCE_DIR, pattern="*.csv", column=TIME_COL,
                 n_rows=EDGE_ROWS):
    """{label: Path} for the window-independent tables (the decimal references).
    They carry no time column, so they belong to every batch rather than to one of them."""
    grouped = {}
    for record in _coverage_records(source_dir, pattern, column, n_rows):
        if record["start_month"] is None:
            grouped.setdefault(record["label"], []).append(record)
    return {label: _pick(records) for label, records in sorted(grouped.items())}


def latest_batch(source_dir=SOURCE_DIR, **kwargs):
    """The newest month span present in the folder, as a (start_month, end_month) key."""
    groups = batches(source_dir, **kwargs)
    if not groups:
        raise FileNotFoundError(f"No time-series extracts under {source_dir!r}")
    return list(groups)[-1]


def resolve_batch(batch, groups):
    """Turn a caller's batch argument into a span key present in `groups`.
    batch: None (newest), a ('YYYY-MM','YYYY-MM') key, or a single 'YYYY-MM' matching one
    endpoint of exactly one batch. Raises rather than guessing when it matches several."""
    if batch is None:
        return list(groups)[-1]
    if isinstance(batch, (tuple, list)):
        span = tuple(batch)
        if span not in groups:
            raise KeyError(f"No batch {span}. Available: {list(groups)}")
        return span
    hits = [span for span in groups if batch in span]
    if not hits:
        raise KeyError(f"No batch containing month {batch!r}. Available: {list(groups)}")
    if len(hits) > 1:
        raise KeyError(f"Month {batch!r} matches {hits} — name the full span instead.")
    return hits[0]


def batch_paths(batch=None, source_dir=SOURCE_DIR, include_static=True, **kwargs):
    """{label: Path} for one batch — the single resolver every loader here goes through.
    batch: see resolve_batch; include_static adds the decimal references."""
    groups = batches(source_dir, **kwargs)
    if not groups:
        return dict(static_paths(source_dir, **kwargs)) if include_static else {}
    paths = dict(groups[resolve_batch(batch, groups)])
    if include_static:
        paths.update(static_paths(source_dir, **kwargs))
    return paths


def batch_matrix(source_dir=SOURCE_DIR, **kwargs):
    """label × span grid of which file each batch resolves to — the batch overview.
    Returns: DataFrame indexed by label, one column per span, empty string where a table is
    MISSING from that batch. A blank cell is the thing to look at: it means a stage asking
    for that table in that window silently falls through to another window's file."""
    groups = batches(source_dir, **kwargs)
    grid = {span_label(span): {label: path.name for label, path in paths.items()}
            for span, paths in groups.items()}
    static = {label: path.name for label, path in static_paths(source_dir, **kwargs).items()}
    if static:
        grid["static"] = static
    return pd.DataFrame(grid).fillna("").sort_index()


def latest_paths(source_dir=SOURCE_DIR):
    """Map each table label -> the winning CSV path, for the NEWEST batch.
    Returns: {label: Path}, decimal references included.

    Back-compatible entry point (backend/, validation.ipynb): same shape as before, but
    "winning" now means "in the newest complete window", not "newest-looking filename"."""
    return batch_paths(None, source_dir=source_dir)


def newest_matching(pattern, source_dir=SOURCE_DIR):
    """Newest versioned CSV whose filename matches a glob, e.g. 'reserve_config_*.csv'.
    Returns: Path, or None when nothing matches.

    Name-based on purpose — this is the by-filename escape hatch for callers that already
    know the file they want. Use batch_paths()/load_table() when the question is which
    window a table belongs to. Never sorted(glob(...))[-1]: a plain lexicographic sort puts
    '..._2025-04-01_2026-03-31.csv' BEFORE '..._20260822T191557Z.csv' (they differ at the
    4th character, '5' < '6'), so an old fetch-stamped file silently keeps winning."""
    matches = list(Path(source_dir).glob(pattern))
    if not matches:
        return None
    scored = [(key, p) for key, p in ((_parse_name(p)[1], p) for p in matches) if key]
    if not scored:
        return sorted(matches)[-1]        # unversioned names: nothing better than a sort
    top = max(key[:2] for key, _ in scored)
    tied = [(key, p) for key, p in scored if key[:2] == top]
    return min(tied, key=lambda item: item[0][2])[1]


def list_tables(source_dir=SOURCE_DIR, batch=None, **kwargs):
    """Return a {label: filename} view of one batch (newest by default)."""
    return {label: path.name
            for label, path in batch_paths(batch, source_dir=source_dir, **kwargs).items()}


def load_table(table, batch=None, source_dir=SOURCE_DIR, **kwargs):
    """Load one result table as a DataFrame (read-only).
    table: a label resolved within `batch`, or a direct CSV path; batch: see resolve_batch.
    Returns: DataFrame as-loaded — time_bucket / asset stay strings."""
    path = Path(table)
    if path.suffix == ".csv" and path.exists():
        return pd.read_csv(path)

    paths = batch_paths(batch, source_dir=source_dir, **kwargs)
    if table not in paths:
        raise KeyError(
            f"Unknown table '{table}' in batch {batch or 'latest'}. "
            f"Available labels: {sorted(paths)}"
        )
    return pd.read_csv(paths[table])


def load_batch(batch=None, tables=None, source_dir=SOURCE_DIR, include_static=True, **kwargs):
    """Load a whole batch into {label: DataFrame} (read-only).
    batch: see resolve_batch; tables: labels to restrict to (all of them by default).
    Every frame returned covers the same month span — that is the point of loading by
    batch rather than table by table."""
    paths = batch_paths(batch, source_dir=source_dir, include_static=include_static, **kwargs)
    if tables is not None:
        missing = [t for t in tables if t not in paths]
        if missing:
            raise KeyError(f"Not in batch {batch or 'latest'}: {missing}. "
                           f"Available: {sorted(paths)}")
        paths = {label: paths[label] for label in tables}
    return {label: pd.read_csv(path) for label, path in paths.items()}


def load_all(source_dir=SOURCE_DIR):
    """Load the newest batch into {label: DataFrame} (read-only). Alias for load_batch()."""
    return load_batch(source_dir=source_dir)


# --- saving: one folder per batch, one dated file per table ---
# The read side of this module decides which files belong together by reading their rows
# (month_span/batches). The write side is the same idea pointed the other way: a frame is
# asked what window it holds, and that answer becomes both the folder it lands in and the
# dates in its filename. Nothing here computes or alters data — it only names and places it.
#
# Two levels of date on purpose:
#   folder  batch_<start_month>_<end_month>   the BATCH identity, month-level, so the whole
#                                             window keeps one folder across re-runs even
#                                             when a re-extract shifts the last day
#   file    <table>_<start_date>_<end_date>   the table's OWN timeline, day-level, because
#                                             frames in one batch legitimately end on
#                                             different days (a 7d rollup stops ~6 days
#                                             short of the 2h panel — see the export cell's
#                                             CUTOFF_7D) and that difference must stay
#                                             readable rather than be rounded away
# The filename form matches query_result_data's `<table>_<start>_<end>.csv`, minus the query
# id these frames do not have, so one set of resolvers reads both folders.

OUTPUT_DIR = "transformed_data"
OUTPUT_PATTERN = "*/*.csv"        # <out_dir>/<batch folder>/<table>_<start>_<end>.csv
BATCH_DIR_PREFIX = "batch_"
MANIFEST_NAME = "_manifest.json"


def date_span(source, column=TIME_COL, n_rows=EDGE_ROWS):
    """The ('YYYY-MM-DD', 'YYYY-MM-DD') a table actually covers — day-level companion to
    month_span, used for filenames where the exact timeline is the point.
    Returns: (None, None) when there is no parseable time column."""
    stamps = _edge_stamps(source, column=column, n_rows=n_rows)
    if stamps.empty:
        return (None, None)
    return (stamps.min().strftime("%Y-%m-%d"), stamps.max().strftime("%Y-%m-%d"))


def batch_dir_name(span, prefix=BATCH_DIR_PREFIX):
    """Folder name for a batch: ('2025-04','2026-03') -> 'batch_2025-04_2026-03'."""
    start, end = span
    if not start:
        raise ValueError("cannot name a batch folder for a span with no dates")
    return f"{prefix}{start}_{end}"


def frame_spans(frames, column=TIME_COL, n_rows=EDGE_ROWS):
    """{name: (start_month, end_month)} for a {name: DataFrame} mapping."""
    return {name: month_span(df, column=column, n_rows=n_rows) for name, df in frames.items()}


def covering_span(spans):
    """The span a set of tables collectively covers: earliest start month, latest end month.
    Tables with no time column are ignored. Raises when none of them has one."""
    dated = [s for s in spans if s[0]]
    if not dated:
        raise ValueError("no table carries a parseable time column — nothing to date")
    return (min(s[0] for s in dated), max(s[1] for s in dated))


def flat_exports(out_dir=OUTPUT_DIR):
    """CSVs sitting loose in out_dir rather than in a batch folder.
    These are the pre-batch exports. They are NOT read by output_paths(), but any stage
    still hardcoding `transformed_data/<name>.csv` keeps reading them — a stale frame that
    raises nothing. Listed so that cannot happen quietly."""
    return sorted(p.name for p in Path(out_dir).glob("*.csv"))


def save_batch(frames, out_dir=OUTPUT_DIR, batch=None, stage=None, index=False,
               column=TIME_COL, n_rows=EDGE_ROWS, replace=True, manifest=True, verbose=True):
    """Write a set of frames as ONE batch: `<out_dir>/batch_<start>_<end>/<table>_<d0>_<d1>.csv`.
    frames: {table_name: DataFrame}; batch: force the folder's span instead of deriving it;
    stage: nest one level deeper, `batch_<span>/<stage>/`, so two stages writing into the
    same out_dir cannot collide; replace: drop this batch's older file for a table when the
    dates in its name changed; manifest: also write `_manifest.json` beside the CSVs.
    Returns: a DataFrame, one row per table, with what was written and what it superseded.

    The span is derived, not passed: each frame is asked (via month_span) what it holds, and
    the folder covers earliest start to latest end. A frame whose span disagrees with the
    rest is written anyway and flagged in the result — the disagreement is information, and
    swallowing it here is how a half-window export becomes invisible.

    `stage` scopes everything that keys off the target folder — superseded-file detection and
    the manifest — automatically, because they all read `target`. `transformed_data/` passes
    no stage and is untouched; `processed_data/` passes one, because it is the folder a
    second stage would otherwise write its own tables into beside transform's."""
    spans = frame_spans(frames, column=column, n_rows=n_rows)
    span = tuple(batch) if batch else covering_span(spans.values())
    target = Path(out_dir) / batch_dir_name(span)
    if stage:
        target = target / stage
    target.mkdir(parents=True, exist_ok=True)

    existing = {p: label_for(p)[0] for p in target.glob("*.csv")}
    rows, manifest_rows = [], []
    for name, df in frames.items():
        first, last = date_span(df, column=column, n_rows=n_rows)
        filename = f"{name}_{first}_{last}.csv" if first else f"{name}.csv"
        path = target / filename

        superseded = [p for p, label in existing.items() if label == name and p != path]
        if replace:
            for stale in superseded:
                stale.unlink()

        df.to_csv(path, index=index)
        row = {
            "table": name,
            "file": filename,
            "rows": len(df),
            "cols": df.shape[1],
            "first_date": first,
            "last_date": last,
            "span": span_label(spans[name]) if spans[name][0] else "no time column",
            "matches_batch": spans[name] == span or not spans[name][0],
            "superseded": ", ".join(sorted(p.name for p in superseded)),
        }
        rows.append(row)
        # The full column list goes to the MANIFEST only, never to `report` — that frame is
        # display()ed in every notebook as a human glance, and a 30-name list per row would
        # drown the five numbers anyone actually reads there.
        manifest_rows.append({
            **{k: v for k, v in row.items() if k != "span"},
            "columns": [str(c) for c in df.columns],
            "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
        })

    report = pd.DataFrame(rows)
    if manifest:
        import json
        # Known limitation, named rather than solved: these are the IN-MEMORY dtypes at write
        # time, not what the next stage sees after pd.read_csv(). They agree for every table
        # here today; a stage that reloads and re-checks would catch it if they ever did not.
        # Two stages (transform, feature_addition) write into the SAME transformed_data/batch_*
        # folder, so the manifest is MERGED, not replaced: this call's tables overwrite their own
        # entries, and every other entry survives while its file still exists. Replacing it left a
        # 14-CSV folder with an 8-table manifest, which check_manifest() could not audit. Each
        # entry carries its own saved_utc, so a later stage can see which stage wrote what, when.
        saved_utc = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        for entry in manifest_rows:
            entry["saved_utc"] = saved_utc
        manifest_path = target / MANIFEST_NAME
        kept = []
        if manifest_path.exists():
            try:
                previous = json.loads(manifest_path.read_text()).get("tables", [])
            except ValueError:
                previous = []
                print(f"  NOTE: {manifest_path} was unreadable and is rebuilt from this call only")
            written = {entry["table"] for entry in manifest_rows}
            kept = [t for t in previous
                    if t.get("table") not in written and (target / t.get("file", "")).exists()]
        manifest_path.write_text(json.dumps({
            "batch": list(span),
            "stage": stage,
            "saved_utc": saved_utc,
            "tables": kept + manifest_rows,
        }, indent=2))

    if verbose:
        print(f"wrote {len(report)} tables -> {target}/")
        odd = report.loc[~report["matches_batch"], "table"].tolist()
        if odd:
            # Day-level differences (a 7d rollup ending ~6 days short) do NOT land here —
            # those still match at month level. This fires only when a frame covers a
            # different set of MONTHS, which is what a half-window export looks like.
            print(f"  NOTE: month span differs from the batch folder: {odd}")
        loose = flat_exports(out_dir)
        if loose:
            print(f"  WARNING: {len(loose)} pre-batch CSV(s) still loose in {out_dir}/ — any "
                  f"stage reading '{out_dir}/<name>.csv' by path is still reading those: "
                  f"{', '.join(loose)}")
    return report


def output_paths(batch=None, out_dir=OUTPUT_DIR):
    """{label: Path} for one saved batch under out_dir — the read side of save_batch().
    Ignores anything loose in out_dir (see flat_exports)."""
    return batch_paths(batch, source_dir=out_dir, include_static=False,
                       pattern=OUTPUT_PATTERN)


def load_output(table, batch=None, out_dir=OUTPUT_DIR):
    """Load one saved frame back by label — `load_output('DF_common_final')`.
    batch: see resolve_batch (None = newest saved batch)."""
    paths = output_paths(batch, out_dir=out_dir)
    if table not in paths:
        raise KeyError(f"'{table}' not in saved batch {batch or 'latest'} under {out_dir}/. "
                       f"Available: {sorted(paths)}")
    return pd.read_csv(paths[table])


# --- pinning a run to ONE timeline ---
# batch= on every loader is a per-call promise, and a per-call promise is one someone forgets:
# a load cell missing the argument silently falls back to the newest window, and the run ends
# up mixing two years with nothing raised. use_batch() resolves the batch ONCE and hands back
# a run that can only see that batch's files — a label outside it raises rather than resolving
# elsewhere, so "which window is this frame from" stops being a question about call sites.
# It is also the cheaper path: the folder is scanned once per run, not once per load.

class BatchRun:
    """One pipeline run pinned to one timeline. Built by use_batch(), not directly.

    Attributes: span ('2025-04','2026-03'), paths {label: Path} — every file this run may
    read, and nothing else — source_dir, and used {label: dict} filled in as frames load."""

    def __init__(self, span, paths, source_dir=SOURCE_DIR):
        self.span = span
        self.paths = dict(paths)
        self.source_dir = source_dir
        self.used = {}

    def __repr__(self):
        return (f"<BatchRun {span_label(self.span)} · {len(self.paths)} tables available "
                f"· {len(self.used)} loaded · from {self.source_dir}/>")

    def tables(self):
        """Every table this run may read, as a DataFrame [label, file] — the whole menu."""
        return pd.DataFrame({"label": list(self.paths),
                             "file": [p.name for p in self.paths.values()]})

    def load(self, table):
        """Load one table from THIS run's batch.
        Refuses any label outside it — a CSV path is not accepted either, because a path is
        exactly how a file from another window would get in. Use tf.load_table() for that."""
        if table not in self.paths:
            raise KeyError(
                f"'{table}' is not in this run's batch {span_label(self.span)}. "
                f"Available: {sorted(self.paths)}"
            )
        path = self.paths[table]
        frame = pd.read_csv(path)
        self.used[table] = {"table": table, "file": path.name,
                            "rows": len(frame), "cols": frame.shape[1]}
        return frame

    def load_all(self, tables=None):
        """Load every table of the batch (or the named subset) into {label: DataFrame}."""
        return {label: self.load(label)
                for label in (tables if tables is not None else self.paths)}

    def path(self, table):
        """The file backing one table, for a stage that rewrites it IN PLACE.
        normalize.ipynb is the only such stage: it scales raw units and overwrites the same
        CSV, so it needs the path the run resolved rather than a fresh glob — otherwise it
        could normalize one window's file while the run believes it read another's."""
        if table not in self.paths:
            raise KeyError(f"'{table}' is not in this run's batch {span_label(self.span)}. "
                           f"Available: {sorted(self.paths)}")
        return self.paths[table]

    def folder(self, root, create=True):
        """This run's batch folder under any root: 'validation_results/batch_2025-04_2026-03'.
        For outputs that are NOT time-series frames — check results, reports — and so cannot
        carry their own dates. They still need one folder per timeline, or validating the
        2024 window silently overwrites the 2025 window's report."""
        target = Path(root) / batch_dir_name(self.span)
        if create:
            target.mkdir(parents=True, exist_ok=True)
        return target

    def loaded(self):
        """What this run has actually read so far, one row per table.
        The audit trail for "no other table was imported": if a file is not here, this run
        never opened it."""
        return pd.DataFrame(list(self.used.values()))

    def save(self, frames, out_dir=OUTPUT_DIR, **kwargs):
        """Save frames under THIS run's timeline — see save_batch().
        The folder is the run's span, not one re-derived from the frames, so an export always
        lands under the window it was produced from; a frame that disagrees is still written
        and flagged in the report rather than quietly re-foldered."""
        return save_batch(frames, out_dir=out_dir, batch=self.span, **kwargs)


def timelines(source_dir=SOURCE_DIR, **kwargs):
    """The selectable timelines in a folder, one row each — the menu for use_batch().
    Returns: DataFrame [batch, tables, first_date, last_date, select_with], oldest first.
    `select_with` is what to paste into BATCH: the month alone where that is unambiguous,
    otherwise the full span."""
    grouped = {}
    for record in _coverage_records(source_dir, **kwargs):
        span = (record["start_month"], record["end_month"])
        if span == (None, None):
            continue
        grouped.setdefault(span, []).append(record)

    months = [m for span in grouped for m in span]
    rows = []
    for span in sorted(grouped):
        records = grouped[span]
        unique_start = months.count(span[0]) == 1
        rows.append({
            "batch": span_label(span),
            "tables": len({r["label"] for r in records}),
            "first_date": min(r["first_time"] for r in records).strftime("%Y-%m-%d"),
            "last_date": max(r["last_time"] for r in records).strftime("%Y-%m-%d"),
            "select_with": repr(span[0]) if unique_start else repr(span),
        })
    return pd.DataFrame(rows)


def use_batch(batch=None, source_dir=SOURCE_DIR, include_static=True, **kwargs):
    """Pin a run to one timeline and return the BatchRun that owns its file list.
    batch: see resolve_batch — None (newest), 'YYYY-MM', or a full span;
    include_static: keep the window-independent tables (the decimal references) available."""
    groups = batches(source_dir, **kwargs)
    if not groups:
        # Distinguish "empty folder" from "folder full of PRE-BATCH files", which is what a
        # stage sees before the notebook that writes it has been re-run.
        pattern = kwargs.get("pattern", "*.csv")
        loose = sorted(p.name for p in Path(source_dir).glob("*.csv"))
        extra = (f" — but {len(loose)} CSV(s) sit loose in it ({', '.join(loose[:3])}"
                 f"{'…' if len(loose) > 3 else ''}), i.e. they predate save_batch(): re-run "
                 f"the notebook that writes {source_dir}/ first"
                 if loose and pattern != "*.csv" else "")
        raise FileNotFoundError(f"No batch matching {pattern!r} under {source_dir!r}{extra}")
    span = resolve_batch(batch, groups)
    paths = dict(groups[span])
    if include_static:
        paths.update(static_paths(source_dir, **kwargs))
    return BatchRun(span, paths, source_dir=source_dir)


def output_timelines(out_dir=OUTPUT_DIR):
    """timelines() for a folder save_batch() wrote — the menu a DOWNSTREAM stage picks from."""
    return timelines(out_dir, pattern=OUTPUT_PATTERN)


def use_output_batch(batch=None, out_dir=OUTPUT_DIR):
    """Pin a run to one SAVED batch — the downstream twin of use_batch().
    Same guarantee, one folder further along: feature_addition / validation read only the
    window they pinned. include_static is off because the decimal references belong to
    extraction, not to an export."""
    return use_batch(batch, source_dir=out_dir, include_static=False,
                     pattern=OUTPUT_PATTERN)


# --- normalized batches: <source_dir>/batch_<s>_<e>/, written by normalize_stage.py ---
# normalize no longer rewrites the flat raw files. Each batch gets its own folder of real-unit
# copies; transform and validation read ONLY those, so a batch that skipped normalize fails at
# the pin with a clear message instead of flowing through as raw values.
NORMALIZED_PATTERN = "batch_*/*.csv"    # NOT "*/*.csv": old/ and tmp/ beside it hold CSVs too


def normalized_folder(span, source_dir=SOURCE_DIR):
    """Where one batch's normalized copies live: <source_dir>/batch_<start>_<end>/."""
    return Path(source_dir) / batch_dir_name(span)


def normalized_timelines(source_dir=SOURCE_DIR):
    """timelines() for the normalized batch folders — the menu transform / validation pick from."""
    return timelines(source_dir, pattern=NORMALIZED_PATTERN)


def _folder_mismatches(groups):
    """Files sitting in a batch folder whose own ROWS say a different span. The folder name is a
    label a person can get wrong (a copied file, a renamed folder); the rows are the authority."""
    return sorted(f"{p.parent.name}/{p.name} (rows say {span_label(span)})"
                  for span, paths in groups.items() for p in paths.values()
                  if p.parent.name != batch_dir_name(span))


def use_normalized_batch(batch=None, source_dir=SOURCE_DIR):
    """Pin a run to one batch's NORMALIZED copies — the twin of use_batch() for the stages after
    normalize. The decimal references have no window and no copy, so they come from the flat
    top level exactly as before. Raises if the batch was never normalized, or if any file sits
    in a folder that does not match what its rows say."""
    groups = batches(source_dir, pattern=NORMALIZED_PATTERN)
    if not groups:
        raise FileNotFoundError(
            f"No normalized batch under {str(source_dir)!r} — run normalize.ipynb for this batch "
            f"first. transform and validation read only the normalized copies, never the flat "
            f"raw files.")
    wrong = _folder_mismatches(groups)
    if wrong:
        raise ValueError("files in the wrong batch folder: " + "; ".join(wrong))
    span = resolve_batch(batch, groups)
    paths = dict(groups[span])
    paths.update(static_paths(source_dir))
    return BatchRun(span, paths, source_dir=source_dir)


# --- per-ASSET scaling: raw integer amounts -> real token units ---
# Division goes through Decimal so the big integers stay exact before the float cast.
RAW_SUFFIX = "_raw"


def raw_amount_columns(df, suffix=RAW_SUFFIX):
    """Return the columns holding raw integer amounts (those ending in ``_raw``)."""
    return [c for c in df.columns if c.endswith(suffix)]


def decimals_map(decimals, asset_col="asset", decimals_col="decimals", unit_col="unit",
                 token_unit="raw_token_amount"):
    """Normalize a `decimals` argument into an {asset_address: int} lookup.
    Accepts int (uniform), dict/Series (used directly), or DataFrame (asset + decimals cols).
    Returns: the lookup. Reference tables use only raw_token_amount rows; dupes keep the first."""
    if isinstance(decimals, int):
        return decimals
    if isinstance(decimals, pd.Series):
        return decimals.dropna().astype(int).to_dict()
    if isinstance(decimals, pd.DataFrame):
        ref = decimals
        if unit_col in ref.columns:
            ref = ref[ref[unit_col] == token_unit]
        ref = ref.dropna(subset=[asset_col, decimals_col]).drop_duplicates(asset_col)
        return dict(zip(ref[asset_col], ref[decimals_col].astype(int)))
    return dict(decimals)


def _scaled_name(col, suffix):
    """New column name for a scaled raw column: strip ``_raw``, else append suffix."""
    if col.endswith(RAW_SUFFIX):
        return col[: -len(RAW_SUFFIX)]
    return f"{col}{suffix}"


def _scale_value(value, dec):
    """Divide one value by 10**dec exactly (Decimal -> float).
    value, dec.
    Returns: float, or NaN when either side is missing."""
    if pd.isna(value) or pd.isna(dec):
        return float("nan")
    return float(Decimal(str(value)) / (Decimal(10) ** int(dec)))


def scale_by_decimals(df, decimals, columns=None, asset_col="asset",
                      decimals_col="decimals", drop_raw=False, suffix="_scaled"):
    """Divide raw integer-amount columns by 10**decimals -> real token units.
    decimals: int/dict/Series/DataFrame (see decimals_map); columns defaults to every *_raw;
    asset_col, decimals_col, drop_raw, suffix. Returns: a NEW DataFrame."""

    # NOTE: *_raw columns lose the suffix rather than gain one (supply_amount_raw -> supply_amount)
    cols = columns if columns is not None else raw_amount_columns(df)
    if not cols:
        return df.copy()

    dmap = decimals_map(decimals, asset_col=asset_col, decimals_col=decimals_col)
    out = df.copy()

    if isinstance(dmap, int):
        dec_per_row = pd.Series(dmap, index=df.index)
    else:
        dec_per_row = df[asset_col].map(dmap)

    decs = list(dec_per_row)
    for col in cols:
        out[_scaled_name(col, suffix)] = [
            _scale_value(raw, dec) for raw, dec in zip(df[col], decs)
        ]

    if drop_raw:
        out = out.drop(columns=[c for c in cols if c in out.columns])
    return out


# --- per-COLUMN scaling: fixed decimals from a metric map ---
# Companion to scale_by_decimals. Aave config metrics carry one decimals for the whole
# column (caps -> 0, debt_ceiling -> 2, bps fields -> 4), keyed by `metric` in the reference.

def column_decimals_map(decimals, metric_col="metric", decimals_col="decimals"):
    """Normalize a per-column decimals argument into a {column_name: int} lookup.
    Accepts dict/Series (keys are column names) or DataFrame (metric + decimals cols).
    Returns: the lookup; null rows dropped, duplicate metrics keep the first."""
    if isinstance(decimals, pd.Series):
        return decimals.dropna().astype(int).to_dict()
    if isinstance(decimals, pd.DataFrame):
        ref = decimals.dropna(subset=[metric_col, decimals_col]).drop_duplicates(metric_col)
        return dict(zip(ref[metric_col], ref[decimals_col].astype(int)))
    return dict(decimals)


def scale_columns_by_decimals(df, decimals, columns=None, metric_col="metric",
                              decimals_col="decimals", overwrite=False,
                              drop_original=False, suffix="_scaled"):
    """Divide whole columns by 10**decimals, with a FIXED decimals PER COLUMN.
    decimals (see column_decimals_map), columns, overwrite, drop_original, suffix.
    Returns: a NEW DataFrame; NaN wherever value or decimals is missing."""
    dmap = column_decimals_map(decimals, metric_col=metric_col, decimals_col=decimals_col)
    cols = [c for c in (columns if columns is not None else df.columns)
            if c in dmap and c in df.columns]
    if not cols:
        return df.copy()

    out = df.copy()
    for col in cols:
        dec = dmap[col]
        out[col if overwrite else f"{col}{suffix}"] = [
            _scale_value(v, dec) for v in df[col]
        ]

    if drop_original and not overwrite:
        out = out.drop(columns=[c for c in cols if c in out.columns])
    return out





# --- price multiplication: value each amount in USD and in ETH ---
# Matched on (time_bucket, asset); the two frames store time_bucket in different string
# formats, so both are normalized to a UTC datetime key first. Missing price or amount
# yields NaN, never a silent 0.

def multiply_by_price(df, prices, amount_columns, time_col="time_bucket",
                      asset_col="asset", usd_col="avg_price_usd",
                      eth_col="avg_price_eth", usd_suffix="_value_usd",
                      eth_suffix="_value_eth"):
    """Value each amount column in USD and ETH against an oracle-price table.
    df, prices, amount_columns, time_col, asset_col, usd_suffix, eth_suffix.
    Returns: a NEW DataFrame with two value columns per amount; NaN when either side is absent."""
    out = df.copy()

    # normalized match key on both sides (the tables differ in time_bucket format)
    left_key = pd.to_datetime(df[time_col], utc=True, format="mixed")
    right_key = pd.to_datetime(prices[time_col], utc=True, format="mixed")

    # one price row per (time, asset): usd + eth, null prices stay NaN
    price = pd.DataFrame({
        "_t": right_key,
        "_a": prices[asset_col],
        "_usd": pd.to_numeric(prices[usd_col], errors="coerce"),
        "_eth": pd.to_numeric(prices[eth_col], errors="coerce"),
    }).drop_duplicates(["_t", "_a"])

    # left-join keeps df's row order; an unmatched (time, asset) -> NaN price
    merged = pd.DataFrame({"_t": left_key, "_a": df[asset_col]}).merge(
        price, on=["_t", "_a"], how="left")
    usd = merged["_usd"].to_numpy()
    eth = merged["_eth"].to_numpy()

    for col in amount_columns:
        amount = pd.to_numeric(df[col], errors="coerce").to_numpy()
        out[f"{col}{usd_suffix}"] = amount * usd
        out[f"{col}{eth_suffix}"] = amount * eth
    return out

# --- null repair after the protocol-panel merge (left joins on time_bucket) ---
# Two kinds of null come out of that merge: EVENT columns, where an absent bucket truly
# means zero events (left null they read >70% missing and poison every ratio), and STATE
# columns sampled from getUserAccountData, which land in ~half the buckets and must be
# carried forward under a staleness cap.

def fill_event_zeros(df, columns):
    """Fill nulls with 0 in EVENT columns (structural zeros from left joins).
    df, columns -> counts, unique-user counts, valued sums only.
    Returns: a NEW DataFrame. Never use on state or ratio columns."""
    out = df.copy()
    cols = [c for c in columns if c in out.columns]
    out[cols] = out[cols].fillna(0)
    return out


def ffill_state_columns(df, columns, time_col="time_bucket", limit=None,
                        observed_col=None, age_col=None):
    """Forward-fill sampled STATE columns along the time axis.
    columns (filled together), limit (staleness cap), observed_col, age_col.
    Returns: a NEW DataFrame in the original row order."""
    order = pd.to_datetime(df[time_col], utc=True, format="mixed").sort_values().index
    out = df.copy()
    cols = [c for c in columns if c in out.columns]

    block = out.loc[order, cols]           # time-ordered view of the state columns
    observed = block.notna().any(axis=1)   # bucket had its own observation
    out.loc[order, cols] = block.ffill(limit=limit)

    # metadata columns assign back by index, so they land on the right rows
    if observed_col:
        out[observed_col] = observed
    if age_col:
        # buckets since the last observed row: position - position of last observation
        pos = pd.Series(range(len(order)), index=order, dtype="float64")
        out[age_col] = pos - pos.where(observed).ffill()
    return out


def aggregate_by_time_bucket(df, time_col, group_cols, agg_func='sum', freq=None):
    """Sum/mean `group_cols` per `time_col`; with `freq`, roll up to a coarser grain.

    Every stamp is an END stamp (12:00 = 10:00→12:00, E67), so a coarse bucket is closed on the
    RIGHT and labelled by its right edge: a `24h` row stamped D 00:00 holds D-1 00:00 < t <= D 00:00,
    i.e. exactly the 12 fine buckets of day D-1. pandas' default (closed/label left) would push the
    22–24h bucket, stamped D 00:00, into the NEXT day."""
    if isinstance(agg_func, str):
        agg_dict = {col: agg_func for col in group_cols}
    else:
        agg_dict = agg_func

    if freq is None:
        result = df.groupby(time_col, as_index=False)[group_cols].agg(agg_dict)
    else:
        df = df.copy()
        # Remove " UTC" suffix if present, then parse
        df[time_col] = pd.to_datetime(
            df[time_col].astype(str).str.replace(' UTC', '', regex=False)
        )
        result = (
            df.set_index(time_col)
              .resample(freq, closed="right", label="right")[group_cols]
              .agg(agg_dict)
              .reset_index()
        )

    return result
