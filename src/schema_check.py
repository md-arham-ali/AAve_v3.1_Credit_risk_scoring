"""Schema-conformance checks — does this table still have the columns we agreed on?

The pipeline's other gates check VALUES (nulls, ranges, invariants). This one checks
STRUCTURE, and it exists because structure drift has already happened here for real: a Dune
query id changed under `supply_withdraw` and the pipeline re-ran against the wrong window
without a single error. A renamed column, a dropped one, an unexpected new one, or a dtype
flip nothing accounts for should stop the run AT the notebook that produced it — not three
stages later where the failure is far from its cause.

The reference is hand-authored, one `schemas/<notebook>_schema.py` per stage. This module is
only the comparison, so it stays generic and lives in `src/` with everything else shared.

Two deliberate non-checks:
  order        nothing in this codebase reads columns positionally, so checking order would
               only ever produce false positives.
  nullability  no separate field: the dtype already carries it. A column that is float64
               because of upstream NaNs (a left join, or a genuinely always-null column like
               collateral_toggle.asset_symbol) IS the nullability signal. A column that
               legitimately differs across windows gets a tuple of accepted dtypes.

Failures are bare asserts, the same idiom as every other hard gate here (validation.ipynb's
`assert _gate.empty`, transform.ipynb's export asserts, data_validation's TABLE_LABELS
self-check). papermill under `set -e` and orchestration's `subprocess.run(check=True)`
already turn an AssertionError into a failed script / a red Dagster asset, so "the run stops"
needs nothing new wired.
"""

import json
from pathlib import Path

import pandas as pd


def dtype_signature(df):
    """{column: dtype_str} for a live DataFrame — the shape a schema is compared against."""
    return {str(c): str(df[c].dtype) for c in df.columns}


def _accepted(expected):
    """Normalize one schema entry's value to the tuple of dtypes it accepts."""
    return (expected,) if isinstance(expected, str) else tuple(expected)


def _columns_of(schema):
    """The {column: dtype} map out of a schema entry, with or without the 'columns' wrapper."""
    return schema.get("columns", schema) if isinstance(schema, dict) else schema


def check_columns(actual, schema, *, table, stage):
    """Compare a {column: dtype_str} signature against one table's schema entry.
    actual: from dtype_signature() or a manifest; schema: {"columns": {col: dtype|tuple}}.
    Hard-asserts on any mismatch — missing, unexpected, or wrong dtype.
    Returns: the actual signature, so a call can be inlined."""
    expected = _columns_of(schema)

    missing = [c for c in expected if c not in actual]
    unexpected = [c for c in actual if c not in expected]
    # Only columns present on both sides can disagree on dtype; the other two lists own the rest.
    wrong = [f"{c}: {actual[c]} not in {_accepted(expected[c])}"
             for c in expected
             if c in actual and actual[c] not in _accepted(expected[c])]

    detail = "; ".join(part for part in (
        f"missing {len(missing)} ({', '.join(missing)})" if missing else "",
        f"unexpected {len(unexpected)} ({', '.join(unexpected)})" if unexpected else "",
        f"dtype {len(wrong)} ({'; '.join(wrong)})" if wrong else "",
    ) if part)

    assert not (missing or unexpected or wrong), (
        f"[{stage}] '{table}' does not match its schema — {detail}. "
        f"Either the upstream table changed (investigate) or the change is intended "
        f"(update schemas/{stage.split(':')[0]}_schema.py)."
    )
    return actual


def check_frame(df, schema, *, table, stage):
    """check_columns() against one live DataFrame's own dtypes."""
    return check_columns(dtype_signature(df), schema, table=table, stage=stage)


def check_frames(frames, schemas, *, stage, verbose=True):
    """Check every {table: DataFrame} pair against `schemas`.
    A frame whose table is absent from `schemas` FAILS — an unregistered table is exactly the
    silent addition this gate exists to catch. The reverse is allowed: a schema with no frame
    just means this run did not produce that table (a partial extraction, one Dune group).
    Returns: {table: signature}."""
    unregistered = [t for t in frames if t not in schemas]
    assert not unregistered, (
        f"[{stage}] {len(unregistered)} table(s) have no schema entry: "
        f"{', '.join(sorted(unregistered))}. Add them to the stage's schema file, or they "
        f"flow through unchecked."
    )

    signatures = {table: check_frame(df, schemas[table], table=table, stage=stage)
                  for table, df in frames.items()}
    if verbose:
        skipped = sorted(set(schemas) - set(frames))
        note = f"; {len(skipped)} not produced by this run ({', '.join(skipped)})" if skipped else ""
        print(f"[{stage}] schema OK: {len(signatures)}/{len(schemas)} tables match{note}")
    return signatures


def check_manifest(manifest_path, schemas, *, stage, tables=None):
    """Check a saved `_manifest.json`'s captured columns/dtypes without reloading the CSVs.
    An audit path, not a gate: use it on an OLD batch folder after a schema changes, to see
    what that batch would have failed on. Manifests written before the columns/dtypes
    extension carry neither and are reported as unusable rather than silently passing."""
    manifest = json.loads(Path(manifest_path).read_text())
    entries = {e["table"]: e for e in manifest.get("tables", [])
               if tables is None or e["table"] in tables}

    stale = [t for t, e in entries.items() if "dtypes" not in e]
    assert not stale, (
        f"[{stage}] {manifest_path}: {len(stale)} table(s) predate the columns/dtypes "
        f"manifest extension ({', '.join(sorted(stale))}) — re-run the stage that wrote it."
    )
    unregistered = [t for t in entries if t not in schemas]
    assert not unregistered, (f"[{stage}] {manifest_path}: no schema entry for "
                              f"{', '.join(sorted(unregistered))}")

    return {table: check_columns(entry["dtypes"], schemas[table], table=table, stage=stage)
            for table, entry in entries.items()}


def assert_schema_file(schemas, *, name):
    """Self-check for a schemas/*.py module, run at ITS import time.
    Mirrors data_validation's TABLE_LABELS assert: a typo'd entry fails when the file is
    imported, not at the first gate that happens to touch that table."""
    assert isinstance(schemas, dict) and schemas, f"{name}: SCHEMAS must be a non-empty dict"
    for table, entry in schemas.items():
        assert isinstance(table, str), f"{name}: table name {table!r} is not a string"
        columns = _columns_of(entry)
        assert isinstance(columns, dict) and columns, \
            f"{name}: '{table}' has no columns map"
        for column, dtype in columns.items():
            assert isinstance(column, str), f"{name}: '{table}' column {column!r} is not a string"
            assert isinstance(dtype, str) or (
                isinstance(dtype, tuple) and dtype
                and all(isinstance(d, str) for d in dtype)
            ), f"{name}: '{table}.{column}' dtype must be a str or a non-empty tuple of str, got {dtype!r}"
    return schemas
