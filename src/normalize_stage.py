"""The normalize STAGE — one raw batch in, one normalized batch folder out. Raw is never rewritten.

    query_result_data/<table>_<id>_<window>.csv     flat raw staging: written by process, then left alone
    query_result_data/batch_<s>_<e>/<same names>    the batch in real units — what transform and validation read
    query_result_data/batch_<s>_<e>/_manifest.json  per-table state + the source file's size / mtime / sha256

normalize.ipynb used to rescale its input IN PLACE: a crash mid-write left a half-scaled file, and
only a magnitude guess (`looks_raw_*`) stood between a re-run and a double scale. Now the batch
folder is rebuilt from the untouched raw files every time, in a hidden staging folder that is
swapped in whole — so it is always either the old complete batch or the new complete batch, and a
crash leaves no half-built folder to be mistaken for one. The notebook calls normalize_batch(run).

The scaling itself lives in normalization.py, unchanged. This module only decides WHICH tables
are scaled, builds the folder, and checks the result.
"""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import normalization as nz
import schema_check as sc
import transform as tf
from normalize_schema import SCHEMAS as NORMALIZE_SCHEMAS

MANIFEST_NAME = tf.MANIFEST_NAME
ORACLE = "oracle_price_usd_eth_weth_6h"      # supplies token decimals — the SAME batch's oracle
CONFIG_REF = "decimal_reference_part4"       # bps / 2dp decimals for reserve_config

# The tables that are rescaled: label -> kind of scaling. Every other time-series table in a batch
# is copied as it came — transform scales those itself (flashloan, supply_withdraw, liquidation …).
# `flashloan` is NOT here: the old notebook row named columns that do not exist, so it was always
# reported "already normalized" and skipped (bugs.md 7b.1). It is dropped rather than carried over.
RULES = {
    "reserve_config":      "bps",
    "reserve_state_rates": "ray",
    "borrow_repay":        "token",
}
TOKEN_COLUMNS = {"borrow_repay": ["borrow_amount", "repay_amount", "net_debt_flow"]}
GUARD_COLUMN = {"bps": "ltv", "ray": "liquidity_rate", "token": "borrow_amount"}
INDEX_COLUMNS = ("liquidity_index", "variable_borrow_index")


def _looks_raw(kind, label, df):
    if kind == "bps":
        return nz.looks_raw_bps(df["ltv"])
    if kind == "ray":
        return nz.looks_raw_ray(df["liquidity_rate"])
    return nz.looks_raw_token_amounts(df, TOKEN_COLUMNS[label])


def _apply(kind, label, df, config_reference, token_decimals):
    if kind == "bps":
        return nz.normalize_reserve_config(df, config_reference)
    if kind == "ray":
        return nz.normalize_reserve_state(df)
    return nz.normalize_amounts_by_token_decimals(df, token_decimals, TOKEN_COLUMNS[label])


def _assert_in_real_units(kind, label, df):
    """Cheap structural checks on what is about to be written — not a full value audit.
    Under-scaled: the table must no longer look raw. Over-scaled: an Aave index starts at 1 RAY
    and only grows, so a scaled index below 1 means it was divided twice."""
    assert not _looks_raw(kind, label, df), f"[normalize] {label} still looks raw after normalization"
    if kind == "ray":
        for col in INDEX_COLUMNS:
            low = pd.to_numeric(df[col], errors="coerce").min()
            assert low >= 1 - 1e-9, (f"[normalize] {label}.{col} minimum is {low:.6g}, below 1.0 — "
                                     f"an index is never below 1; scaled twice?")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _peak(df, kind):
    return float(pd.to_numeric(df[GUARD_COLUMN[kind]], errors="coerce").abs().max())


def normalize_batch(run, *, verbose=True):
    """Build this batch's normalized folder from its raw files.

    run: a RAW BatchRun — tf.use_batch(BATCH, source_dir="query_result_data").
    Per table: a rescaled table whose raw file still looks raw is scaled; one that does not look
    raw is copied unchanged and marked `already_normalized` (a batch whose flat files were rescaled
    in place, before this stage stopped doing that); everything else is copied. Returns a
    DataFrame, one row per table: table, state, source, and the guard column's max before/after."""
    span, source_dir = run.span, Path(run.source_dir)
    final = tf.normalized_folder(span, source_dir)
    staging = source_dir / f".building_{final.name}"
    static = set(tf.static_paths(str(source_dir)))
    tables = sorted(label for label in run.paths if label not in static)

    # Schema gate on the way IN, as before. No gate on the way out: scaling renames nothing, and
    # process_schema accepts both the raw and the rescaled dtype for the columns it touches.
    reference = run.load(CONFIG_REF)
    sc.check_frame(reference, NORMALIZE_SCHEMAS[CONFIG_REF], table=CONFIG_REF, stage="normalize")
    oracle = run.load(ORACLE)
    sc.check_frame(oracle, NORMALIZE_SCHEMAS[ORACLE], table=ORACLE, stage="normalize")
    token_decimals = dict(zip(oracle["asset"], oracle["decimals"]))

    if staging.exists():                       # left by a crashed run — derived data, safe to drop
        shutil.rmtree(staging)
    staging.mkdir()
    report, entries = [], []
    try:
        for label in tables:
            source = run.paths[label]
            target = staging / source.name
            state, before, after = "copied", None, None
            if label in RULES:
                kind = RULES[label]
                df = run.load(label)
                sc.check_frame(df, NORMALIZE_SCHEMAS[label], table=label, stage="normalize")
                if _looks_raw(kind, label, df):
                    before = _peak(df, kind)
                    df = _apply(kind, label, df, reference, token_decimals)
                    _assert_in_real_units(kind, label, df)
                    after = _peak(df, kind)
                    df.to_csv(target, index=False)
                    state = "normalized"
                else:
                    _assert_in_real_units(kind, label, df)
                    shutil.copyfile(source, target)       # not copy2: the copy keeps its own mtime
                    state = "already_normalized"
            else:
                shutil.copyfile(source, target)
            stat = source.stat()
            entries.append({"table": label, "file": source.name, "state": state,
                            "source_size": stat.st_size, "source_mtime": stat.st_mtime,
                            "source_sha256": _sha256(source)})
            report.append({"table": label, "state": state, "source": source.name,
                           "max_before": before, "max_after": after})
            if verbose:
                note = f"   max {before:.4g} -> {after:.4g}" if before is not None else ""
                print(f"{source.name:62s} {state}{note}")
        (staging / MANIFEST_NAME).write_text(json.dumps({
            "batch": list(span),
            "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tables": entries,
        }, indent=2))
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)    # nothing half-built is left behind
        raise

    # Swap the finished folder in whole. The old one is renamed away first, so at no point is
    # there a folder holding a mix of old and new files.
    if final.exists():
        retired = source_dir / f".retired_{final.name}"
        if retired.exists():
            shutil.rmtree(retired)
        final.rename(retired)
        staging.rename(final)
        shutil.rmtree(retired)
    else:
        staging.rename(final)
    if verbose:
        print(f"-> {final}/  ({len(tables)} tables, raw files untouched)")
    return pd.DataFrame(report)


# ══ TEMPORARY_END_STAMP_SHIFT ═════════════════════════════════════════════════════════════════
# The SQL stamps every bucket at its START (12:00 = 12:00→14:00); the project wants END stamps
# (12:00 = 10:00→12:00). Until the 9 queries emit end stamps, the normalized copies are shifted
# forward one bucket here. Raw files are never touched. Delete this section (and its notebook
# cell) when the SQL is fixed.
END_STAMP_MARKER = ".end_stamped"


def _restamp(col, hours):
    """Shift a time_bucket column by `hours`, written back in the table's own text format
    (`… .000 UTC` on 7 tables, bare on oracle + reserve_config — B.3a D-7)."""
    sample = str(col.dropna().iloc[0])
    out = (pd.to_datetime(col, utc=True) + pd.Timedelta(hours=hours)).dt.strftime("%Y-%m-%d %H:%M:%S")
    if "." in sample:
        out += "." + "0" * len(sample.split(".")[1].replace(" UTC", ""))
    if sample.endswith(" UTC"):
        out += " UTC"
    return out


def shift_to_end_stamps(run, bucket_hours, *, verbose=True):
    """Shift time_bucket +bucket_hours in every CSV of run's normalized folder, once.

    Run right after normalize_batch(run) — it rebuilds the folder, which also clears the marker.
    A row whose new stamp falls past the batch's last month is dropped: its end stamp is the next
    month's 00:00, and month_span would then file the table under a different batch, which
    use_normalized_batch refuses. Only supply_withdraw reaches that far (D-1), and transform trims
    it at the oracle cutoff anyway, so nothing downstream loses a row it kept before."""
    folder = tf.normalized_folder(run.span, run.source_dir)
    marker = folder / END_STAMP_MARKER
    assert folder.is_dir(), f"{folder} missing — run normalize_batch(run) first"
    assert not marker.exists(), "already end-stamped; re-run normalize_batch(run) to rebuild first"
    window_end = pd.Period(run.span[1], "M").end_time.tz_localize("UTC")

    rows = []
    for path in sorted(folder.glob("*.csv")):
        df = pd.read_csv(path, dtype=str, keep_default_na=False)   # strings: other columns byte-identical
        if "time_bucket" not in df.columns:
            rows.append((path.name, "skipped (no time_bucket)", None, None, 0))
            continue
        old_min = pd.to_datetime(df["time_bucket"], utc=True).min()
        df["time_bucket"] = _restamp(df["time_bucket"], bucket_hours)
        new_ts = pd.to_datetime(df["time_bucket"], utc=True)
        keep = new_ts <= window_end
        df = df.loc[keep]
        assert new_ts[keep].min() - old_min == pd.Timedelta(hours=bucket_hours), path.name
        df.to_csv(path, index=False)
        rows.append((path.name, "shifted", str(old_min), str(new_ts[keep].min()), int((~keep).sum())))

    marker.write_text(f"time_bucket shifted +{bucket_hours}h (TEMPORARY_END_STAMP_SHIFT)\n")
    report = pd.DataFrame(rows, columns=["file", "action", "first_old", "first_new", "rows_dropped_past_window"])
    if verbose:
        print(f"-> {folder}/  end-stamped (+{bucket_hours}h), marker {END_STAMP_MARKER}")
    return report
