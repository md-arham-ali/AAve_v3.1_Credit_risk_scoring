"""Reusable data-validation checks for the Aave V3.1 Dune result tables.

Flow per table:
  1. pandas checks first  -> null counts, dtypes, duplicates  
  2. Great Expectations   -> structural expectations           (tp be run after step 1)

The checks are column-aware: each expectation is only added when the columns it
needs are present, so the SAME functions work across the different tables.
Libraries imported:
    pandas              -> dataframes + the basic null/type/duplicate checks
    great_expectations  -> the expectation suite 
    re, datetime, pathlib -> standard library
"""

import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Friendly result names per Dune query id (falls back to query_<id> otherwise).
TABLE_LABELS = {
    "7702138": "supply_withdraw",
    "7702164": "borrow_repay",
    "7711042": "reserve_state_rates",
    "7711190": "reserve_config",
    "7711212": "liquidation",
    "7711227": "flashloan",
    "7711236": "user_account",
    "7711248": "collateral_toggle",
    "7714873": "oracle_price_usd_eth_weth_6h",
}

# Columns that, when present, are the table's address-style keys, sometimes unique for tx hashes, though not necessary for this project.
ADDRESS_COLS = ("asset", "collateral_asset", "debt_asset")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def table_name_from_path(path):
    """Derive a short, table-specific name from a query_result_data_<id>_<ts>.csv file."""
    stem = Path(path).stem
    match = re.search(r"_(\d+)_", stem)
    qid = match.group(1) if match else stem
    return TABLE_LABELS.get(qid, f"query_{qid}")


def load_csv(path):
    """Read a result CSV. Values are left as-loaded (time_bucket/asset stay strings)."""
    return pd.read_csv(path)


def key_columns(df):
    """Best-effort composite key used for the uniqueness check."""
    cols = set(df.columns)
    if "time_bucket" in cols:
        if "asset" in cols:
            return ["time_bucket", "asset"]
        if "collateral_asset" in cols and "debt_asset" in cols:
            return ["time_bucket", "collateral_asset", "debt_asset"]
        return ["time_bucket"]
    if "metric" in cols:  # decimal-reference tables
        return [c for c in ("asset", "metric") if c in cols]
    return []


def required_columns(df):
    """Columns that must never be null (the key columns that are always populated)."""
    cols = set(df.columns)
    if "time_bucket" in cols:
        return ["time_bucket"] + [c for c in ADDRESS_COLS if c in cols]
    if "metric" in cols:  # decimal-reference asset is intentionally nullable
        return ["metric"]
    return []


def _expected_kind(col):
    """Expected coarse type for the pandas type check ('string' / 'numeric' / 'any')."""
    c = col.lower()
    if c in ("time_bucket", "metric", "unit") or c.endswith("symbol") \
            or c == "asset" or c.endswith("_asset"):
        return "string"
    if c.endswith("_count") or c.endswith("_block") or c == "decimals":
        return "numeric"
    # big-int amounts/rates/indexes: precision-sensitive -> report dtype, don't assert
    return "any"


# --------------------------------------------------------------------------- #
# Step 1 — pandas basic checks (run first)
# --------------------------------------------------------------------------- #
def pandas_report(df, table_name):
    """Return (summary_df, null_counts_df) — the basic NULL/duplicate report."""
    n_rows, n_cols = df.shape
    key = key_columns(df)

    null_counts = pd.DataFrame([
        {
            "table": table_name,
            "column": c,
            "dtype": str(df[c].dtype),
            "n_null": int(df[c].isna().sum()),
            "null_pct": round(100 * df[c].isna().mean(), 4) if n_rows else 0.0,
            "n_unique": int(df[c].nunique(dropna=True)),
        }
        for c in df.columns
    ])

    summary = pd.DataFrame([{
        "table": table_name,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "key_columns": ",".join(key),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "n_duplicate_keys": int(df.duplicated(subset=key).sum()) if key else None,
        "n_total_null_cells": int(df.isna().sum().sum()),
    }])
    return summary, null_counts


def type_report(df, table_name):
    """Lightweight per-column type check (numeric vs string); big-int cols are not asserted."""
    rows = []
    for c in df.columns:
        kind = _expected_kind(c)
        is_num = pd.api.types.is_numeric_dtype(df[c])
        if kind == "numeric":
            type_ok = bool(is_num)
        elif kind == "string":
            type_ok = bool(not is_num)
        else:
            type_ok = None  # not asserted
        rows.append({
            "table": table_name, "column": c, "dtype": str(df[c].dtype),
            "expected_kind": kind, "type_ok": type_ok,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Step 2 — Great Expectations checks (run after step 1)
# --------------------------------------------------------------------------- #
def _add_expectations(gdf, df):
    """Register the column-aware basic expectations on a GE PandasDataset (GE 0.18 API)."""
    cols = set(df.columns)

    gdf.expect_table_row_count_to_be_between(min_value=1)         # table not empty

    for c in required_columns(df):                               # key columns not null
        gdf.expect_column_values_to_not_be_null(c)

    key = key_columns(df)                                        # key is unique (no dup PKs)
    if len(key) == 1:
        gdf.expect_column_values_to_be_unique(key[0])
    elif len(key) > 1:
        gdf.expect_compound_columns_to_be_unique(key)

    if "time_bucket" in cols:                                    # format + window + 6h grid
        gdf.expect_column_values_to_match_regex(
            "time_bucket", r"^(2025-1[12]|2026-01)-\d{2} (00|06|12|18):00:00")

    for c in ADDRESS_COLS:                                       # 20-byte hex address shape
        if c in cols:
            gdf.expect_column_values_to_match_regex(c, r"^0x[0-9a-f]{40}$")

    for c in df.columns:                                        # counts / blocks non-negative
        if c.endswith("_count") or c.endswith("_block"):
            gdf.expect_column_values_to_be_between(c, min_value=0)

    if "decimals" in cols:                                       # sane token decimals
        gdf.expect_column_values_to_be_between("decimals", min_value=0, max_value=36)


def run_great_expectations(df, table_name):
    """Run the basic expectation suite (GE 0.18) and return a tidy results table."""
    import great_expectations as ge

    gdf = ge.from_pandas(df)                  # wrap the dataframe as a GE dataset
    _add_expectations(gdf, df)                # register only the applicable checks
    report = gdf.validate(result_format="SUMMARY", catch_exceptions=True)

    rows = []
    for r in report.results:
        cfg = r.expectation_config
        detail = r.result or {}
        info = r.exception_info or {}
        col = cfg.kwargs.get("column") or cfg.kwargs.get("column_list", "")
        if isinstance(col, (list, tuple)):
            col = ", ".join(col)  # compound-key checks -> readable "a, b"
        rows.append({
            "table": table_name,
            "expectation": cfg.expectation_type,
            "column": col,
            "success": bool(r.success),
            "element_count": detail.get("element_count"),
            "unexpected_count": detail.get("unexpected_count"),
            "unexpected_percent": detail.get("unexpected_percent"),
            "exception": info.get("exception_message") or "",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Readable Markdown report
# --------------------------------------------------------------------------- #
def _df_to_md(df):
    """Render a DataFrame as a GitHub-flavored Markdown table (no extra dependencies)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        cells = []
        for v in row:
            try:
                blank = bool(pd.isna(v))
            except (TypeError, ValueError):
                blank = False  # lists / non-scalars are never "NA"
            text = "" if blank else str(v)
            cells.append(text.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _table_block(table_name, summary, null_counts, types, ge_results):
    """Markdown section for ONE table (h2 + h3 subsections), used inside the combined report."""
    failed = ge_results[ge_results["success"] == False]  # noqa: E712
    status = "PASS ✅" if failed.empty else f"FAIL ❌ ({len(failed)} expectation(s) failed)"

    summary_kv = summary.drop(columns=["table"], errors="ignore").T.reset_index()
    summary_kv.columns = ["field", "value"]

    def no_table(frame):
        return frame.drop(columns=["table"], errors="ignore")

    parts = [
        f"## {table_name}",
        "",
        f"**Overall:** {status}",
        "",
        "### Summary", "", _df_to_md(summary_kv), "",
        "### Null counts (per column)", "", _df_to_md(no_table(null_counts)), "",
        "### Type checks", "", _df_to_md(no_table(types)), "",
        "### Great Expectations", "", _df_to_md(no_table(ge_results)), "",
    ]
    if not failed.empty:
        parts += ["### ⚠️ Failed expectations", "", _df_to_md(no_table(failed)), ""]
    parts += ["---", ""]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Orchestration — pandas first, then GE
# --------------------------------------------------------------------------- #
def validate_table(df, table_name):
    """Run all checks for one table; return the frames + its Markdown block (no file write)."""
    summary, null_counts = pandas_report(df, table_name)   # step 1 (pandas)
    types = type_report(df, table_name)
    ge_results = run_great_expectations(df, table_name)     # step 2 (great expectations)
    return {
        "table": table_name,
        "summary": summary,
        "null_counts": null_counts,
        "type_checks": types,
        "ge_results": ge_results,
        "block": _table_block(table_name, summary, null_counts, types, ge_results),
    }


def validate_batch(paths, results_dir="validation_results", report_name="validation_report.md"):
    """Validate every table in `paths` and save ONE combined Markdown report.

    Returns {report_path, overview (DataFrame), results (list of validate_table dicts)}.
    """
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)

    results, overview, blocks = [], [], []
    for path in paths:
        name = table_name_from_path(path)
        res = validate_table(load_csv(path), name)
        failed = res["ge_results"][res["ge_results"]["success"] == False]  # noqa: E712
        overview.append({
            "table": name,
            "rows": int(res["summary"].iloc[0]["n_rows"]),
            "overall": "PASS ✅" if failed.empty else f"FAIL ❌ ({len(failed)})",
        })
        blocks.append(res["block"])
        results.append(res)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    header = "\n".join([
        "# Validation report",
        "",
        f"- **Generated:** {generated}",
        f"- **Tables:** {len(results)}",
        "",
        "## Overview",
        "",
        _df_to_md(pd.DataFrame(overview)),
        "",
        "---",
        "",
    ])
    report_path = out / report_name
    report_path.write_text(header + "\n" + "\n".join(blocks), encoding="utf-8")
    return {"report_path": report_path, "overview": pd.DataFrame(overview), "results": results}

