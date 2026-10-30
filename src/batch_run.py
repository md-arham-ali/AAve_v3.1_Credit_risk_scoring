"""Batch runner — one batch through normalize -> transform -> feature_addition -> validation.

Plumbing only. The notebooks keep their config and call sites; this module decides nothing
about HOW a batch is processed. It pins one batch, checks the run can work BEFORE spending
minutes on it, runs the four notebooks in order, stops at the first failure and leaves a record
of what happened. Dagster (§4) calls run_batch(); a person can call it from a terminal:

    python src/batch_run.py status
    python src/batch_run.py run 2025-04 --dry-run          # preflight only, runs nothing
    python src/batch_run.py run 2025-04
    python src/batch_run.py run 2024-04:2025-03 --stages transform feature_addition

Extraction (process.ipynb) is deliberately not a stage: it spends Dune credits.
"""
import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "schemas", ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import transform as tf                                              # noqa: E402
from process_schema import SCHEMAS as PROCESS_SCHEMAS               # noqa: E402
from transform_schema import SCHEMAS as TRANSFORM_SCHEMAS           # noqa: E402
from feature_addition_schema import SCHEMAS as FEATURE_SCHEMAS      # noqa: E402

# Absolute paths throughout: nothing here depends on the caller's working directory.
RAW_DIR = ROOT / "query_result_data"
TRANSFORMED_DIR = ROOT / "transformed_data"
PROCESSED_DIR = ROOT / "processed_data"
VALIDATION_DIR = ROOT / "validation_results"
RUNS_DIR = ROOT / "batch_runs"
NOTEBOOKS = ROOT / "notebooks"

KERNEL = "aave-main"          # the notebooks' own metadata says python3, which resolves badly
STAGES = ("normalize", "transform", "feature_addition", "validation")
PARAM_TAG, INJECTED_TAG, PARAM_NAME = "parameters", "injected-parameters", "BATCH"

# What each stage must WRITE. TRANSFORM_OUTPUTS mirrors the `_out_frames` dict in
# transform.ipynb's export cell — rename a frame there and the post-check below fails naming it,
# which is the prompt to update this tuple. The other two come from the schema files, so they
# cannot drift: FEATURE_OUTPUTS is feature_addition's output gate, PROCESSED_OUTPUTS is the rest
# of transform_schema (the per-table intermediates saved under processed_data/).
TRANSFORM_OUTPUTS = ("DF_common_final_1", "DF_common_1", "df_liq_2h", "df_liq_24h", "df_liq_7d",
                     "df_user_account_2h", "df_user_account_24h", "df_user_account_7d")
FEATURE_OUTPUTS = tuple(FEATURE_SCHEMAS)
PROCESSED_OUTPUTS = tuple(k for k in TRANSFORM_SCHEMAS if k not in TRANSFORM_OUTPUTS)


# What each stage must find on disk before it starts, and what it leaves for the next one. A need
# is satisfied either by an earlier stage in the same run or by files already there — so a run can
# start mid-chain, but never on top of something that is not there.
NEEDS = {"normalize": (), "transform": ("normalized",), "feature_addition": ("transform",),
         "validation": ("normalized", "transform")}
PRODUCES = {"normalize": "normalized", "transform": "transform",
            "feature_addition": "features", "validation": "validation"}


class BatchRunError(RuntimeError):
    """A batch run that cannot start or did not finish. The run record says why."""


# --------------------------------------------------------------------------- notebook shape
def _batch_uses(source):
    """Line numbers where `BATCH` is READ in a cell's code. Comments and strings do not count —
    parsed with ast; a cell with IPython magics will not parse, so it falls back to a regex over
    comment-stripped lines."""
    try:
        return [n.lineno for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.Name) and n.id == PARAM_NAME and isinstance(n.ctx, ast.Load)]
    except SyntaxError:
        return [i for i, line in enumerate(source.splitlines(), 1)
                if re.search(rf"\b{PARAM_NAME}\b", line.split("#", 1)[0])]


def parameters_cell_problems(path):
    """Why this notebook cannot take BATCH from papermill — [] when it can.

    Papermill injects the value in a NEW cell placed right after the cell tagged `parameters`.
    So the tagged cell must hold only `BATCH = ...`; if it also pins the run
    (`run = tf.use_batch(BATCH)`) the pin is made before the value arrives, and the notebook
    silently processes the default. Tested: tag-only cell then use cell -> value arrives; tag +
    use in one cell -> default. A missing tag is only a log warning and exit 0."""
    name = Path(path).name
    cells = json.loads(Path(path).read_text())["cells"]
    code = [(i, "".join(c["source"]), c.get("metadata", {}).get("tags", []))
            for i, c in enumerate(cells) if c["cell_type"] == "code"]
    tagged = [i for i, _, tags in code if PARAM_TAG in tags]
    if not tagged:
        return [f"{name}: no cell is tagged '{PARAM_TAG}'"]
    if len(tagged) > 1:
        return [f"{name}: {len(tagged)} cells are tagged '{PARAM_TAG}' (cells {tagged}) — keep one"]

    problems, tag_i, used_after = [], tagged[0], False
    for i, source, _ in code:
        if i == tag_i:
            try:
                body = ast.parse(source).body
            except SyntaxError as exc:
                problems.append(f"{name}: parameters cell {i} does not parse ({exc.msg})")
                continue
            assigns = [s for s in body if isinstance(s, ast.Assign) and len(s.targets) == 1
                       and isinstance(s.targets[0], ast.Name) and s.targets[0].id == PARAM_NAME]
            if not assigns:
                problems.append(f"{name}: parameters cell {i} does not assign {PARAM_NAME}")
            for stmt in (s for s in body if s not in assigns):
                first = ast.get_source_segment(source, stmt).splitlines()[0][:70]
                problems.append(f"{name}: parameters cell {i}, line {stmt.lineno} is more than "
                                f"`{PARAM_NAME} = ...` — move it to the next cell: {first}")
        elif _batch_uses(source):
            if i < tag_i:
                problems.append(f"{name}: cell {i} reads {PARAM_NAME} BEFORE the parameters cell "
                                f"(line {_batch_uses(source)[0]})")
            else:
                used_after = True
    if not used_after:
        problems.append(f"{name}: no cell after the parameters cell reads {PARAM_NAME}")
    return problems


def _injected_value(executed_path):
    """The BATCH value papermill actually injected into an executed copy, or None."""
    for cell in json.loads(Path(executed_path).read_text())["cells"]:
        if INJECTED_TAG in cell.get("metadata", {}).get("tags", []):
            for stmt in ast.parse("".join(cell["source"])).body:
                if (isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name)
                        and stmt.targets[0].id == PARAM_NAME):
                    return ast.literal_eval(stmt.value)
    return None


# --------------------------------------------------------------------------- what is on disk
def _labels_in(folder):
    """{label: mtime} for the CSVs of one batch folder — labels resolved the pipeline's way."""
    folder = Path(folder)
    if not folder.is_dir():
        return {}
    return {tf.label_for(p)[0]: p.stat().st_mtime for p in folder.glob("*.csv")}


def _manifest_tables(folder):
    path = Path(folder) / tf.MANIFEST_NAME
    if not path.exists():
        return set()
    try:
        return {t["table"] for t in json.loads(path.read_text()).get("tables", [])}
    except ValueError:
        return set()


def _missing(folder, labels):
    got = _labels_in(folder)
    return [lb for lb in labels if lb not in got]


def _raw_tables():
    """The time-series labels a batch must hold — every table in process_schema that is not a
    window-independent decimal reference."""
    return sorted(set(PROCESS_SCHEMAS) - set(tf.static_paths(str(RAW_DIR))))


def _normalized_manifest(span):
    path = tf.normalized_folder(span, RAW_DIR) / tf.MANIFEST_NAME
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except ValueError:
        return None


def _normalized_stale(span):
    """Raw files that changed (re-fetched) after this batch was normalized: [] when none. Size and
    mtime are the cheap test; the manifest also holds a sha256 for a stricter audit."""
    manifest = _normalized_manifest(span)
    if manifest is None:
        return []
    changed = []
    for entry in manifest.get("tables", []):
        source = RAW_DIR / entry["file"]
        if not source.exists():
            changed.append(f"{entry['file']} (gone)")
            continue
        stat = source.stat()
        if stat.st_size != entry["source_size"] or abs(stat.st_mtime - entry["source_mtime"]) > 1:
            changed.append(entry["file"])
    return changed


def _absent(need, span):
    """What is missing on disk for one prerequisite of a stage — [] when it is all there."""
    folder = tf.batch_dir_name(span)
    if need == "normalized":
        missing = _missing(RAW_DIR / folder, _raw_tables())
        return missing or [f"{f} changed since normalizing" for f in _normalized_stale(span)]
    if need == "transform":
        return _missing(TRANSFORMED_DIR / folder, TRANSFORM_OUTPUTS)
    return []


# --------------------------------------------------------------------------- preflight
def _as_batch(batch):
    if batch is None:
        raise ValueError(
            "pass an explicit batch ('2025-04' or '2024-04:2025-03'). None would mean 'the newest', "
            "which silently changes meaning the day a newer batch lands.")
    if isinstance(batch, str) and ":" in batch:
        return tuple(batch.split(":"))
    return batch


def preflight(batch, stages=STAGES):
    """Everything that can be known WITHOUT running a notebook. Returns
    {span, stages, issues}; run_batch refuses to start while `issues` is non-empty."""
    span = tf.use_batch(_as_batch(batch), source_dir=str(RAW_DIR)).span
    raw = tf.use_batch(span, source_dir=str(RAW_DIR))
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise ValueError(f"unknown stage(s) {unknown}; valid: {STAGES}")
    stages = tuple(s for s in STAGES if s in stages)          # always the canonical order
    issues, folder = [], tf.batch_dir_name(span)

    if not set(TRANSFORM_OUTPUTS) <= set(TRANSFORM_SCHEMAS):
        issues.append(f"TRANSFORM_OUTPUTS names frames missing from transform_schema: "
                      f"{sorted(set(TRANSFORM_OUTPUTS) - set(TRANSFORM_SCHEMAS))}")
    if "normalize" in stages:
        gone = sorted(set(PROCESS_SCHEMAS) - set(raw.paths))
        if gone:
            issues.append(f"raw batch {tf.span_label(span)} is missing: {gone}")
    for stage in stages:
        nb = NOTEBOOKS / f"{stage}.ipynb"
        issues += parameters_cell_problems(nb) if nb.exists() else [f"{nb} does not exist"]
    made = set()
    for stage in stages:
        for need in NEEDS[stage]:
            if need not in made:
                absent = _absent(need, span)
                if absent:
                    issues.append(f"{stage} needs '{need}' first; problem: {absent}")
        made.add(PRODUCES[stage])
    lock = RUNS_DIR / folder / ".lock"
    if lock.exists():
        issues.append(f"batch is locked ({lock.read_text().strip()}) — delete {lock} if stale")
    return {"span": span, "stages": stages, "issues": issues}


@contextmanager
def _lock(span):
    """One run per batch at a time — normalize rewrites its input in place, so two runs on one
    batch could scale the same file twice. Different batches run side by side."""
    path = RUNS_DIR / tf.batch_dir_name(span) / ".lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise BatchRunError(f"{tf.span_label(span)} is already running: {path.read_text().strip()}")
    with os.fdopen(fd, "w") as handle:
        handle.write(f"pid {os.getpid()} since {datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}")
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- post-checks
def _verify(stage, span, since):
    """What a stage must have written, FRESH. A zero exit only says no cell raised — a stage that
    quietly wrote nothing, or left last run's files, would otherwise count as done."""
    folder, problems = tf.batch_dir_name(span), []

    def fresh(where, directory, labels):
        got = _labels_in(directory)
        for label in labels:
            if label not in got:
                problems.append(f"{where}: '{label}' was not written")
            elif got[label] < since - 1:
                problems.append(f"{where}: '{label}' is older than this stage — not rewritten")

    if stage == "transform":
        fresh("transformed_data", TRANSFORMED_DIR / folder, TRANSFORM_OUTPUTS)
        fresh("processed_data", PROCESSED_DIR / folder / "transform", PROCESSED_OUTPUTS)
        expected = set(TRANSFORM_OUTPUTS)
    elif stage == "feature_addition":
        fresh("transformed_data", TRANSFORMED_DIR / folder, FEATURE_OUTPUTS)
        expected = set(TRANSFORM_OUTPUTS) | set(FEATURE_OUTPUTS)
    elif stage == "validation":
        report = VALIDATION_DIR / folder / "validation_report.md"
        if not report.exists() or report.stat().st_mtime < since - 1:
            problems.append(f"validation_results: {report.name} was not rewritten")
        csvs = list((VALIDATION_DIR / folder).glob("*__transform_validation.csv"))
        if not any(p.stat().st_mtime >= since - 1 for p in csvs):
            problems.append("validation_results: no *__transform_validation.csv was rewritten")
        return problems
    elif stage == "normalize":
        fresh("query_result_data", RAW_DIR / folder, _raw_tables())
        lost = sorted(set(_raw_tables()) - {t["table"] for t in (_normalized_manifest(span) or {}).get("tables", [])})
        if lost:
            problems.append(f"query_result_data manifest does not list {lost}")
        return problems
    else:
        return problems
    lost = sorted(expected - _manifest_tables(TRANSFORMED_DIR / folder))
    if lost:
        problems.append(f"transformed_data manifest does not list {lost}")
    return problems


# --------------------------------------------------------------------------- the run
def _git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _write(record, path):
    path.write_text(json.dumps(record, indent=2))


def run_batch(batch, stages=STAGES, *, dry_run=False, kernel=KERNEL):
    """Run one batch through the chosen stages, in order, stopping at the first failure.

    batch: '2025-04', '2024-04:2025-03' or a ('2024-04','2025-03') span — never None.
    stages: any subset of STAGES; always run in canonical order.
    dry_run: preflight only — nothing executes and nothing is written.
    Returns the run record (also saved as batch_runs/<batch folder>/<stamp>/run.json). Raises
    BatchRunError when preflight fails or a stage fails; the record is written first."""
    plan = preflight(batch, stages)
    span, stages, issues = plan["span"], plan["stages"], plan["issues"]
    if dry_run:
        return {"batch": list(span), "stages": list(stages), "issues": issues, "dry_run": True}
    if issues:
        raise BatchRunError(f"preflight failed for {tf.span_label(span)}:\n  - " +
                            "\n  - ".join(issues))
    import papermill as pm                                    # only a real run needs it

    with _lock(span):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = RUNS_DIR / tf.batch_dir_name(span) / stamp
        run_dir.mkdir(parents=True)
        record = {"batch": list(span), "started_utc": stamp, "status": "running",
                  "git_sha": _git("rev-parse", "--short", "HEAD"),
                  "git_dirty": bool(_git("status", "--porcelain", "--", "src", "notebooks", "schemas")),
                  "stages": []}
        record_path = run_dir / "run.json"
        _write(record, record_path)
        failure = None

        for stage in stages:
            executed = run_dir / f"{stage}.ipynb"
            entry = {"stage": stage, "executed_copy": str(executed.relative_to(ROOT)),
                     "status": "running"}
            record["stages"].append(entry)
            began, clock = time.time(), time.monotonic()
            try:
                pm.execute_notebook(str(NOTEBOOKS / f"{stage}.ipynb"), str(executed),
                                    parameters={PARAM_NAME: list(span)}, kernel_name=kernel,
                                    cwd=str(ROOT), progress_bar=False)
                injected = _injected_value(executed)
                problems = [] if injected == list(span) else [
                    f"papermill injected {PARAM_NAME}={injected!r}, expected {list(span)}"]
                problems += _verify(stage, span, began)
                entry["status"] = "failed" if problems else "ok"
                if problems:
                    entry["problems"], failure = problems, f"{stage}: " + "; ".join(problems)
            except pm.PapermillExecutionError as exc:         # a cell raised — asserts and gates land here
                entry["status"] = "failed"
                entry["error"] = {"cell": exc.exec_count, "type": exc.ename,
                                  "message": str(exc.evalue)[:500]}
                failure = f"{stage}: cell {exc.exec_count} raised {exc.ename}: {str(exc.evalue)[:200]}"
            entry["seconds"] = round(time.monotonic() - clock, 1)
            _write(record, record_path)
            if failure:
                break

        record["status"] = "failed" if failure else "ok"
        record["finished_utc"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        _write(record, record_path)
    if failure:
        raise BatchRunError(f"{tf.span_label(span)} failed at {failure}\n  record: {record_path}")
    return record


# --------------------------------------------------------------------------- status
def _last_run(span):
    runs = sorted((RUNS_DIR / tf.batch_dir_name(span)).glob("*/run.json"))
    if not runs:
        return "", ""
    record = json.loads(runs[-1].read_text())
    return record.get("status", ""), record.get("started_utc", "")


def status():
    """One row per extracted batch: how far it has got, read from disk — no state of its own.
    stale_features: a feature frame is older than a transform frame it was built from.
    stale_validation: the report is older than a frame it validated."""
    rows = []
    for span in tf.batches(source_dir=str(RAW_DIR)):
        folder = tf.batch_dir_name(span)
        raw = tf.use_batch(span, source_dir=str(RAW_DIR))
        made = _labels_in(TRANSFORMED_DIR / folder)
        t_times = [made[l] for l in TRANSFORM_OUTPUTS if l in made]
        f_times = [made[l] for l in FEATURE_OUTPUTS if l in made]
        report = VALIDATION_DIR / folder / "validation_report.md"
        newest = max(made.values(), default=0)
        last_status, last_started = _last_run(span)
        norm = _labels_in(RAW_DIR / folder)
        rows.append({
            "batch": tf.span_label(span),
            "raw": f"{len(set(PROCESS_SCHEMAS) & set(raw.paths))}/{len(PROCESS_SCHEMAS)}",
            "normalized": f"{len(set(_raw_tables()) & set(norm))}/{len(_raw_tables())}",
            "stale_normalized": bool(_normalized_stale(span)),
            "transform": f"{len(t_times)}/{len(TRANSFORM_OUTPUTS)}",
            "features": f"{len(f_times)}/{len(FEATURE_OUTPUTS)}",
            "stale_features": bool(t_times and f_times and min(f_times) < max(t_times) - 1),
            "validated": report.exists(),
            "stale_validation": bool(report.exists() and report.stat().st_mtime < newest - 1),
            "last_run": f"{last_status} {last_started}".strip(),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- command line
def main(argv=None):
    parser = argparse.ArgumentParser(description="Run or inspect batch processing.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="one row per extracted batch: how far it got")
    run = sub.add_parser("run", help="run a batch through the stages")
    run.add_argument("batch", help="'2025-04' or a full span '2024-04:2025-03' (never omitted)")
    run.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    run.add_argument("--dry-run", action="store_true", help="preflight only; runs nothing")
    args = parser.parse_args(argv)

    if args.command == "status":
        print(status().to_string(index=False))
        return 0
    try:
        result = run_batch(args.batch, tuple(args.stages), dry_run=args.dry_run)
    except (BatchRunError, KeyError, ValueError) as exc:
        print(f"FAILED: {exc}")
        return 1
    if args.dry_run:
        print(f"batch {result['batch']}  stages {result['stages']}")
        for issue in result["issues"]:
            print(f"  - {issue}")
        print("preflight OK — nothing run" if not result["issues"]
              else f"{len(result['issues'])} issue(s) — fix before a real run")
        return 1 if result["issues"] else 0
    print(f"OK  {result['batch']}  " + "  ".join(
        f"{s['stage']} {s['seconds']}s" for s in result["stages"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
