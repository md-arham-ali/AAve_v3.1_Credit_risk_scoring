"""Publish a run bundle — one immutable folder per run, a manifest, a gate, then `latest.json`.

Plumbing only. It decides nothing about HOW a batch is processed: it gathers what the stages
already wrote into `<dest>/runs/<run_id>/`, records where it came from, checks it, and moves the
`<dest>/manifests/latest.json` pointer ONLY when the check passes. The folder layout mirrors the
four roots backend/dashboard_data.py reads, so serving a bundle means pointing those roots at it.

Uploading `<dest>` to a bucket is §4's job (copy `runs/<run_id>/` first, `latest.json` last).

    python src/publish_bundle.py publish 2025-04 --dest bundles [--started UTC] [--reduced]
    python src/publish_bundle.py publish 2025-04 --dest bundles --dry-run   # lists, writes nothing
    python src/publish_bundle.py check bundles/runs/<run_id>                 # re-run the gate

Dagster (§4) calls `publish(...)` as the last asset. Stdlib only, apart from `transform` for the
batch folder name.
"""
import argparse
import csv
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

SCHEMA_VERSION = 1
RUNS_DIR, MANIFESTS_DIR = "runs", "manifests"
MANIFEST_NAME, LATEST_NAME = "manifest.json", "latest.json"
BATCH_RUNS = ROOT / "batch_runs"

# A bundle without these two models is a REDUCED run: fine as an integration test, never promotable.
TORCH_MODELS = ("lstm__classifier", "transformer__classifier")

# Bundle section -> where it comes from. `{batch}` is transform.batch_dir_name(span). `take` is what
# gets copied; `need` must each match at least one taken file or the gate fails. model_data/ is the
# Stage-0 §3 location — if the model stage lands elsewhere, pass `sections=` instead of editing this.
SECTIONS = {
    "model_results": {
        "src": "model_results",
        "take": ["*.json", "predictions_*.csv", "pri_timeseries.csv"],
        "need": ["run_manifest.json", "pri_timeseries.csv", "*__classifier.json"]},
    "model_data/{batch}": {
        "src": "model_data/{batch}",
        "take": ["DF_model_dataset_24h.csv", "DF_model_train.csv", "DF_model_val.csv",
                 "DF_model_test.csv", "model_split_meta.json"],
        "need": ["DF_model_dataset_24h.csv", "DF_model_train.csv", "DF_model_val.csv",
                 "DF_model_test.csv", "model_split_meta.json"]},
    "transformed_data/{batch}": {
        "src": "transformed_data/{batch}",
        "take": ["DF_common_final_1_*.csv", "DF_common_final_2_*.csv", "_manifest.json"],
        "need": ["DF_common_final_1_*.csv", "DF_common_final_2_*.csv", "_manifest.json"]},
    "validation_results/{batch}": {
        "src": "validation_results/{batch}",
        "take": ["*__transform_validation.csv", "validation_report.md"],
        "need": ["*__transform_validation.csv"]},
}


class PublishError(RuntimeError):
    """A bundle that cannot be built (unknown batch, run_id already published)."""


# --------------------------------------------------------------------------- small helpers
def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv_rows(path):
    with open(path, newline="") as fh:
        return max(sum(1 for _ in csv.reader(fh)) - 1, 0)


def _utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_utc(text):
    """A UTC datetime from '20261001T111126Z' or an ISO string; None when it is neither."""
    if not text:
        return None
    for parse in (lambda s: datetime.fromisoformat(s),
                  lambda s: datetime.strptime(s, "%Y%m%dT%H%M%SZ")):
        try:
            when = parse(str(text))
        except ValueError:
            continue
        return when if when.tzinfo else when.replace(tzinfo=timezone.utc)
    return None


def _git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _as_batch(batch):
    if batch is None:
        raise PublishError("pass an explicit batch ('2025-04' or '2024-04:2025-03'), never None")
    return tuple(batch.split(":")) if isinstance(batch, str) and ":" in batch else batch


def _describe(path, rel):
    return {"path": rel, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


# --------------------------------------------------------------------------- gathering
def _collect(root, folder, sections):
    """([(source file, path inside the bundle)], [required patterns that matched nothing])."""
    pairs, missing = [], []
    for dest_tpl, spec in sections.items():
        src, dest = Path(root) / spec["src"].format(batch=folder), dest_tpl.format(batch=folder)
        taken = []
        for pattern in spec["take"]:
            taken += [p for p in sorted(src.glob(pattern)) if p.is_file() and p not in taken]
        pairs += [(p, f"{dest}/{p.name}") for p in taken]
        missing += [f"{dest}/{pat}" for pat in spec["need"]
                    if not any(fnmatch.fnmatch(p.name, pat) for p in taken)]
    return pairs, missing


def _last_batch_run(folder):
    """(run.json record, its folder) of the newest finished-ok batch_run for this batch, else (None, None)."""
    for path in sorted((BATCH_RUNS / folder).glob("*/run.json"), reverse=True):
        record = json.loads(path.read_text())
        if record.get("status") == "ok":
            return record, path.parent
    return None, None


def _inputs(root, span, folder):
    """Raw extracts and their normalized copies with size + sha256 — what the run was built FROM."""
    import transform as tf
    source = Path(root) / "query_result_data"
    raw = tf.use_batch(span, source_dir=str(source)).paths            # {label: path}
    normalized = sorted((source / folder).glob("*.csv"))
    return {"raw": [_describe(Path(p), Path(p).name) for p in sorted(raw.values())],
            "normalized": [_describe(p, f"{folder}/{p.name}") for p in normalized]}


# --------------------------------------------------------------------------- the gate
def check_bundle(bundle_dir, manifest):
    """The publish-integrity gate. Returns [{name, ok, detail}] — five checks, no more until a real
    failure shows a gap:
      1 every file exists and matches its recorded size + sha256, and every required file is there
      2 the run finished ok and the validation reports hold zero HARD failures
      3 the model run happened after the pipeline started (a stale model_results/ cannot pass)
      4 the model's recorded split rows equal the row counts of the three split CSVs
      5 every batch tag in the bundle equals the batch being published
    """
    bundle, checks = Path(bundle_dir), []

    def add(name, bad, ok_detail=""):
        checks.append({"name": name, "ok": not bad, "detail": "; ".join(bad) if bad else ok_detail})

    # 1 — files
    bad = [f"required but absent: {p}" for p in manifest.get("missing_required", [])]
    for entry in manifest.get("files", []):
        path = bundle / entry["path"]
        if not path.is_file():
            bad.append(f"missing {entry['path']}")
        elif path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            bad.append(f"changed since copied: {entry['path']}")
    add("files complete and intact", bad, f"{len(manifest.get('files', []))} files")

    # 2 — status and validation
    bad, hard = [], []
    if manifest.get("status") != "ok":
        bad.append(f"run status is {manifest.get('status')!r}")
    for report in bundle.glob("validation_results/*/*__transform_validation.csv"):
        with open(report, newline="") as fh:
            hard += [f"{report.name.split('__')[0]}: {r.get('check')}" for r in csv.DictReader(fh)
                     if r.get("severity") == "hard" and float(r.get("n_fail") or 0) > 0]
    if hard:                                       # count failing CHECKS, not failing rows
        bad.append(f"{len(hard)} hard validation check(s) failing — " + "; ".join(hard[:3])
                   + (" ..." if len(hard) > 3 else ""))
    add("run ok, zero hard validation failures", bad, "status ok, 0 hard failures")

    # 3 — freshness
    started = _parse_utc(manifest.get("started_utc"))
    ran = _parse_utc((manifest.get("model") or {}).get("run_at_utc"))
    bad = []
    if started is None:
        bad.append("no pipeline start time recorded — pass --started, or run batch_run first")
    elif ran is None:
        bad.append("model_results/run_manifest.json has no run_at_utc")
    elif ran < started:
        bad.append(f"model run {ran:%Y-%m-%d %H:%M}Z is older than pipeline start {started:%Y-%m-%d %H:%M}Z")
    add("model results are fresh", bad, f"model run {ran:%Y-%m-%d %H:%M}Z" if ran else "")

    # 4 — split rows
    want, bad = (manifest.get("model") or {}).get("split_rows") or {}, []
    for split in ("train", "val", "test"):
        path = next(iter(sorted(bundle.glob(f"model_data/*/DF_model_{split}.csv"))), None)
        if path is None:
            bad.append(f"DF_model_{split}.csv not in bundle")
        elif _csv_rows(path) != want.get(split):
            bad.append(f"{split}: csv has {_csv_rows(path)} rows, run_manifest says {want.get(split)}")
    add("split rows match the model's manifest", bad, "train/val/test row counts agree")

    # 5 — batch tags
    span, bad = list(manifest.get("batch") or []), []
    tagged = [(p, json.loads(p.read_text()).get("batch"))
              for p in sorted(bundle.glob("transformed_data/*/_manifest.json"))]
    tagged += [(p, json.loads(p.read_text()).get("batch"))
               for p in sorted(bundle.glob("model_data/*/model_split_meta.json"))]
    if not tagged:
        bad.append("no batch-tagged manifest or split meta in the bundle")
    bad += [f"{p.relative_to(bundle)} says batch {tag!r}, expected {span}" if tag is not None
            else f"{p.relative_to(bundle)} carries no batch tag"
            for p, tag in tagged if tag is None or list(tag) != span]
    add("batch tags agree with the batch", bad, f"batch {span}")
    return checks


# --------------------------------------------------------------------------- publish
def publish(batch, dest, *, started_utc=None, extra_stages=(), status="ok", reduced=False,
            include_logs=True, sections=None, root=ROOT, run_id=None, dry_run=False):
    """Build one bundle for `batch` under `dest`, check it, and move latest.json if it passes.

    started_utc: when this pipeline run began; the model results must be newer. Default: the start
        of the newest ok batch_run record for the batch.
    extra_stages: [{'stage','seconds','status'}] for stages batch_run does not cover (the model
        stage, timed by §4). Merged into the manifest next to batch_run's own timings.
    status: 'ok' or 'failed' — the caller's verdict on the run. A failed run is still bundled (the
        evidence is worth keeping) but the gate rejects it, so latest.json does not move.
    reduced: set True when the run deliberately skipped models (e.g. torch). Marks it non-promotable.
    sections: override SECTIONS when the stages write elsewhere.
    dry_run: list what would be copied and what is missing; write nothing.
    Returns the manifest (also saved in the bundle). A bundle whose gate fails is kept but not pointed at."""
    import transform as tf
    root, dest = Path(root), Path(dest)
    span = tuple(tf.use_batch(_as_batch(batch), source_dir=str(root / "query_result_data")).span)
    folder = tf.batch_dir_name(span)
    pairs, missing = _collect(root, folder, sections or SECTIONS)
    if dry_run:
        return {"batch": list(span), "would_copy": [rel for _, rel in pairs], "missing_required": missing,
                "dry_run": True}

    record, record_dir = _last_batch_run(folder)
    sha = _git("rev-parse", "--short", "HEAD") or "nogit"
    run_id = run_id or f"{_utc_stamp()}-{sha}"
    final, staging = dest / RUNS_DIR / run_id, dest / RUNS_DIR / f".{run_id}.tmp"
    if final.exists() or staging.exists():
        raise PublishError(f"{final} already exists — run ids are never reused")
    staging.mkdir(parents=True)

    try:
        files = []
        for src, rel in pairs:
            target = staging / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            files.append(_describe(target, rel))
        if include_logs and record_dir:                         # the executed notebooks are the proof
            for src in sorted(record_dir.glob("*")):
                target = staging / "logs" / "batch_run" / src.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)
                files.append(_describe(target, f"logs/batch_run/{src.name}"))

        model_manifest = staging / "model_results" / "run_manifest.json"
        model = {}
        if model_manifest.is_file():
            model = {k: v for k, v in json.loads(model_manifest.read_text()).items()
                     if k in ("run_at_utc", "python", "versions", "split_rows", "n_features",
                              "embargo_days", "thresholds")}
        torch_missing = [m for m in TORCH_MODELS if not (staging / "model_results" / f"{m}.json").is_file()]
        stages = [{"stage": s["stage"], "seconds": s.get("seconds"), "status": s.get("status"),
                   "source": "batch_run"} for s in (record or {}).get("stages", [])]
        stages += [dict(s, source=s.get("source", "caller")) for s in extra_stages]
        try:
            inputs, inputs_error = _inputs(root, span, folder), ""
        except Exception as exc:                                  # inputs are provenance, not a gate
            inputs, inputs_error = {"raw": [], "normalized": []}, f"{type(exc).__name__}: {exc}"

        manifest = {
            "schema": SCHEMA_VERSION, "run_id": run_id, "published_utc": _utc_stamp(),
            "batch": list(span), "status": status,
            "started_utc": started_utc or (record or {}).get("started_utc", ""),
            "started_from": "caller" if started_utc else ("batch_run" if record else "none"),
            "git": {"sha": _git("rev-parse", "--short", "HEAD"),
                    "dirty": bool(_git("status", "--porcelain", "--", "src", "notebooks", "schemas",
                                       "domains", "orchestration", "docker"))},
            "stages": stages, "model": model,
            "reduced": bool(reduced or torch_missing), "reduced_missing": torch_missing,
            "inputs": inputs, "inputs_error": inputs_error,
            "files": files, "missing_required": missing,
        }
        (staging / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))

        checks = check_bundle(staging, manifest)
        passed = all(c["ok"] for c in checks)
        reasons = [c["name"] for c in checks if not c["ok"]]
        if manifest["reduced"]:
            reasons.append("reduced model set" + (f" (missing {torch_missing})" if torch_missing else ""))
        manifest["gate"] = {"passed": passed, "checks": checks}
        manifest["promotable"] = passed and not manifest["reduced"]
        manifest["not_promotable_because"] = [] if manifest["promotable"] else reasons
        (staging / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
        os.replace(staging, final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)               # never leave a half-built bundle
        raise

    if passed:                                                    # the pointer moves last, and only here
        pointer = {"run_id": run_id, "bundle": f"{RUNS_DIR}/{run_id}", "batch": list(span),
                   "manifest_sha256": sha256_file(final / MANIFEST_NAME),
                   "published_utc": manifest["published_utc"], "promotable": manifest["promotable"]}
        (dest / MANIFESTS_DIR).mkdir(parents=True, exist_ok=True)
        tmp = dest / MANIFESTS_DIR / f".{LATEST_NAME}.tmp"
        tmp.write_text(json.dumps(pointer, indent=2))
        os.replace(tmp, dest / MANIFESTS_DIR / LATEST_NAME)
    return manifest


# --------------------------------------------------------------------------- command line
def main(argv=None):
    parser = argparse.ArgumentParser(description="Publish or check a run bundle.")
    sub = parser.add_subparsers(dest="command", required=True)
    pub = sub.add_parser("publish", help="build a bundle, run the gate, move latest.json if it passes")
    pub.add_argument("batch", help="'2025-04' or a full span '2024-04:2025-03' (never omitted)")
    pub.add_argument("--dest", required=True, help="folder that holds runs/ and manifests/")
    pub.add_argument("--started", default=None, help="pipeline start, UTC (default: newest batch_run record)")
    pub.add_argument("--status", choices=("ok", "failed"), default="ok")
    pub.add_argument("--reduced", action="store_true", help="the run skipped models on purpose")
    pub.add_argument("--no-logs", action="store_true")
    pub.add_argument("--dry-run", action="store_true", help="list what would be copied; write nothing")
    chk = sub.add_parser("check", help="re-run the gate on an existing bundle (e.g. after a download)")
    chk.add_argument("bundle", help="path to runs/<run_id>")
    args = parser.parse_args(argv)

    if args.command == "check":
        manifest = json.loads((Path(args.bundle) / MANIFEST_NAME).read_text())
        checks = check_bundle(args.bundle, manifest)
        for c in checks:
            print(f"{'PASS' if c['ok'] else 'FAIL'}  {c['name']}  {c['detail']}")
        return 0 if all(c["ok"] for c in checks) else 1

    try:
        result = publish(args.batch, args.dest, started_utc=args.started, status=args.status,
                         reduced=args.reduced, include_logs=not args.no_logs, dry_run=args.dry_run)
    except (PublishError, KeyError, ValueError) as exc:
        print(f"FAILED: {exc}")
        return 1
    if args.dry_run:
        print(f"batch {result['batch']}: {len(result['would_copy'])} files would be copied")
        for pattern in result["missing_required"]:
            print(f"  missing required: {pattern}")
        return 1 if result["missing_required"] else 0
    for c in result["gate"]["checks"]:
        print(f"{'PASS' if c['ok'] else 'FAIL'}  {c['name']}  {c['detail']}")
    print(f"run {result['run_id']}  gate {'passed' if result['gate']['passed'] else 'FAILED'}  "
          f"promotable {result['promotable']}")
    if result["gate"]["passed"]:
        print(f"latest.json -> {result['run_id']}")
    else:
        print("latest.json NOT moved; the bundle is kept for inspection")
    return 0 if result["gate"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
