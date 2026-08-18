"""MLflow logging for model results — one flat MLflow run per model (29 per pipeline run).

Sits BESIDE model_persistence.save_model_result; model_results/*.json stays the
source of truth. Fail-soft: unreachable server means every function no-ops.
"""

# TODO: move these calls into model_training's runners so the notebook stops doing it.

import os

DEFAULT_URI = "http://localhost:5001"    # host-side; in-container it is http://mlflow:5001

_ENABLED = None          # None = not yet probed
_mlflow = None


def _connect(uri=None, experiment=None, quiet=False):
    """Import mlflow, point it at the server, verify it answers.
    Reads the tracking URI from the module config.
    Returns: the mlflow module, or None when tracking is unavailable."""

    # print(), not warnings.warn() — the notebook's filterwarnings swallowed it entirely.
    global _ENABLED, _mlflow
    uri = uri or os.getenv("MLFLOW_TRACKING_URI") or DEFAULT_URI
    experiment = experiment or os.getenv("MLFLOW_EXPERIMENT", "credit_risk")
    try:
        import mlflow
        mlflow.set_tracking_uri(uri)
        # set_experiment round-trips to the server, so it fails here rather than
        # silently creating a local ./mlruns store if the server is unreachable.
        exp = mlflow.set_experiment(experiment)
        _mlflow, _ENABLED = mlflow, True
        if not quiet:
            print(f" MLflow ON  -> {uri}  (experiment '{experiment}', "
                  f"id {exp.experiment_id})")
    except Exception as exc:                       # noqa: BLE001 - fail soft
        _mlflow, _ENABLED = None, False
        print(f" MLflow OFF -> {type(exc).__name__}: {exc}")
        print(f"    tried {uri}. Results are still saved to model_results/.")
        print("    fixes: `docker compose up -d mlflow`, check the port, or "
              "`pip install mlflow==2.22.0`")
    return _mlflow


def init(uri=None, experiment=None):
    """Force a fresh connection attempt and report the outcome. Call once, early.
    Clears the module-wide cached verdict from _connect.
    Returns: bool — whether tracking is live."""
    global _ENABLED, _mlflow
    _ENABLED, _mlflow = None, None
    return _connect(uri, experiment) is not None


def _client():
    """Cached accessor — probes once, then reuses the verdict."""
    if _ENABLED is not None:
        return _mlflow if _ENABLED else None
    return _connect(quiet=False)


def flatten_metrics(result):
    """Result dict -> flat {metric_name: float} that MLflow can sort on.
    Nested targets/metrics collapse to UI-friendly names, e.g. test_roc_auc_1d, test_r2.
    Returns: dict of finite floats only — MLflow rejects NaN."""
    import math
    out = {}

    def _put(key, val):
        if isinstance(val, (int, float)) and math.isfinite(val):
            out[key] = float(val)

    for target, per_split in (result.get("targets") or {}).items():
        h = target.replace("y_stress_", "")            # "1d" / "3d" / "7d"
        for split, m in per_split.items():
            for name, val in m.items():
                _put(f"{split}_{name}_{h}", val)

    for split, m in (result.get("metrics") or {}).items():
        for name, val in m.items():
            _put(f"{split}_{name}", val)

    cv = result.get("cv") or {}
    _put("cv_auc_mean", cv.get("mean"))
    _put("cv_auc_std", cv.get("std"))
    _put("cv_rmse_mean", cv.get("rmse_mean"))

    ci = result.get("test_auc_ci") or {}
    _put("test_auc_ci_lo", ci.get("lo"))
    _put("test_auc_ci_hi", ci.get("hi"))

    return out


def log_model_result(result, split_meta=None, extra_tags=None, run_name=None):
    """Log ONE runner result as ONE MLflow run.
    result: a runner's return dict; split_meta: model_split_meta.json; extra_tags: {str: str}.
    Returns: the run id, or None when tracking is off."""
    mlflow = _client()
    if mlflow is None:
        return None

    name = result.get("name", "unknown")
    kind = result.get("kind", "model")

    with mlflow.start_run(run_name=run_name or f"{name}__{kind}") as run:
        mlflow.set_tags({
            "model": name,
            "kind": kind,
            "status": result.get("status", "unknown"),
            **(extra_tags or {}),
        })

        # the model's own tuned hyperparameters
        for k, v in (result.get("params") or {}).items():
            mlflow.log_param(k, v)

        # without these, runs from different splits are indistinguishable in the UI
        if split_meta:
            mlflow.log_param("embargo_days", split_meta.get("embargo_days"))
            mlflow.log_param("split_fractions", split_meta.get("split_fractions"))
            mlflow.log_param("n_features", len(split_meta.get("feature_cols", [])))
            for h, thr in (split_meta.get("thresholds") or {}).items():
                mlflow.log_param(f"stress_threshold_{h}d", thr)

        metrics = flatten_metrics(result)
        if metrics:
            mlflow.log_metrics(metrics)

        # PRI blend composition — which components carried weight this run
        for comp, w in (result.get("weights") or {}).items():
            mlflow.log_metric(f"weight_{comp}", float(w))

        # visible per-call confirmation: silence used to be indistinguishable
        # from success, which is how 29 no-op'd calls went unnoticed
        print(f"    mlflow: logged {name}__{kind} "
              f"({len(metrics)} metrics) run {run.info.run_id[:8]}")
        return run.info.run_id


def log_all_results(results, split_meta=None, extra_tags=None):
    """Log a whole {name__kind: result} mapping — one MLflow run per entry.
    results mapping + split_meta; used by the backfill script.
    Returns: {stem: run_id} for whatever logged successfully."""
    ids = {}
    for stem, res in results.items():
        rid = log_model_result(res, split_meta=split_meta, extra_tags=extra_tags,
                               run_name=stem)
        if rid:
            ids[stem] = rid
    print(f" logged {len(ids)}/{len(results)} results to MLflow")
    return ids
