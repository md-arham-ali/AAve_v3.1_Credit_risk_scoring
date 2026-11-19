# Aave V3.1 — Liquidation-stress pipeline

Extract Aave V3.1 (Ethereum mainnet) activity from [Dune Analytics](https://dune.com), turn it into a clean
2-hour panel, and predict **liquidation stress**: will the liquidation debt covered over the next 24h / 3d / 7d
land in the top 25% (the train 75th percentile)?

- **Data:** two batches, 2024-04 → 2025-03 and 2025-04 → 2026-03, joined into one 2024-04 → 2026-03 run.
- **Rows:** 2-hour buckets keyed on `time_bucket`. The 24h target has 5,945 train / 1,287 val / 1,300 test rows.
- **Features:** 43 panel columns, plus an optional panel + liquidation pool of 49.

## Pipeline

| Stage | What it does | Code |
|-------|--------------|------|
| 1. Extract | Credit-optimized DuneSQL, one scan per table, raw integer amounts. | `optimized_queries/`, `process.ipynb`, `src/dune_fetch.py` |
| 2. Normalize | Raw → real units (decimals, RAY, bps) per batch, schema-gated. | `normalize.ipynb`, `src/normalize_stage.py` |
| 3. Validate | Structural checks, finance-aware invariants, statistical profiling. | `validation.ipynb`, `src/data_validation.py`, `src/adv_validation.py` |
| 4. Transform | Value in USD / ETH, aggregate to protocol-level 2h series. | `transform.ipynb`, `src/transform.py` |
| 5. Features | Derived metrics, rolled windows (12h / 3d / 7d), the model panel. | `feature_addition.ipynb`, `model_features.ipynb`, `model_dataset.ipynb` |
| 6. Split | Time split with an embargo per horizon, walk-forward folds on train, scaler fit on train only. | `model_split.ipynb`, `src/model_split.py` |
| 7. Train | Baselines, tuned models, ensemble and the PRI blend. | `model_training.ipynb`, `src/model_*.py` |
| 8. Evidence | Re-runnable checks behind every tuning decision; test is read once per setup. | `model_tuning_evidence.ipynb`, `src/model_lab.py` |
| 9. Report | Cross-model leaderboard. | `model_results.ipynb` |

**Orchestration and serving:**
- `orchestration/definitions.py`: 8 Dagster assets that run the notebooks with papermill.
- `docker-compose.yml`: containers for extraction, validate-transform, credit-risk prep and model, plus MLflow.
- `backend/` (FastAPI) and `frontend/`: a read-only dashboard over the results.

## Models

All choices are made on walk-forward CV over train. Val only sets the F1 threshold, and test is read once per finished setup.

- **Baselines:** persistence and prevalence.
- **Linear and probabilistic:** logistic regression, Elastic Net, ridge classifier, Gaussian NB.
- **Trees:** random forest, XGBoost, CatBoost, LightGBM.
- **Other:** SVC, MLP, IsolationForest (anomaly), Cox PH (survival).
- **Sequence and time series:** LSTM, Transformer encoder, SARIMAX, HMM regime model.
- **Blends:** equal-weight ensemble of 11 models, and the Protocol Risk Index (PRI).
- **Dropped, kept as commented cells in the notebook:** extra trees, HistGradientBoosting, KNN, and the ridge / SVR / XGBoost regressors.

## Results (24h ROC-AUC, latest official run, 2026-11-18)

| Model | CV | Val | Test |
|---|---|---|---|
| Persistence (baseline) | – | 0.584 | 0.537 |
| Random forest | 0.672 | 0.584 | 0.616 |
| XGBoost | 0.675 | 0.555 | 0.618 |
| CatBoost | 0.682 | 0.553 | 0.617 |
| Logistic regression | 0.683 | 0.551 | 0.619 |
| LSTM | 0.695 | 0.538 | 0.616 |
| Transformer | 0.679 | 0.570 | 0.618 |
| SARIMAX | 0.684 | 0.594 | 0.599 |
| HMM regime | 0.690 | 0.551 | 0.617 |
| Ensemble (mean of 11) | 0.691 | 0.560 | 0.618 |
| PRI | – | 0.586 | 0.620 |

What the work found:
- **One column carries the 24h signal.** `unique_repayers` alone scores CV 0.683, and most tuned models end on it.
  CV sits at 0.68–0.70, val at 0.54–0.59 and test at about 0.62. The test CI is about ±0.12.
- **Training-control tuning is exhausted.** Over three rounds, loss, optimiser, schedule, regularisation and
  hyper-parameters never moved a score beyond the seed noise.
- **Columns are the only lever.** Round three moved six models to panel + liq columns (val target, test not read yet):

| Model | Val official → round three | CV official → round three |
|---|---|---|
| XGBoost | 0.555 → 0.659 | 0.675 → 0.691 |
| CatBoost | 0.553 → 0.648 | 0.682 → 0.702 |
| LSTM | 0.538 → 0.636 | 0.695 → 0.682 |
| Transformer | 0.570 → 0.660 | 0.679 → 0.671 |
| SARIMAX | 0.594 → 0.656 | 0.684 → 0.678 |
| HMM regime | 0.551 → 0.637 | 0.690 → 0.680 |

**Still open:**
- The test read for round three (evidence section 8).
- A fix for the HMM mixture-emission NaN.
- Checking panel + liq on a new batch.
- A health-factor feature (debt close to liquidation).

## Where results live

| What | Where |
|---|---|
| Official model results (config, metrics, CV) | `model_results/<model>__<kind>.json`; load with `mps.load_model_results()` |
| Scores per row | `model_results/predictions_*.csv`, `oof_*.csv`, `tuning_*.csv` |
| Fitted models, run identity | `model_results/fitted/`, `model_results/run_manifest.json` |
| Round-three stage tables | notebook outputs, or `model_results/round3/` after `lab.save_round_three(...)` |
| MLflow | experiment `credit_risk_2h` at http://localhost:5001 (`docker compose up -d mlflow`), data in `mlflow_data/` |
| Decisions and project log | `context/Descisions.md`, `context/context.md`, `context/context_modelling+research.md` |

## Layout

```
optimized_queries/   extraction SQL + docs
src/                 all logic (fetch, normalize, validate, transform, features, split, models, lab)
domains/credit_risk/ domain config (model_config.py)
notebooks/           orchestration notebooks, one per stage
orchestration/       Dagster assets
docker/              Dockerfiles per service
backend/, frontend/  dashboard API and UI
context/             project log, decisions, plans
query_result_data/   raw + normalized extracts per batch   (git-ignored)
transformed_data/    transformed frames per batch          (git-ignored)
model_data/          model panel, targets and split        (git-ignored)
model_results/       saved model results                   (git-ignored)
```

## Setup

Two environments (validation pins an older Great Expectations):

```bash
# main env: fetch, transform, features, models
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# validation env: Great Expectations 0.18.22 / pandas 2.3.3 / numpy 1.26.4
python -m venv .venv-ge && source .venv-ge/bin/activate
pip install -r requirements-ge.txt   # exposes the "aave-ge" Jupyter kernel
```

Create a `.env` with your Dune API key (never committed, see `.gitignore` and `.env.example`):

```
DUNE_API_KEY=your_key_here
```

## Conventions

- **Notebooks orchestrate, modules define.** Notebooks hold settings and calls; logic lives in `src/`.
- **`time_bucket` is the primary key**; buckets are stamped at their end.
- **Versioned exports:** each fetch writes `query_result_data/<table>_<query_id>_<start>_<end>.csv`. Resolve them with
  `transform.latest_paths()` / `newest_matching()`, never `sorted(glob(...))[-1]`.
- **Decimals-safe:** raw uint256 / int256 values are parsed as exact Python `int` wherever an exact identity matters.
- **No leakage:** every fit (scaler, column choice, model) reads train rows only; test is never used to choose anything.
