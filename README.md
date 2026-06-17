# Aave V3.1 — On-chain Data Pipeline

Extract, validate, and transform Aave V3.1 (Ethereum mainnet) protocol activity from
[Dune Analytics](https://dune.com) into clean, analysis-ready time series.

All data is bucketed into fixed **6-hour windows** keyed on **`(time_bucket, asset)`**,
over the window **2025-11-01 → 2026-01-31** (≈368 buckets). Across the window there are
**58 distinct transacted assets** (58/58 have decimals, 48/58 have a price feed; the
remainder are price-feedless PT-* tokens).

## Pipeline

| Stage | What it does | Code |
|-------|--------------|------|
| **1. Extract** | Credit-optimized, partition-pruned DuneSQL — one scan per table, raw integer amounts only, plus per-part decimal references and an on-chain `asset_symbol` label. | `optimized_queries/`, `queries/` |
| **2. Fetch** | Pulls a query's **latest stored** result via the Dune API (no re-run, **no credits**) → DataFrame → versioned CSV. | `dune_fetch.py`, `process.ipynb` |
| **3. Validate** | Structural checks (nulls, dtypes, keys, address/time formats) → finance-aware invariants (net-flow integrity, RAY index monotonicity, HF sentinels, …) → univariate pre-EDA statistical profiling. | `data_validation.py`, `adv_validation.py`, `validation.ipynb` |
| **4. Transform** | Scale raw amounts by `10**decimals` → value in USD/ETH via the oracle-price join → aggregate to protocol-level 6h series, with decimals-safe diagnostics. | `transform.py`, `transform.ipynb` |

## Layout

```
optimized_queries/     # 9 extraction queries + decimal-reference SQL + docs
queries/               # oracle price (USD/ETH/WETH) query
dune_fetch.py          # Dune API result fetcher (stored results, no credits)
data_validation.py     # basic structural validation suite (pandas + GE)
adv_validation.py      # finance-aware + statistical (pre-EDA) validation
transform.py           # load / scale / value / aggregate helpers
*.ipynb                # orchestration notebooks (process / validation / transform)
context/context.md     # running event log of the project
query_result_data/     # versioned CSV extracts        (git-ignored, reproducible)
validation_results/    # per-table validation outputs   (git-ignored, generated)
```

## Setup

Two isolated environments (validation pins older Great Expectations):

```bash
# main env — fetch + transform (pandas 3.0, matplotlib, python-dotenv)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# validation env — Great Expectations 0.18.22 / pandas 2.3.3 / numpy 1.26.4
python -m venv .venv-ge && source .venv-ge/bin/activate
pip install -r requirements-ge.txt   # exposes the "aave-ge" Jupyter kernel
```

Create a `.env` with your Dune API key (never committed — see `.gitignore`):

```
DUNE_API_KEY=your_key_here
```

## Conventions

- **Notebooks orchestrate; modules define.** Keep action only in `*.ipynb`; put
  constants, helpers, and definitions in the `.py` modules.
- **`time_bucket` is the primary key**; column order reflects decreasing priority.
- **Versioned exports**: each fetch writes `query_result_data/query_result_data_{query_id}_{fetch_time}.csv`.
- **Decimals-safe**: raw uint256/int256 values are parsed as exact Python `int` (never
  float) wherever an exact identity matters — pandas/GE silently corrupt big ints.
