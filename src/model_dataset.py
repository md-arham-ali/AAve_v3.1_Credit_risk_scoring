"""Modeling stage — daily credit-risk dataset assembly.

2h panel -> daily grain, joined with the 24h liq/user risk frames, restricted to
the credit-risk feature set, plus the forward-looking stress/volume targets.
"""

# TODO: the 7d variant waits on more bulk data.

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import transform as tf
import EDA as eda
from data_validation import canonicalize_keys
from model_config import (TIME_COL, HORIZONS, TARGET_BASE_COL, REG_TARGET,
                          LIQ_CONDITIONAL_COLS, CREDIT_RISK_COLS, CREDIT_RISK_WEIGHTS,
                          CREDIT_RISK_PRIORITY, AGG_OVERRIDES, AGG_RULES,
                          RATIO_RECIPES, SHARE_RECIPES)


def daily_agg_map(columns):
    """Aggregation function per column for the 2h->daily rollup.
    columns -> AGG_OVERRIDES first, then the ordered AGG_RULES matchers.
    Returns: {column: agg}; unmatched columns fall back to mean."""
    out = {}
    for col in columns:
        if col in AGG_OVERRIDES:
            out[col] = AGG_OVERRIDES[col]
            continue
        for matcher, agg in AGG_RULES:
            if matcher(col):
                out[col] = agg
                break
        else:
            out[col] = "mean"
    return out


def build_daily_panel(df, time_col=TIME_COL, freq="24h"):
    """Aggregate the 2h protocol panel to daily grain with type-appropriate functions.
    df -> counts/flows summed, ratios and state averaged (via transform.aggregate_by_time_bucket).
    Returns: daily DataFrame with canonical string keys."""
    value_cols = [c for c in df.columns if c != time_col]
    amap = daily_agg_map(value_cols)
    daily = tf.aggregate_by_time_bucket(df, time_col, value_cols, agg_func=amap, freq=freq)
    return recompute_daily_ratios(canonicalize_keys(daily))


def recompute_daily_ratios(daily, ratios=RATIO_RECIPES, shares=SHARE_RECIPES):
    """Rebuild ratio columns from their daily-summed components (ratio-of-sums).
    df -> overwrites the mean-of-2h-ratios values left by the rollup.
    Returns: the frame with consistent daily ratios."""
    out = daily.copy()
    n = 0
    for col, (num, den) in ratios.items():
        if col in out.columns and num in out.columns and den in out.columns:
            out[col] = out[num] / out[den].replace(0, np.nan)
            n += 1
    for col, (num, dens) in shares.items():
        if col in out.columns and all(d in out.columns for d in dens):
            denom = sum(out[d] for d in dens)
            out[col] = out[num] / denom.replace(0, np.nan)
            n += 1
    print(f" recomputed {n} daily ratios as ratio-of-sums")
    return out


def join_model_frames(daily, liq, user, time_col=TIME_COL):
    """Inner-join the daily panel with the liquidation and user-account risk frames.
    Duplicated borrow-context columns are allclose-verified, then dropped.
    Returns: the joined frame; prints how many rows the inner joins drop."""
    daily = canonicalize_keys(daily)
    liq = canonicalize_keys(liq)
    user = canonicalize_keys(user)

    dup_liq = [c for c in ("borrow_tx_count", "unique_borrowers",
                           "borrow_amount_value_usd", "borrow_amount_value_eth")
               if c in liq.columns]
    dup_user = [c for c in ("borrow_amount_value_usd",) if c in user.columns]

    check = daily[[time_col] + dup_liq].merge(
        liq[[time_col] + dup_liq], on=time_col, suffixes=("_panel", "_liq"))
    for c in dup_liq:
        a = check[f"{c}_panel"].astype(float).to_numpy()
        b = check[f"{c}_liq"].astype(float).to_numpy()
        if not np.allclose(a, b, rtol=1e-6, equal_nan=True):
            raise AssertionError(f"borrow-context mismatch between panel and liq frame: {c}")

    merged = daily.merge(liq.drop(columns=dup_liq), on=time_col, how="inner")
    merged = merged.merge(user.drop(columns=dup_user), on=time_col, how="inner")
    dropped = max(len(daily), len(liq), len(user)) - len(merged)
    print(f" inner join: {len(merged)} rows kept, {dropped} dropped "
          f"(daily {len(daily)} / liq {len(liq)} / user {len(user)})")
    return merged


def fill_conditional_zeros(df, columns=LIQ_CONDITIONAL_COLS, indicator_col="has_liquidation",
                           count_col="liquidation_tx_count"):
    """Zero-fill conditional liquidation ratios; add the has-liquidation indicator.
    df -> NA ratios on zero-liquidation days become 0 (the honest value).
    Returns: the frame plus the indicator, keeping "no event" separable from "small event"."""
    out = df.copy()
    cols = [c for c in columns if c in out.columns]
    out[cols] = out[cols].fillna(0.0)
    if count_col in out.columns:
        out[indicator_col] = (out[count_col].fillna(0) > 0).astype(int)
    return out


def sync_priorities(weights=None):
    """Push the tiered CREDIT_RISK_WEIGHTS into column_priority.json (the EDA map).
    Replaces the old flat 2.0 sync — one weighting scheme for the whole project.
    Returns: the refreshed map DataFrame."""
    weights = CREDIT_RISK_WEIGHTS if weights is None else weights
    eda.register_columns(list(weights))
    return eda.update_priorities(dict(weights))


def select_credit_risk_columns(df, source="constant", threshold=CREDIT_RISK_PRIORITY):
    """Ordered training-feature list — credit-risk metrics only.
    source="constant" intersects CREDIT_RISK_COLS; "priority" takes score >= threshold.
    Returns: list of column names; missing expected columns are printed, not raised."""
    if source == "priority":
        pri = eda.get_priorities()
        wanted = pri.loc[pri["score"] >= threshold, "column"].tolist()
    else:
        wanted = list(CREDIT_RISK_COLS)
    present = [c for c in wanted if c in df.columns]
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        print(f" ⚠️ {len(missing)} expected credit-risk columns missing: {missing}")
    print(f" selected {len(present)} credit-risk feature columns")
    return present


def add_forward_targets(df, target_col=TARGET_BASE_COL, horizons=HORIZONS, time_col=TIME_COL):
    """Add forward-looking targets and trim the incomplete trailing window.
    target_next_1d = t+1; target_fwd_max_{h}d = max over t+1..t+h; y_reg_log1p = log1p of the former.
    Returns: the frame minus the last max(horizons) rows (count printed)."""

    # binarization into y_stress_{h}d waits until after the split — keeps the threshold train-only
    out = df.sort_values(time_col, kind="stable").reset_index(drop=True).copy()
    s = out[target_col].astype(float)
    out["target_next_1d"] = s.shift(-1)
    for h in horizons:
        fwd = s[::-1].rolling(h, min_periods=h).max()[::-1].shift(-1)
        out[f"target_fwd_max_{h}d"] = fwd
    out[REG_TARGET] = np.log1p(out["target_next_1d"])
    trim = max(horizons)
    print(f" dropped {trim} trailing rows with incomplete forward windows")
    return out.iloc[:-trim].reset_index(drop=True)
