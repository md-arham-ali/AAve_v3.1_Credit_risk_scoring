"""Credit-risk-scoring domain config — panel constants + this domain's settings.

WHERE THIS LIVES AND WHY
    This file is NOT in src/. src/ is the shared library baked into aave-core and
    inherited by every container; this file is baked into (and mounted onto) the
    credit-risk containers only, at /app/domain, which sits on PYTHONPATH ahead of
    nothing and beside /app/src:

        ENV PYTHONPATH=/app/src:/app/domain

    So `from model_config import ...` inside src/model_dataset.py resolves to THIS
    file when running a credit-risk container, and fails to resolve at all in the
    extraction or validate-transform containers — which is intended. Those stages
    are domain-agnostic and must not be able to reach domain settings.

ADDING A DOMAIN
    Copy this folder to domains/<new_domain>/, edit SECTION 2 only, and point the
    new domain's container at it. SECTION 1 describes the shared Aave 2h panel and
    should stay identical unless that domain genuinely rolls the panel up
    differently. No src/ module changes — the seven model_* modules bind these
    names as default argument values and don't care where the file came from.

Imported by: model_dataset, model_split, model_zoo, model_evaluation,
model_training, model_persistence, model_benchmarks, and the five model notebooks.
"""

# importlib.util must be imported explicitly — `import importlib` alone does not
# bind the submodule, so find_spec below would raise AttributeError.
import importlib.util

import numpy as np

LIFELINES_AVAILABLE = importlib.util.find_spec("lifelines") is not None

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


# =========================================================================== #
# SECTION 1 — PANEL / PIPELINE constants
# True for any domain built on the Aave V3.1 2h protocol panel. Keep identical
# across domains unless a domain genuinely aggregates the panel differently.
# =========================================================================== #

TIME_COL = "time_bucket"

SPLIT_FRACTIONS = (0.70, 0.15, 0.15)

RANDOM_STATE = 42

N_CV_SPLITS = 5

SKEW_LOG1P_THRESHOLD = 2.5

NEAR_CONSTANT_FRAC = 0.99

REG_TARGET = "y_reg_log1p"


AGG_OVERRIDES = {
    "protocol_turnover_usd": "sum",   # per-bucket total value transacted -> daily total
    "protocol_turnover_eth": "sum",
    "user_activity": "sum",           # per-bucket active-user total -> daily activity
}


AGG_RULES = (
    (lambda c: c.endswith("_count"), "sum"),
    (lambda c: c.startswith("unique_"), "sum"),
    (lambda c: c.endswith("_value_usd") or c.endswith("_value_eth"), "sum"),
    (lambda c: c.startswith("net_"), "sum"),
)


# Ratios rebuilt as ratio-of-sums after the 2h->daily rollup. Averaging the twelve
# 2h ratios instead would weight a bucket with $10 of volume the same as one with
# $10M — see model_dataset.recompute_daily_ratios.
RATIO_RECIPES = {
    "supply_withdrawal_ratio": ("supply_amount_value_usd", "withdrawal_amount_value_usd"),
    "borrow_repay_ratio": ("borrow_amount_value_usd", "repay_amount_value_usd"),
    "leverage_indicator": ("borrow_amount_value_usd", "supply_amount_value_usd"),
    "repayment_discipline": ("unique_repayers", "unique_borrowers"),
}


SHARE_RECIPES = {
    "borrower_participation_rate": ("unique_borrowers", ("unique_borrowers", "unique_suppliers")),
    "collateral_usage_rate": ("collateral_enabled_count",
                              ("collateral_enabled_count", "collateral_disabled_count")),
    "flashloan_usage_intensity": ("flashloan_tx_count", ("borrow_tx_count", "supply_tx_count")),
}


def rank_normalize(scores, reference):
    """ECDF of `reference` applied to `scores` → [0, 1].

    Puts classifier probabilities, anomaly scores and partial hazards on one
    common footing (the fraction of reference values each score exceeds).
    """
    ref = np.sort(np.asarray(reference, dtype=float))
    ref = ref[~np.isnan(ref)]
    s = np.asarray(scores, dtype=float)
    if ref.size == 0:
        return np.full(s.shape, 0.5)
    return np.searchsorted(ref, s, side="right") / ref.size


# =========================================================================== #
# SECTION 2 — THE DOMAIN: credit-risk scoring
# Everything below is specific to "will forward liquidation debt-covered USD
# exceed the train p75 over the next 1/3/7 days?". This is the only section a new
# domain rewrites.
# =========================================================================== #

DOMAIN_NAME = "credit_risk_scoring"

# The raw column every target derives from — a real on-chain fact (dollars of debt
# liquidators actually covered), time-shifted forward in
# model_dataset.add_forward_targets to become the label.
TARGET_BASE_COL = "liquidation_debt_covered_value_usd"

HORIZONS = (1, 3, 7)

# Per-horizon train quantile of that horizon's OWN forward-max distribution. One
# shared daily threshold made y_stress_7d ~76% positive; quantiling each horizon
# separately holds every target near (1 - q) positives on train.
STRESS_QUANTILE = 0.75

# Where this domain's results land. A second domain MUST change this or it will
# overwrite these.
MODEL_RESULTS_DIR = "model_results"

# Priority score pushed into column_priority.json for this domain's features.
FEATURE_PRIORITY = 2.0

STRESS_TARGETS = tuple(f"y_stress_{h}d" for h in HORIZONS)

# Label embargo: the widest horizon, so a row's forward-max window can never
# reach across a split boundary.
EMBARGO = max(HORIZONS)


# Liquidation ratios that come out of the join as NA on zero-liquidation days. The
# honest value is 0 (no liquidation stress), not "unknown" — see
# model_dataset.fill_conditional_zeros. E29 precedent from the 2h panel.
CONDITIONAL_ZERO_COLS = (
    "liquidation_rate", "liquidation_volume_ratio_eth",
    "liquidation_severity_usd", "liquidation_severity_eth",
    "avg_liquidation_debt_usd", "avg_liquidation_debt_eth",
    "liquidator_concentration", "liquidation_user_ratio",
    "market_stress_index_usd", "market_stress_index_eth",
)


FEATURE_COLS = (
    # liquidation events + ratios (DF_liq_features_24h)
    "liquidation_tx_count", "unique_liquidated_users", "unique_liquidators",
    "as_collateral_tx_count", "as_debt_tx_count",
    "liquidated_collateral_value_usd", "liquidated_collateral_value_eth",
    "liquidation_debt_covered_value_usd", "liquidation_debt_covered_value_eth",
    "liquidation_rate", "liquidation_volume_ratio_eth",
    "liquidation_severity_usd", "liquidation_severity_eth",
    "avg_liquidation_debt_usd", "avg_liquidation_debt_eth",
    "liquidator_concentration", "liquidation_user_ratio",
    "market_stress_index_usd", "market_stress_index_eth", "has_liquidation",
    # user-account solvency state (DF_user_features_24h)
    "avg_total_collateral_base", "avg_total_debt_base", "avg_available_borrows_base",
    "avg_current_liquidation_threshold", "avg_ltv",
    "borrow_capacity_utilization", "remaining_borrow_capacity", "risk_buffer",
    "ltv_utilization", "distance_to_liquidation", "debt_expansion_ratio",
    "capital_efficiency", "overcollateral_margin", "has_sampled_borrow_position",
    # credit flows (daily-aggregated DF_common_final_1)
    "borrow_tx_count", "unique_borrowers",
    "borrow_amount_value_usd", "borrow_amount_value_eth", "repay_amount_value_usd",
    "borrow_repay_ratio", "repayment_discipline", "net_borrow_demand_usd",
    "borrower_participation_rate", "leverage_indicator",
    "collateral_enabled_count", "collateral_disabled_count", "collateral_usage_rate",
    # market context (daily-aggregated DF_common_final_1)
    "net_liquidity_flow_usd", "supply_withdrawal_ratio", "protocol_turnover_usd",
    "flashloan_usage_intensity", "flashloan_amount_value_usd", "user_activity",
)


# Tiered importance prior — the ONE weighting scheme for this domain. Read by
# model_features.ipynb (QA), model_dataset.sync_priorities (column_priority.json)
# and select_credit_risk_columns. NOT read by any model's .fit(): models learn
# their own coefficients from the data. This is for reporting and selection only.
FEATURE_WEIGHTS = {
    # Tier 1 — position health & leverage
    "ltv_utilization": 9.0, "distance_to_liquidation": 9.0,
    "borrow_capacity_utilization": 7.0, "avg_ltv": 7.0, "risk_buffer": 6.0,
    "leverage_indicator": 5.0, "overcollateral_margin": 4.0,
    "avg_current_liquidation_threshold": 3.0, "has_sampled_borrow_position": 1.0,
    # Tier 2 — realized liquidation events
    "liquidation_rate": 6.0, "liquidation_severity_usd": 3.0,
    "liquidation_severity_eth": 2.0, "liquidation_tx_count": 2.0,
    "liquidated_collateral_value_usd": 2.0, "liquidated_collateral_value_eth": 1.0,
    "liquidation_debt_covered_value_usd": 2.0, "liquidation_debt_covered_value_eth": 1.0,
    "liquidation_volume_ratio_eth": 2.0, "liquidation_user_ratio": 1.0,
    "unique_liquidated_users": 1.0, "unique_liquidators": 1.0,
    "avg_liquidation_debt_usd": 1.0, "avg_liquidation_debt_eth": 1.0,
    "liquidator_concentration": 1.0, "as_collateral_tx_count": 1.0,
    "as_debt_tx_count": 1.0, "has_liquidation": 1.0,
    # Tier 3 — debt dynamics & repayment behavior
    "borrow_repay_ratio": 4.0, "repayment_discipline": 3.0,
    "debt_expansion_ratio": 3.0, "net_borrow_demand_usd": 2.0,
    "borrow_tx_count": 2.0, "unique_borrowers": 2.0,
    "borrow_amount_value_usd": 2.0, "borrow_amount_value_eth": 1.0,
    "repay_amount_value_usd": 2.0, "borrower_participation_rate": 2.0,
    # Tier 4 — market stress & exposure context
    "market_stress_index_usd": 3.0, "market_stress_index_eth": 2.0,
    "capital_efficiency": 2.0, "avg_total_debt_base": 1.0,
    "avg_total_collateral_base": 1.0, "avg_available_borrows_base": 0.5,
    "remaining_borrow_capacity": 0.5, "collateral_enabled_count": 1.0,
    "collateral_disabled_count": 1.0, "collateral_usage_rate": 1.0,
    "net_liquidity_flow_usd": 1.0, "supply_withdrawal_ratio": 1.0,
    "protocol_turnover_usd": 1.0, "flashloan_usage_intensity": 1.0,
    "flashloan_amount_value_usd": 1.0, "user_activity": 1.0,
}


# --- legacy aliases -------------------------------------------------------- #
# src/model_dataset.py and model_features.ipynb import the CREDIT_RISK_* names.
# Kept so nothing downstream needed editing for this move; new code should prefer
# the generic FEATURE_* names above, which are what a second domain would define.
CREDIT_RISK_COLS = FEATURE_COLS
CREDIT_RISK_WEIGHTS = FEATURE_WEIGHTS
CREDIT_RISK_PRIORITY = FEATURE_PRIORITY
LIQ_CONDITIONAL_COLS = CONDITIONAL_ZERO_COLS
