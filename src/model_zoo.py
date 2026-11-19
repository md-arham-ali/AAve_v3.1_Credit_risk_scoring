"""Modeling stage — model registry, house defaults, tuning grids, loss options, and build/score helpers.

Defaults live here. The notebook changes a value by naming it (fixed=, grid=, loss=, or a keyword
for the LSTM / Transformer / Cox / IsolationForest runners) and never edits this file.
show_params(name) prints what a model takes.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import difflib
import inspect
import itertools
import json

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge, Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (AdaBoostClassifier, RandomForestClassifier, RandomForestRegressor,
                              ExtraTreesClassifier, ExtraTreesRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

# optional (like lifelines in model_training): without it the catboost entry simply does not exist
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from model_config import RANDOM_STATE, rank_normalize


# =========================================================================== #
# Models and house defaults (initialization: stays in this file)
# =========================================================================== #

def _factory(cls, **defaults):
    """Model factory whose overrides REPLACE defaults (tunable without kw clashes).
    The class and the house defaults ride on the factory, so show_params can print them."""
    def make(**kw):
        return cls(**{**defaults, **kw})
    make.cls, make.defaults = cls, defaults
    return make


CLASSIFIER_SPECS = {
    "logistic_regression": {"scaled": True, "factory": _factory(LogisticRegression,
        class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE)},
    # elastic net = logistic regression with a mixed L1 / L2 penalty (l1_ratio); saga is the solver that takes it
    "elastic_net": {"scaled": True, "factory": _factory(LogisticRegression,
        l1_ratio=0.5, solver="saga", class_weight="balanced", max_iter=5000,
        random_state=RANDOM_STATE)},
    "ridge_classifier": {"scaled": True, "factory": _factory(RidgeClassifier,
        class_weight="balanced", random_state=RANDOM_STATE)},
    "gaussian_nb": {"scaled": False, "factory": _factory(GaussianNB)},
    "kneighbors": {"scaled": True, "factory": _factory(KNeighborsClassifier,
        n_neighbors=7)},
    "svc_rbf": {"scaled": True, "factory": _factory(SVC,
        kernel="rbf", probability=True, class_weight="balanced",
        random_state=RANDOM_STATE)},
    "decision_tree": {"scaled": False, "factory": _factory(DecisionTreeClassifier,
        max_depth=4, min_samples_leaf=12, class_weight="balanced",
        random_state=RANDOM_STATE)},
    "random_forest": {"scaled": False, "factory": _factory(RandomForestClassifier,
        n_estimators=400, class_weight="balanced", random_state=RANDOM_STATE,
        n_jobs=-1)},
    # stumps as the weak learner; class_weight sits on the stump because AdaBoost has none of its own
    "adaboost": {"scaled": False, "factory": _factory(AdaBoostClassifier,
        estimator=DecisionTreeClassifier(max_depth=1, class_weight="balanced",
                                         random_state=RANDOM_STATE),
        n_estimators=200, learning_rate=0.5, random_state=RANDOM_STATE)},
    "extra_trees": {"scaled": False, "factory": _factory(ExtraTreesClassifier,
        n_estimators=400, class_weight="balanced", random_state=RANDOM_STATE,
        n_jobs=-1)},
    "hist_gradient_boosting": {"scaled": False,
                               "factory": _factory(HistGradientBoostingClassifier,
        class_weight="balanced", random_state=RANDOM_STATE)},
    "xgboost": {"scaled": False, "factory": _factory(XGBClassifier,
        n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9,
        colsample_bytree=0.9, eval_metric="logloss", random_state=RANDOM_STATE,
        n_jobs=-1)},
    "lightgbm": {"scaled": False, "factory": _factory(LGBMClassifier,
        n_estimators=400, learning_rate=0.05, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)},
    "mlp": {"scaled": True, "factory": _factory(MLPClassifier,
        hidden_layer_sizes=(32, 16), alpha=1e-3, max_iter=2000, early_stopping=True,
        random_state=RANDOM_STATE)},
}
if CATBOOST_AVAILABLE:
    # verbose / allow_writing_files: CatBoost prints every round and writes a catboost_info/ folder by default
    CLASSIFIER_SPECS["catboost"] = {"scaled": False, "factory": _factory(CatBoostClassifier,
        iterations=400, depth=4, learning_rate=0.05, random_seed=RANDOM_STATE, thread_count=-1,
        verbose=0, allow_writing_files=False)}


REGRESSOR_SPECS = {
    "ridge": {"scaled": True, "factory": _factory(Ridge,
        alpha=1.0, random_state=RANDOM_STATE)},
    "lasso": {"scaled": True, "factory": _factory(Lasso,
        alpha=0.01, max_iter=50000, random_state=RANDOM_STATE)},
    "kneighbors": {"scaled": True, "factory": _factory(KNeighborsRegressor,
        n_neighbors=7)},
    "svr_rbf": {"scaled": True, "factory": _factory(SVR, kernel="rbf", C=10.0)},
    "random_forest": {"scaled": False, "factory": _factory(RandomForestRegressor,
        n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1)},
    "extra_trees": {"scaled": False, "factory": _factory(ExtraTreesRegressor,
        n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1)},
    "hist_gradient_boosting": {"scaled": False,
                               "factory": _factory(HistGradientBoostingRegressor,
        random_state=RANDOM_STATE)},
    "xgboost": {"scaled": False, "factory": _factory(XGBRegressor,
        n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9,
        colsample_bytree=0.9, random_state=RANDOM_STATE, n_jobs=-1)},
    "lightgbm": {"scaled": False, "factory": _factory(LGBMRegressor,
        n_estimators=400, learning_rate=0.05, random_state=RANDOM_STATE,
        n_jobs=-1, verbosity=-1)},
}


# The runners that are not sklearn-style models. Same rule as the specs above: the default lives
# here, and the notebook names a key to change it (run_lstm(data, epochs=40, lr=3e-4)).
_SEQUENCE_TRAINING = {
    "dropout": 0.1, "epochs": 30, "batch_size": 256, "patience": 5,
    "lr": 1e-3, "optimizer": "Adam", "optimizer_params": {},       # any name in torch.optim
    "scheduler": None, "scheduler_params": {},                      # any per-epoch torch.optim.lr_scheduler
    "warmup_epochs": 0,                                             # linear ramp; 0 = off
    "loss": "bce", "loss_params": {},                               # "bce" or "focal" ({"gamma": 2.0})
}
RUNNER_SPECS = {
    "lstm": {"lookback": 84, "hidden": 32, **_SEQUENCE_TRAINING},
    "transformer": {"lookback": 84, "d_model": 32, "n_heads": 4, "n_layers": 2,
                    **_SEQUENCE_TRAINING},
    "cox_ph": {"penalizer": 0.10, "l1_ratio": 0.0, "top_n_features": 15},
    "isolation_forest": {"n_estimators": 400, "contamination": "auto", "max_samples": "auto",
                         "max_features": 1.0},
    # series models (model_series.py). sarimax: the seasonal part steps one HORIZON back (period None =
    # the horizon in rows), because the public label series is made of overlapping windows
    "sarimax": {"order": (0, 0, 0), "seasonal_order": (1, 0, 0), "period": None,
                "enforce_stationarity": True, "enforce_invertibility": True, "concentrate_scale": False,
                "method": "lbfgs", "maxiter": 50},
    "hmm_regime": {"n_components": 3, "covariance_type": "diag", "n_iter": 100, "tol": 1e-3,
                   "n_init": 3, "min_covar": 1e-3},
}

# Runner settings that are OFF unless the notebook names them. They are kept out of RUNNER_SPECS so a
# run that does not use them keeps the settings (and settings_hash) it always had.
_SEQUENCE_OPTIONAL = {
    "n_seeds": 1,            # train this many nets (seeds RANDOM_STATE, +1, ...) and average their scores
    "stop": "val",           # early stopping of the real fit: "val", or "train_tail" (last 15% of train, embargoed)
    "clip_norm": None,       # gradient-norm clipping; None = off
    # 2026-11-16 (round two). All off unless named, so a run that does not use them keeps its settings_hash.
    "ema": None,             # weight averaging: decay per BATCH (0.99 = about the last 4 epochs); None = off. Scored and stopped on the averaged net
    "input_noise": 0.0,      # Gaussian noise (in scaled units) added to every training window; 0 = off
    "stop_metric": "loss",   # early stopping watches the stop rows' "loss" (as before) or their ranking quality "auc"
}
RUNNER_OPTIONAL = {"lstm": {**_SEQUENCE_OPTIONAL, "n_layers": 1},
                   "transformer": {**_SEQUENCE_OPTIONAL, "pool": "last", "ff_mult": 2},
                   # SARIMAX: an observation-noise term next to the state noise (off in the house model). A trend term is
                   # NOT offered: statsmodels would then add an intercept to the regressors and _sarimax_scores reads
                   # the exogenous coefficients by name, so the forecast would be built from the wrong vector
                   "sarimax": {"measurement_error": False},
                   # HMM: Gaussian-mixture emissions per state, and the Dirichlet / covariance priors that regularise EM
                   "hmm_regime": {"emission": "gaussian", "n_mix": 2, "transmat_prior": 1.0, "startprob_prior": 1.0,
                                  "covars_prior": None, "covars_weight": None}}


# =========================================================================== #
# Default tuning grids — small on purpose, widened from the notebook at runtime
# =========================================================================== #

# A dict tries every combination of its LIST values. A list of dicts pairs values that must move
# together (tree count with learning rate). See expand_grid.
CLASSIFIER_GRIDS = {
    "logistic_regression": {"C": [0.01, 0.1, 1.0]},
    "elastic_net": {"C": [0.01, 0.1, 1.0], "l1_ratio": [0.2, 0.5, 0.8]},
    "ridge_classifier": {"alpha": [1.0, 10.0, 100.0]},
    "gaussian_nb": {"var_smoothing": [1e-9, 1e-7, 1e-5]},
    "kneighbors": {"n_neighbors": [25, 50, 100]},
    "svc_rbf": {"C": [0.01, 0.1, 1.0]},
    # leaf 12 / 48 rows = 1 / 4 days at 2h: a smaller leaf just memorises one stress episode
    "decision_tree": {"max_depth": [3, 5], "min_samples_leaf": [12, 48]},
    "random_forest": {"max_depth": [6, None], "min_samples_leaf": [12, 48]},
    "adaboost": {"n_estimators": [100, 300], "learning_rate": [0.1, 0.5]},
    "extra_trees": {"max_depth": [6, None], "min_samples_leaf": [12, 48]},
    "hist_gradient_boosting": {"max_depth": [3, 5], "l2_regularization": [0.0, 1.0]},
    "xgboost": {"max_depth": [3, 5], "reg_lambda": [1.0, 10.0]},
    "catboost": {"depth": [2, 4], "l2_leaf_reg": [3.0, 30.0]},
    "lightgbm": {"num_leaves": [15, 31], "min_child_samples": [12, 48]},
    "mlp": {"alpha": [1e-4, 1e-3, 1e-2]},
}


REGRESSOR_GRIDS = {
    "ridge": {"alpha": [1.0, 10.0, 100.0]},
    # lasso alpha and SVR C/epsilon scale with y; moved up 10x when the target went from
    # log1p(score) (std 1.2) to the score (std 18.5). Ridge alpha and the trees do not depend on y's scale.
    "lasso": {"alpha": [0.1, 1.0, 10.0]},
    "kneighbors": {"n_neighbors": [25, 50, 100]},
    "svr_rbf": {"C": [10.0, 100.0], "epsilon": [1.0, 5.0]},
    "random_forest": {"max_depth": [6, None], "min_samples_leaf": [12, 48]},
    "extra_trees": {"max_depth": [6, None], "min_samples_leaf": [12, 48]},
    "hist_gradient_boosting": {"max_depth": [3, 5], "l2_regularization": [0.0, 1.0]},
    "xgboost": {"max_depth": [3, 5], "reg_lambda": [1.0, 10.0]},
    "lightgbm": {"num_leaves": [15, 31], "min_child_samples": [12, 48]},
}


RUNNER_GRIDS = {
    "lstm": {"dropout": [0.1, 0.3]},
    "transformer": {"dropout": [0.1, 0.3]},
    "cox_ph": {"penalizer": [0.01, 0.1, 1.0]},
    "isolation_forest": {"max_features": [0.5, 1.0]},
    "sarimax": {"seasonal_order": [(1, 0, 0), (1, 0, 1), (2, 0, 0)]},   # a tuple is ONE value
    "hmm_regime": {"n_components": [2, 3, 4]},
}


# =========================================================================== #
# Picking the winner of a tuning run
# =========================================================================== #

# "one_se" needs to know which config is the SIMPLER one. "low" = a lower value is simpler
# (shallower tree, smaller C); "high" = a higher value is simpler (bigger leaf, stronger
# penalty, more neighbours). Compared in the listed order. None counts as "no limit" (inf).
SIMPLICITY = {
    "logistic_regression": [("C", "low")],
    "elastic_net": [("C", "low"), ("l1_ratio", "high")],
    "ridge_classifier": [("alpha", "high")],
    "ridge": [("alpha", "high")],
    "lasso": [("alpha", "high")],
    "gaussian_nb": [("var_smoothing", "high")],
    "kneighbors": [("n_neighbors", "high")],
    "svc_rbf": [("C", "low")],
    "svr_rbf": [("C", "low"), ("epsilon", "high")],
    "decision_tree": [("max_depth", "low"), ("min_samples_leaf", "high")],
    "adaboost": [("n_estimators", "low"), ("learning_rate", "low")],
    "random_forest": [("max_depth", "low"), ("min_samples_leaf", "high"), ("n_estimators", "low")],
    "extra_trees": [("max_depth", "low"), ("min_samples_leaf", "high"), ("n_estimators", "low")],
    "hist_gradient_boosting": [("max_depth", "low"), ("l2_regularization", "high"),
                               ("max_leaf_nodes", "low")],
    "xgboost": [("max_depth", "low"), ("reg_lambda", "high"), ("n_estimators", "low")],
    "catboost": [("depth", "low"), ("l2_leaf_reg", "high"), ("iterations", "low")],
    "lightgbm": [("num_leaves", "low"), ("min_child_samples", "high"), ("n_estimators", "low")],
    "mlp": [("alpha", "high")],
    "lstm": [("dropout", "high"), ("hidden", "low")],
    "transformer": [("dropout", "high"), ("d_model", "low")],
    "cox_ph": [("penalizer", "high"), ("top_n_features", "low")],
    "isolation_forest": [("max_features", "low")],
    "hmm_regime": [("n_components", "low")],
}

# The rule each model is picked by unless the notebook passes select=. Everything not listed: "one_se".
# Sequence models swing the most between folds, so their rule rewards the stable config.
SELECT_DEFAULT = {"lstm": "mean_minus_std", "transformer": "mean_minus_std",
                  "isolation_forest": "best", "sarimax": "best"}


def default_select(name):
    """The selection rule a model uses when the notebook does not pass select=."""
    return SELECT_DEFAULT.get(name, "one_se")


# =========================================================================== #
# Loss options — loss=None keeps each library's own loss (today's behaviour)
# =========================================================================== #

# A name maps to the parameters that set it. {} = the model's only loss, nothing to set.
# The loss can also go in a grid (e.g. {"criterion": ["gini", "entropy"]}) and be picked on CV.
_TREE_CLF_LOSSES = {"gini": {"criterion": "gini"}, "entropy": {"criterion": "entropy"},
                    "log_loss": {"criterion": "log_loss"}}
CLASSIFIER_LOSSES = {
    "logistic_regression": {"log_loss": {}},
    "elastic_net": {"log_loss": {}},
    "ridge_classifier": {"squared_error": {}},
    "gaussian_nb": {},                                  # no loss: fitted by counting
    "kneighbors": {},                                   # no loss: votes of neighbours
    "svc_rbf": {"hinge": {}},
    "decision_tree": _TREE_CLF_LOSSES,
    "random_forest": _TREE_CLF_LOSSES,
    "adaboost": {},                                     # no loss to choose: SAMME re-weighting
    "extra_trees": _TREE_CLF_LOSSES,
    "hist_gradient_boosting": {"log_loss": {"loss": "log_loss"}},
    "xgboost": {"logistic": {"objective": "binary:logistic"}},
    "catboost": {"logloss": {"loss_function": "Logloss"}, "cross_entropy": {"loss_function": "CrossEntropy"}},
    "lightgbm": {"binary": {"objective": "binary"}, "cross_entropy": {"objective": "cross_entropy"}},
    "mlp": {"log_loss": {}},
}

# absolute_error on random_forest / extra_trees is slow (much slower than squared_error).
# huber's width is on the score scale (0-100, std 18.5): xgboost huber_slope and lightgbm alpha
# both default to ~1, which makes huber almost absolute_error here. Set them through fixed=.
_TREE_REG_LOSSES = {"squared_error": {"criterion": "squared_error"},
                    "absolute_error": {"criterion": "absolute_error"},
                    "friedman_mse": {"criterion": "friedman_mse"},
                    "poisson": {"criterion": "poisson"}}
REGRESSOR_LOSSES = {
    "ridge": {"squared_error": {}},
    "lasso": {"squared_error": {}},
    "kneighbors": {},
    "svr_rbf": {"epsilon_insensitive": {}},
    "random_forest": _TREE_REG_LOSSES,
    "extra_trees": _TREE_REG_LOSSES,
    "hist_gradient_boosting": {"squared_error": {"loss": "squared_error"},
                               "absolute_error": {"loss": "absolute_error"},
                               "poisson": {"loss": "poisson"}},
    "xgboost": {"squared_error": {"objective": "reg:squarederror"},
                "absolute_error": {"objective": "reg:absoluteerror"},
                "huber": {"objective": "reg:pseudohubererror"}},
    "lightgbm": {"squared_error": {"objective": "regression"},
                 "absolute_error": {"objective": "regression_l1"},
                 "huber": {"objective": "huber"},
                 "poisson": {"objective": "poisson"}},
}

SEQUENCE_LOSSES = ("bce", "focal", "pairwise")        # pairwise = a logistic AUC surrogate over every (positive, negative) pair of a batch


def loss_params(name, kind, loss):
    """The parameters that set `loss` on a registry model.
    loss=None -> {} (the library's own loss). Unknown loss -> ValueError naming the options."""
    if loss is None:
        return {}
    options = (CLASSIFIER_LOSSES if kind == "classifier" else REGRESSOR_LOSSES).get(name, {})
    if loss not in options:
        raise ValueError(f"{name} ({kind}): no loss named {loss!r}. "
                         f"Options: {sorted(options) or 'none, this model has no loss to choose'}")
    return dict(options[loss])


# =========================================================================== #
# Building, checking and showing parameters
# =========================================================================== #

def _specs(kind):
    return CLASSIFIER_SPECS if kind == "classifier" else REGRESSOR_SPECS


def build_model(name, kind="classifier", **overrides):
    """Instantiate a registry model with house defaults (+ tuned overrides)."""
    # every spec gets RANDOM_STATE where it accepts one, so runs are reproducible
    return _specs(kind)[name]["factory"](**overrides)


def expand_grid(grid):
    """Grid -> list of configs.
    A dict tries every combination of its LIST values; a tuple or a scalar is ONE value, so
    hidden_layer_sizes=(32, 16) stays whole. A list of dicts is expanded dict by dict, which pairs
    values that must move together:
        [{"n_estimators": 400, "learning_rate": 0.05}, {"n_estimators": 800, "learning_rate": 0.025}]
    None or {} -> [{}], one config: the defaults."""
    if not grid:
        return [{}]
    out = []
    for part in (grid if isinstance(grid, list) else [grid]):
        keys = list(part)
        values = [v if isinstance(v, list) else [v] for v in part.values()]
        out += [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    return out


def grid_around(config, rows):
    """A stage that tries `rows` against a config already in hand (round two, 2026-11-16).
    config: the settings of the option being challenged, e.g. res["config_by_target"]["y_stress_..._24h"].
    rows: list of dicts, each one option to try (the keys it sets replace the config's own).
    Every key any row names is taken out of the fixed part (fixed wins over a grid), and a first row spells
    out the config's own values for those keys, so the challenged option is scored on the same folds as
    the rows: it is the reference row of mev.paired_folds. Keys the config lacks stay at the house default
    in the reference row, which is what the config used.
    Returns: {"fixed": dict, "grid": list of dicts, "reference": the first grid row} -> run_*(fixed=..., grid=...) or
    run_lstm(**fixed, grid=...); mev.vs_reference(res["tuning"], ["reference"]) reads the table against it."""
    # a LIST value in a grid is a set of options (expand_grid), a tuple is one value: a config read back from a
    # result holds SARIMAX's order as a list, so it must become a tuple here or the row would split into three
    whole = lambda v: tuple(v) if isinstance(v, list) else v
    keys = {k for r in rows for k in r}
    reference = {k: whole(config[k]) for k in keys if k in config}
    rows = [{k: whole(v) for k, v in r.items()} for r in rows]
    return {"fixed": {k: v for k, v in config.items() if k not in keys},
            "grid": [reference] + [dict(r) for r in rows], "reference": reference}


def drop_keys(config, keys):
    """A config without the named keys: frees them for the next grid, because a value in fixed=
    wins over the grid. drop_keys(XGB_FIXED_3, ["learning_rate", "n_estimators"]).
    Returns: a new dict."""
    return {k: v for k, v in config.items() if k not in set(keys)}


def sample_grid(space, n, seed=RANDOM_STATE, where=None):
    """Random search: n configs drawn without replacement from every combination of `space`'s
    LIST values (a tuple or a scalar is one value, as in expand_grid). Same space + n + seed = same configs.
    where: optional filter, config -> bool, to leave out combinations that cannot work
        (a leaf bigger than the rows a small max_samples gives a tree); the draw is from what passes.
    Returns: a list of dicts in grid order, to pass as grid= (scored by the same tuner and table)."""
    if not isinstance(space, dict):
        raise TypeError("sample_grid takes one dict of lists: {param: [values], ...}")
    nested = sorted(k for k, v in space.items() if isinstance(v, list) and any(isinstance(x, list) for x in v))
    if nested:
        raise ValueError(f"sample_grid: lists inside the value list of {nested} would be expanded again; "
                         "use a tuple for a value that is itself a sequence")
    # a value listed twice would only make the same config twice as likely
    space = {k: list({json.dumps(x, sort_keys=True, default=str): x for x in v}.values())
             if isinstance(v, list) else v for k, v in space.items()}
    configs = expand_grid(space)
    if where is not None:
        configs = [c for c in configs if where(c)]
    if n < 1 or not configs:
        raise ValueError(f"sample_grid: nothing to draw (n={n}, {len(configs)} config(s) in the space)")
    if n >= len(configs):
        print(f" sample_grid: all {len(configs)} config(s) of the space (n={n})")
        return configs
    pick = np.sort(np.random.default_rng(seed).choice(len(configs), size=n, replace=False))
    print(f" sample_grid: {n} of {len(configs)} config(s), seed {seed}")
    # numpy scalars (a space built with np.arange) as Python numbers: one_se ranks only real ints / floats
    return [{k: (v.item() if isinstance(v, np.generic) else v) for k, v in configs[int(i)].items()} for i in pick]


# xgboost booster parameters that its sklearn wrapper passes through but get_params() does not list.
# xgboost has no public list of them, so only the ones the loss options need are named here.
_XGB_NATIVE = {"huber_slope", "quantile_alpha", "tweedie_variance_power",
               # 2026-11-16: dart and gblinear booster settings, and the ranking objectives' pair settings
               "rate_drop", "skip_drop", "sample_type", "normalize_type", "one_drop",
               "feature_selector", "top_k", "updater", "lambdarank_pair_method", "lambdarank_num_pair_per_sample"}


def _lightgbm_native():
    """LightGBM's own parameter names and aliases (huber's alpha, max_bin, ...): get_params() does not
    list them. Read from lightgbm's alias table, a private API, so an empty set if it ever moves."""
    try:
        from lightgbm.basic import _ConfigAliases
        return {n for names in _ConfigAliases._get_all_param_aliases().values() for n in names}
    except Exception:                                   # noqa: BLE001 - fall back to get_params only
        return set()


def accepted_names(name, kind="classifier"):
    """Every parameter name the model takes: get_params() of the house model (+ the library's native
    names for xgboost / lightgbm), or the runner's spec."""
    if name in RUNNER_SPECS:
        return set(RUNNER_SPECS[name]) | set(RUNNER_OPTIONAL.get(name, {}))
    # nested names (AdaBoost's estimator__max_depth) are not constructor arguments, so build_model
    # cannot take them: leaving them out stops the run at the name check, not deep inside CV
    names = {n for n in build_model(name, kind).get_params() if "__" not in n}
    if name == "xgboost":
        names |= _XGB_NATIVE
    if name == "lightgbm":
        names |= _lightgbm_native()
    if name == "catboost":
        # CatBoost's get_params() lists only what was set: its constructor signature has every name
        names |= set(inspect.signature(CatBoostClassifier.__init__).parameters) - {"self"}
    return names


def check_names(name, kind, *settings):
    """Stop the run when a setting names a parameter the model does not take.
    xgboost and lightgbm accept unknown names silently and the notebook hides warnings,
    so without this a typo like max_dpeth=8 does nothing and nobody sees it.
    settings: dicts (fixed values, grid configs). Raises ValueError with close matches."""
    known = accepted_names(name, kind)
    bad = sorted({k for s in settings for k in (s or {}) if k not in known})
    if bad:
        hints = {b: difflib.get_close_matches(b, known, n=2) for b in bad}
        raise ValueError(f"{name} ({kind}): unknown parameter(s) "
                         + ", ".join(f"{b!r}" + (f" (did you mean {h}?)" if h else "")
                                     for b, h in hints.items())
                         + f". show_params({name!r}) lists the names.")


def _plain(value):
    """A parameter value as something JSON and MLflow can hold."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return str(value)


def model_settings(model):
    """Every setting a fitted registry model used: get_params(), house defaults included.
    Returns: a plain dict, safe for JSON."""
    return {k: _plain(v) for k, v in sorted(model.get_params().items())}


# the name each library gives its random seed, where it is not random_state
SEED_ARG = {"catboost": "random_seed"}

# settings the runners fill themselves at fit time: a house value here would switch that off
_RUNNER_SET = {"xgboost": {"scale_pos_weight"}}


def _is_plain(value):
    """None, bool, numbers, strings, and tuples / lists / dicts of those: safe in a config and in JSON."""
    if value is None or isinstance(value, (bool, int, float, str, np.integer, np.floating)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_plain(v) for v in value)
    if isinstance(value, dict):
        return all(_is_plain(v) for v in value.values())
    return False


def fill_defaults(name, configs, kind="classifier"):
    """Finalist configs that all name the same keys: a key a config left out gets the house value,
    which the model uses anyway. Same fits and settings_hash for every registry setting; the tuning table
    stops showing None for 'house default', and one_se ranks the real values (a missing max_depth is not
    'simplest'). A setting the runner fills itself (xgboost scale_pos_weight) must be in every config or none.
    configs: a dict or a list of dicts. Returns: a list of new dicts."""
    configs = configs if isinstance(configs, list) else [configs]
    check_names(name, kind, *configs)
    keys = list(dict.fromkeys(k for c in configs for k in c))
    missing = [k for k in keys if any(k not in c for c in configs)]
    auto = [k for k in missing if k in _RUNNER_SET.get(name, set())]
    if auto:
        raise ValueError(f"{name}: {auto} is filled by the runner from the train labels; set it in every config or none")
    house = build_model(name, kind).get_params()
    no_house = [k for k in missing if k not in house]
    if no_house:
        raise ValueError(f"{name}: {no_house} have no house value in get_params(): set them in every config")
    objects = [k for k in missing if not _is_plain(house[k])]
    if objects:
        raise ValueError(f"{name}: the house value of {objects} is an object, set it in every config by hand")
    return [{k: (c[k] if k in c else house[k]) for k in keys} for c in configs]


def show_params(name, kind="classifier"):
    """Print what a model takes: house defaults, default grid, selection rule, loss options and
    every accepted name. kind is ignored for lstm / transformer / cox_ph / isolation_forest.
    Returns: the same as a dict."""
    if name in RUNNER_SPECS:
        defaults, grid = RUNNER_SPECS[name], RUNNER_GRIDS.get(name)
        losses = list(SEQUENCE_LOSSES) if name in ("lstm", "transformer") else []
        names = sorted(defaults)
    else:
        if name not in _specs(kind):
            raise ValueError(f"no {kind} named {name!r}. Options: {sorted(_specs(kind))}")
        defaults = _specs(kind)[name]["factory"].defaults
        grid = (CLASSIFIER_GRIDS if kind == "classifier" else REGRESSOR_GRIDS).get(name)
        losses = sorted((CLASSIFIER_LOSSES if kind == "classifier" else REGRESSOR_LOSSES).get(name, {}))
        names = sorted(accepted_names(name, kind))
    info = {"name": name, "kind": kind if name not in RUNNER_SPECS else "runner",
            "house_defaults": defaults, "default_grid": grid,
            "grid_configs": len(expand_grid(grid)), "select": default_select(name),
            "simplicity": SIMPLICITY.get(name, []), "losses": losses, "accepted_names": names}
    print(f" {name} ({info['kind']})")
    print(f"   house defaults : {defaults or 'library defaults only'}")
    print(f"   default grid   : {grid or 'none'} -> {info['grid_configs']} config(s)")
    print(f"   select         : {info['select']}  (simpler = {info['simplicity'] or 'n/a'})")
    print(f"   losses         : {losses or 'none to choose'}")
    print(f"   accepted names : {', '.join(names)}")
    return info


# =========================================================================== #
# Scoring helpers
# =========================================================================== #

def _xgb_overrides(y_train):
    """scale_pos_weight = n_neg / n_pos computed from the TRAIN labels."""
    # NOTE: xgboost renames params between majors — this is the compatibility shim
    pos = int(np.sum(y_train))
    neg = int(len(y_train) - pos)
    return {"scale_pos_weight": (neg / pos) if pos else 1.0}


def predict_scores(model, X, X_train=None):
    """Probability-like score in [0, 1] for any classifier.
    model, X, plus TRAIN decision scores for the rank-normalize fallback.
    Returns: np.ndarray in [0, 1] — uncalibrated when predict_proba is absent."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    raw = model.decision_function(X)
    ref = model.decision_function(X_train) if X_train is not None else raw
    return rank_normalize(raw, ref)


def score_is_probability(model):
    """True when the model emits calibrated-ish probabilities (Brier meaningful)."""
    # gates the Brier metric; rank scores must not be scored as probabilities
    return hasattr(model, "predict_proba")
