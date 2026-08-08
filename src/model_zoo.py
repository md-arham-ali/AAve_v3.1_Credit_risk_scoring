"""Modeling stage — the model zoo: registry, tuning grids, build + score helpers.

Declarative registry of the 11 classifiers and 9 regressors (house defaults +
small walk-forward tuning grids) plus the factory and the score helpers
(``build_model``, ``predict_scores``, ``score_is_probability``, ``_xgb_overrides``).
Consumed by model_evaluation (CV/tuning) and model_training (fitting).
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge, Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              ExtraTreesClassifier, ExtraTreesRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor)
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from model_config import RANDOM_STATE, rank_normalize


def _factory(cls, **defaults):
    """Model factory whose overrides REPLACE defaults (tunable without kw clashes)."""
    return lambda **kw: cls(**{**defaults, **kw})


CLASSIFIER_SPECS = {
    "logistic_regression": {"scaled": True, "factory": _factory(LogisticRegression,
        class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE)},
    "ridge_classifier": {"scaled": True, "factory": _factory(RidgeClassifier,
        class_weight="balanced", random_state=RANDOM_STATE)},
    "gaussian_nb": {"scaled": False, "factory": _factory(GaussianNB)},
    "kneighbors": {"scaled": True, "factory": _factory(KNeighborsClassifier,
        n_neighbors=7)},
    "svc_rbf": {"scaled": True, "factory": _factory(SVC,
        kernel="rbf", probability=True, class_weight="balanced",
        random_state=RANDOM_STATE)},
    "random_forest": {"scaled": False, "factory": _factory(RandomForestClassifier,
        n_estimators=400, class_weight="balanced", random_state=RANDOM_STATE,
        n_jobs=-1)},
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


CLASSIFIER_GRIDS = {
    "logistic_regression": [{"C": c} for c in (0.1, 1.0, 10.0)],
    "ridge_classifier": [{"alpha": a} for a in (0.1, 1.0, 10.0)],
    "gaussian_nb": [{"var_smoothing": v} for v in (1e-9, 1e-7, 1e-5)],
    "kneighbors": [{"n_neighbors": k} for k in (5, 9, 15)],
    "svc_rbf": [{"C": c} for c in (0.5, 2.0, 8.0)],
    "random_forest": [{"max_depth": d, "min_samples_leaf": l}
                      for d in (None, 6) for l in (1, 5)],
    "extra_trees": [{"max_depth": d, "min_samples_leaf": l}
                    for d in (None, 6) for l in (1, 5)],
    "hist_gradient_boosting": [{"learning_rate": lr, "max_leaf_nodes": n}
                               for lr in (0.05, 0.1) for n in (7, 31)],
    "xgboost": [{"max_depth": d, "learning_rate": lr}
                for d in (2, 4) for lr in (0.03, 0.05)],
    "lightgbm": [{"max_depth": d, "learning_rate": lr}
                 for d in (3, -1) for lr in (0.03, 0.05)],
    "mlp": [{"alpha": a, "hidden_layer_sizes": h}
            for a in (1e-3, 1e-2) for h in ((32, 16), (64,))],
}


REGRESSOR_GRIDS = {
    "ridge": [{"alpha": a} for a in (1.0, 10.0, 100.0)],
    "lasso": [{"alpha": a} for a in (0.01, 0.1, 1.0)],
    "kneighbors": [{"n_neighbors": k} for k in (5, 9, 15)],
    "svr_rbf": [{"C": c, "epsilon": e} for c in (1.0, 10.0) for e in (0.1, 0.5)],
    "random_forest": [{"max_depth": d, "min_samples_leaf": l}
                      for d in (None, 6) for l in (1, 5)],
    "extra_trees": [{"max_depth": d, "min_samples_leaf": l}
                    for d in (None, 6) for l in (1, 5)],
    "hist_gradient_boosting": [{"learning_rate": lr, "max_leaf_nodes": n}
                               for lr in (0.05, 0.1) for n in (7, 31)],
    "xgboost": [{"max_depth": d, "learning_rate": lr}
                for d in (2, 4) for lr in (0.03, 0.05)],
    "lightgbm": [{"max_depth": d, "learning_rate": lr}
                 for d in (3, -1) for lr in (0.03, 0.05)],
}


def build_model(name, kind="classifier", **overrides):
    """Instantiate a registry model with house defaults (+ tuned overrides)."""
    specs = CLASSIFIER_SPECS if kind == "classifier" else REGRESSOR_SPECS
    return specs[name]["factory"](**overrides)


def _xgb_overrides(y_train):
    """scale_pos_weight = n_neg / n_pos computed from the TRAIN labels."""
    pos = int(np.sum(y_train))
    neg = int(len(y_train) - pos)
    return {"scale_pos_weight": (neg / pos) if pos else 1.0}


def predict_scores(model, X, X_train=None):
    """Probability-like score in [0, 1] for any classifier.

    predict_proba when available; otherwise the decision_function rank-normalized
    against the TRAIN decision scores (comparable across splits — not the previous
    within-split ranks). Callers must treat rank scores as uncalibrated.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    raw = model.decision_function(X)
    ref = model.decision_function(X_train) if X_train is not None else raw
    return rank_normalize(raw, ref)


def score_is_probability(model):
    """True when the model emits calibrated-ish probabilities (Brier meaningful)."""
    return hasattr(model, "predict_proba")
