"""Model catalogue: estimator pipelines and their hyper-parameter search spaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sklearn
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:  # optional extra "boosting"
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:  # optional extra "boosting"
    LGBMClassifier = None

SKLEARN_VERSION = tuple(int(x) for x in sklearn.__version__.split(".")[:2])

MODEL_KEYS = ("dummy", "logistic_regression", "decision_tree", "random_forest",
              "extra_trees", "hist_gradient_boosting", "xgboost", "lightgbm")


@dataclass(frozen=True)
class ModelOptions:
    random_state: int = 42
    n_jobs: int = -1
    balance_classes: bool = True
    target_encoding_smoothing: float = 10.0


def preprocessor(num, cat, impute: bool, scale: bool, opts: ModelOptions) -> ColumnTransformer:
    """Imputation, scaling and target encoding, all fitted on training data only."""
    steps = []
    if impute:
        steps.append(("imp", SimpleImputer(strategy="median")))
    if scale:
        steps.append(("esc", StandardScaler()))
    blocks = []
    if num:
        blocks.append(("num", Pipeline(steps) if steps else "passthrough", num))
    if cat:
        # Category mean of the target, smoothed with m = target_encoding_smoothing;
        # scikit-learn cross-fits internally on the training data to avoid leakage.
        blocks.append(("cat", target_encoder(opts), cat))
    return ColumnTransformer(blocks)


def target_encoder(opts: ModelOptions) -> TargetEncoder:
    """Same cross-fitting folds on every scikit-learn version.

    scikit-learn >= 1.9 deprecates `random_state` in favour of a CV splitter; for a
    binary target the old default was exactly StratifiedKFold(5, shuffle=True).
    """
    if SKLEARN_VERSION >= (1, 9):
        return TargetEncoder(smooth=opts.target_encoding_smoothing,
                             cv=StratifiedKFold(5, shuffle=True, random_state=opts.random_state))
    return TargetEncoder(smooth=opts.target_encoding_smoothing, random_state=opts.random_state)


def available(name: str) -> bool:
    return not ((name == "xgboost" and XGBClassifier is None) or (name == "lightgbm" and LGBMClassifier is None))


def build_model(name: str, opts: ModelOptions, num, cat):
    """Return (Pipeline, search_space), or (None, None) when the library is missing."""
    rs, nj = opts.random_state, opts.n_jobs
    cw = "balanced" if opts.balance_classes else None

    def pipe(clf, impute=True, scale=False):
        return Pipeline([("pre", preprocessor(num, cat, impute, scale, opts)), ("clf", clf)])

    if name == "dummy":
        return Pipeline([("clf", DummyClassifier(strategy="prior"))]), {}
    if name == "logistic_regression":
        return (pipe(LogisticRegression(max_iter=2000, class_weight=cw), True, True),
                {"clf__C": loguniform(1e-3, 1e2)})
    if name == "decision_tree":
        return (pipe(DecisionTreeClassifier(max_depth=8, min_samples_leaf=50, class_weight=cw, random_state=rs)),
                {"clf__max_depth": randint(3, 14), "clf__min_samples_leaf": randint(20, 400)})
    if name == "random_forest":
        return (pipe(RandomForestClassifier(n_estimators=200, min_samples_leaf=10, max_samples=0.5,
                                            n_jobs=nj, class_weight=cw, random_state=rs)),
                {"clf__min_samples_leaf": randint(3, 60), "clf__max_features": uniform(0.2, 0.6)})
    if name == "extra_trees":
        return (pipe(ExtraTreesClassifier(n_estimators=200, min_samples_leaf=10, max_samples=0.5,
                                          bootstrap=True, n_jobs=nj, class_weight=cw, random_state=rs)),
                {"clf__min_samples_leaf": randint(3, 60), "clf__max_features": uniform(0.2, 0.6)})
    if name == "hist_gradient_boosting":
        return (pipe(HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, class_weight=cw,
                                                    random_state=rs), impute=False),
                {"clf__learning_rate": loguniform(0.02, 0.3), "clf__max_leaf_nodes": randint(15, 127),
                 "clf__min_samples_leaf": randint(20, 300), "clf__l2_regularization": loguniform(1e-3, 10),
                 "clf__max_iter": randint(150, 500)})
    if name == "xgboost":
        if XGBClassifier is None:
            return None, None
        return (pipe(XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
                                   colsample_bytree=0.8, tree_method="hist", eval_metric="logloss",
                                   scale_pos_weight=1.0, n_jobs=nj, random_state=rs), impute=False),
                {"clf__n_estimators": randint(200, 700), "clf__learning_rate": loguniform(0.02, 0.2),
                 "clf__max_depth": randint(3, 10), "clf__subsample": uniform(0.6, 0.4),
                 "clf__colsample_bytree": uniform(0.5, 0.5), "clf__min_child_weight": randint(1, 30),
                 "clf__reg_lambda": loguniform(0.1, 20)})
    if name == "lightgbm":
        if LGBMClassifier is None:
            return None, None
        return (pipe(LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8,
                                    subsample_freq=1, colsample_bytree=0.8, scale_pos_weight=1.0,
                                    n_jobs=nj, random_state=rs, verbose=-1), impute=False),
                {"clf__n_estimators": randint(200, 700), "clf__learning_rate": loguniform(0.02, 0.2),
                 "clf__num_leaves": randint(15, 150), "clf__min_child_samples": randint(20, 300),
                 "clf__colsample_bytree": uniform(0.5, 0.5), "clf__reg_lambda": loguniform(0.1, 20)})
    raise ValueError(f"Unknown model: {name}")


def set_class_weight(est, y, opts: ModelOptions) -> None:
    """Boosting has no class_weight: use scale_pos_weight = negatives / positives."""
    key = "clf__scale_pos_weight"
    if opts.balance_classes and key in est.get_params():
        pos = max(int(np.sum(y)), 1)
        est.set_params(**{key: (len(y) - pos) / pos})
