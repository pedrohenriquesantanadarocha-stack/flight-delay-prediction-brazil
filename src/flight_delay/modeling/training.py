"""Hyper-parameter search, grouped cross-validation and the per-model training loop."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

from ..config import ModelingConfig
from ..evaluation.thresholds import candidate_thresholds
from .catalog import ModelOptions, build_model, set_class_weight
from .split import cv_splitter

log = logging.getLogger(__name__)


@dataclass
class ModelResult:
    name: str
    model: object
    params: dict
    cv: pd.DataFrame
    thresholds: dict
    p_test: np.ndarray
    fit_seconds: float
    predict_seconds: float
    tuning_score: float = float("nan")
    extra: dict = field(default_factory=dict)


def _plain(v):
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


def tune(est, space, X, y, groups, mcfg: ModelingConfig, random_state: int):
    """Randomised search with grouped CV on a training subsample. Returns (model, params, score)."""
    t = mcfg.tuning
    if not t.enabled or not space:
        return est, {}, np.nan
    rng = np.random.RandomState(random_state)
    n = min(len(X), t.max_sample)
    idx = rng.choice(len(X), n, replace=False)
    Xs, ys, gs = X.iloc[idx], y[idx], groups[idx]
    folds = list(cv_splitter(mcfg.split_strategy, t.cv, random_state).split(Xs, ys, gs))
    search = RandomizedSearchCV(est, space, n_iter=t.n_iter, scoring=t.scoring, cv=folds,
                                random_state=random_state, n_jobs=1, refit=False)
    search.fit(Xs, ys)
    best = {k: _plain(v) for k, v in search.best_params_.items()}
    return clone(est).set_params(**best), best, float(search.best_score_)


def cross_validate(est, X, y, groups, mcfg: ModelingConfig, opts: ModelOptions):
    """CV on the training set: out-of-fold predictions plus per-fold metrics and fit time."""
    oof = np.zeros(len(y))
    rows = []
    for k, (tr, va) in enumerate(cv_splitter(mcfg.split_strategy, mcfg.cv_folds, opts.random_state)
                                 .split(X, y, groups)):
        m = clone(est)
        set_class_weight(m, y[tr], opts)
        t0 = time.time()
        m.fit(X.iloc[tr], y[tr])
        elapsed = time.time() - t0
        p = m.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        rows.append({"fold": k, "roc_auc": roc_auc_score(y[va], p),
                     "pr_auc": average_precision_score(y[va], p),
                     "brier": brier_score_loss(y[va], p), "fit_seconds": elapsed})
    return oof, pd.DataFrame(rows)


def train_models(X_tr, y_tr, g_tr, X_te, num, cat, mcfg: ModelingConfig, opts: ModelOptions,
                 beta: float, recall_target: float, cost_ratio_fn_fp: float) -> dict[str, ModelResult]:
    """Tune, cross-validate, pick thresholds on out-of-fold TRAIN predictions, refit and score the test set.

    A failing model is logged and skipped so it does not take the others down.
    """
    results: dict[str, ModelResult] = {}
    for name in mcfg.models:
        est, space = build_model(name, opts, num, cat)
        if est is None:
            log.warning("[%s] skipped: library not installed (uv sync --extra boosting)", name)
            continue
        try:
            log.info("=== %s ===", name)
            set_class_weight(est, y_tr, opts)
            est, params, score = tune(est, space, X_tr, y_tr, g_tr, mcfg, opts.random_state)
            if params:
                log.info("  best params (PR-AUC %.4f in search): %s", score, params)
            oof, cv = cross_validate(est, X_tr, y_tr, g_tr, mcfg, opts)
            thr = candidate_thresholds(y_tr, oof, beta, recall_target, cost_ratio_fn_fp)
            final = clone(est)
            set_class_weight(final, y_tr, opts)
            t0 = time.time()
            final.fit(X_tr, y_tr)
            t_fit = time.time() - t0
            t0 = time.time()
            p_te = final.predict_proba(X_te)[:, 1]
            t_pred = time.time() - t0
            results[name] = ModelResult(name, final, params, cv, thr, p_te, t_fit, t_pred, score)
            log.info("  CV PR-AUC %.4f ± %.4f | fit %.1fs", cv["pr_auc"].mean(), cv["pr_auc"].std(), t_fit)
        except Exception as e:  # noqa: BLE001
            log.error("  [%s] FAILED: %s: %s", name, type(e).__name__, e)
    if not results:
        raise RuntimeError("No model was trained.")
    return results


def select_best(results: dict[str, ModelResult]) -> str:
    """Best model by mean CV PR-AUC (the test set plays no part in the choice)."""
    cand = {n: r.cv["pr_auc"].mean() for n, r in results.items() if n != "dummy"}
    if not cand:
        raise RuntimeError("Only the baseline was trained; nothing to select.")
    return max(cand, key=cand.get)
