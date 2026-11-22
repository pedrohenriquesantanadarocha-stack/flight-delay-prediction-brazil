"""Diagnostics of the selected model: permutation importance, ablation, learning curve, segments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ..features.contract import CATEGORICAL
from ..modeling.catalog import ModelOptions, build_model, set_class_weight
from .bootstrap import paired_bootstrap


def permutation_importance(model, X, y, groups: dict[str, list[str]], n_repeats: int, seed: int) -> pd.DataFrame:
    """Drop in ROC-AUC and PR-AUC when one variable (or a whole group) is shuffled.

    Unlike impurity importance (MDI) it is not biased towards continuous variables.
    """
    rng = np.random.RandomState(seed)
    p0 = model.predict_proba(X)[:, 1]
    roc0, pr0 = roc_auc_score(y, p0), average_precision_score(y, p0)
    rows = []
    for name, cols in groups.items():
        d_roc, d_pr = [], []
        for _ in range(n_repeats):
            Xp = X.copy()
            idx = rng.permutation(len(X))  # same permutation for every column of the group
            for c in cols:
                Xp[c] = X[c].to_numpy()[idx]
            p = model.predict_proba(Xp)[:, 1]
            d_roc.append(roc0 - roc_auc_score(y, p))
            d_pr.append(pr0 - average_precision_score(y, p))
        rows.append({"variable": name, "roc_auc_drop": np.mean(d_roc), "roc_auc_drop_sd": np.std(d_roc),
                     "pr_auc_drop": np.mean(d_pr), "pr_auc_drop_sd": np.std(d_pr)})
    return pd.DataFrame(rows).sort_values("pr_auc_drop", ascending=False).reset_index(drop=True)


# Configurations of the ablation study: label -> groups (in this order)
ABLATIONS: dict[str, tuple[str, ...]] = {
    "calendar_traffic": ("calendar", "traffic"),
    "plus_airline_route": ("calendar", "traffic", "airline_route"),
    "plus_weather_origin": ("calendar", "traffic", "airline_route", "weather_origin"),
    "full": ("calendar", "traffic", "airline_route", "weather_origin", "weather_destination"),
    "no_weather": ("calendar", "traffic", "airline_route"),
    "weather_only": ("weather_origin", "weather_destination"),
}
ABLATION_REFERENCE = "no_weather"


def ablation(name, params, groups, X_tr, y_tr, X_te, y_te, opts: ModelOptions,
             n_boot: int, random_state: int) -> pd.DataFrame:
    """Refit the selected model on subsets of groups: how much does weather add?"""
    probs, rows = {}, []
    for key, wanted in ABLATIONS.items():
        gs = [g for g in wanted if g in groups]
        if not gs:
            continue
        cols = [c for g in gs for c in groups[g]]
        num = [c for c in cols if c not in CATEGORICAL]
        cat = [c for c in cols if c in CATEGORICAL]
        est, _ = build_model(name, opts, num, cat)
        est.set_params(**params)
        set_class_weight(est, y_tr, opts)
        est.fit(X_tr[cols], y_tr)
        probs[key] = est.predict_proba(X_te[cols])[:, 1]
        rows.append({"configuration": key, "n_features": len(cols)})
    ref = ABLATION_REFERENCE if ABLATION_REFERENCE in probs else next(iter(probs))
    boot = paired_bootstrap(y_te, probs, ref, n_boot, random_state)
    return pd.DataFrame(rows).set_index("configuration").join(boot)


def learning_curve(name, params, X_tr, y_tr, X_te, y_te, num, cat, opts: ModelOptions,
                   fractions=(0.05, 0.10, 0.25, 0.50, 0.75, 1.0)) -> pd.DataFrame:
    """Test performance as a function of training size (justifies the sampling)."""
    order = np.random.RandomState(opts.random_state).permutation(len(X_tr))
    rows = []
    for f in fractions:
        idx = order[: max(int(len(order) * f), 500)]
        est, _ = build_model(name, opts, num, cat)
        est.set_params(**params)
        set_class_weight(est, y_tr[idx], opts)
        est.fit(X_tr.iloc[idx], y_tr[idx])
        p = est.predict_proba(X_te)[:, 1]
        rows.append({"train_fraction": f, "n_train": len(idx), "roc_auc": roc_auc_score(y_te, p),
                     "pr_auc": average_precision_score(y_te, p)})
    return pd.DataFrame(rows)


def segment_performance(df_te: pd.DataFrame, y, p, threshold: float) -> pd.DataFrame:
    """Performance by hour band, month, airline (top 8) and origin airport (top 10)."""
    d = pd.DataFrame({"y": y, "p": p, "pred": (p >= threshold).astype(int)}, index=df_te.index)
    d["hour_band"] = pd.cut(df_te["dep_hour"], [-1, 5, 8, 11, 14, 17, 20, 23],
                            labels=["00-05", "06-08", "09-11", "12-14", "15-17", "18-20", "21-23"])
    d["month"] = df_te["month"]
    dims = [("hour_band", None), ("month", None)]
    if "airline_icao" in df_te:
        d["airline"] = df_te["airline_icao"]
        dims.append(("airline", 8))
    d["origin"] = df_te["origin_icao"]
    dims.append(("origin", 10))
    rows = []
    for dim, top_n in dims:
        allowed = d[dim].value_counts().index[:top_n] if top_n else None
        for seg, g in d.groupby(dim, observed=True):
            if allowed is not None and seg not in allowed:
                continue
            n, pos = len(g), int(g["y"].sum())
            tp = int(((g["pred"] == 1) & (g["y"] == 1)).sum())
            rows.append({
                "dimension": dim, "segment": str(seg), "n": n, "delay_rate": pos / n,
                "roc_auc": roc_auc_score(g["y"], g["p"]) if pos >= 30 and n - pos >= 30 else np.nan,
                "recall": tp / pos if pos else np.nan,
                "precision": tp / max(int(g["pred"].sum()), 1),
            })
    return pd.DataFrame(rows)
