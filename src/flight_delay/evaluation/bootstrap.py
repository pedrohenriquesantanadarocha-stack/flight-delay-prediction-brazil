"""Paired bootstrap: 95% CIs for ROC-AUC/PR-AUC and differences against a reference."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def paired_bootstrap(y, probs: dict[str, np.ndarray], reference: str, n_boot: int, random_state: int) -> pd.DataFrame:
    """All models are evaluated on the SAME resamples, so differences are paired."""
    names = list(probs)
    point = {n: {"roc": roc_auc_score(y, p), "pr": average_precision_score(y, p)} for n, p in probs.items()}
    rows = {n: {"roc_auc": point[n]["roc"], "pr_auc": point[n]["pr"]} for n in names}
    if n_boot > 0:
        rng = np.random.RandomState(random_state)
        samples = {n: {"roc": [], "pr": []} for n in names}
        for _ in range(n_boot):
            idx = rng.randint(0, len(y), len(y))
            yb = y[idx]
            if yb.min() == yb.max():
                continue
            for n, p in probs.items():
                samples[n]["roc"].append(roc_auc_score(yb, p[idx]))
                samples[n]["pr"].append(average_precision_score(yb, p[idx]))
        for n in names:
            for m, col in (("roc", "roc_auc"), ("pr", "pr_auc")):
                a = np.array(samples[n][m])
                rows[n][f"{col}_lo"], rows[n][f"{col}_hi"] = np.percentile(a, [2.5, 97.5])
                if reference in probs and n != reference:
                    d = a - np.array(samples[reference][m])
                    rows[n][f"d_{col}"] = point[n][m] - point[reference][m]
                    rows[n][f"d_{col}_lo"], rows[n][f"d_{col}_hi"] = np.percentile(d, [2.5, 97.5])
    return pd.DataFrame(rows).T


def format_ci(v, lo, hi, decimals: int = 3) -> str:
    if lo is None or hi is None or pd.isna(lo) or pd.isna(hi):
        return f"{v:.{decimals}f}"
    return f"{v:.{decimals}f} [{lo:.{decimals}f}; {hi:.{decimals}f}]"
