"""Decision thresholds chosen on out-of-fold TRAIN predictions (the test set never takes part)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve


def candidate_thresholds(y, p, beta: float, recall_target: float, cost_ratio_fn_fp: float) -> dict[str, float]:
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    b2 = beta ** 2
    fb = (1 + b2) * prec * rec / np.maximum(b2 * prec + rec, 1e-12)
    ok = np.where(rec >= recall_target)[0]
    pos = float(np.sum(y))
    tp = rec * pos
    fp = tp * (1 - prec) / np.maximum(prec, 1e-12)
    cost = cost_ratio_fn_fp * (pos - tp) + fp
    return {
        "default_0.5": 0.5,
        "f_beta": float(thr[int(np.argmax(fb))]),
        "recall_target": float(thr[ok[-1]] if len(ok) else thr[0]),
        "cost": float(thr[int(np.argmin(cost))]),
    }
