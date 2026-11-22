"""Threshold-dependent metrics and top-k analysis."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef


def threshold_metrics(y, p, threshold: float, beta: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    b2 = beta ** 2
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fb = (1 + b2) * prec * rec / (b2 * prec + rec) if (b2 * prec + rec) else 0.0
    mcc = matthews_corrcoef(y, pred) if pred.min() != pred.max() else 0.0
    return {"precision": prec, "recall": rec, "specificity": spec, "f1": f1, "f_beta": fb,
            "mcc": mcc, "balanced_accuracy": (rec + spec) / 2,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def top_k(y, p, frac: float) -> dict:
    """If the operator only flags the `frac` riskiest flights, how much does it catch?"""
    k = max(1, int(len(y) * frac))
    idx = np.argsort(-p, kind="stable")[:k]
    prec = float(y[idx].mean())
    return {"precision_topk": prec, "lift_topk": prec / float(y.mean()),
            "recall_topk": float(y[idx].sum() / y.sum())}
