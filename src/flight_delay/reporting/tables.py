"""Comparison tables of the paper (Portuguese headers, raw LaTeX allowed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from ..evaluation.bootstrap import format_ci
from ..evaluation.diagnostics import ABLATIONS
from ..evaluation.metrics import threshold_metrics, top_k
from .labels import GROUP_LABELS, MODEL_NAMES, OPERATING_POINTS

ABLATION_LABELS = {
    "calendar_traffic": "Calendário + tráfego",
    "plus_airline_route": "+ companhia e rota",
    "plus_weather_origin": "+ clima na origem",
    "full": "Completo (+ clima no destino)",
    "no_weather": "Sem clima",
    "weather_only": "Somente clima",
}
assert set(ABLATION_LABELS) == set(ABLATIONS)


def cv_table(results) -> pd.DataFrame:
    return pd.DataFrame({
        MODEL_NAMES[n]: {
            "ROC-AUC": f"{r.cv['roc_auc'].mean():.3f} ± {r.cv['roc_auc'].std():.3f}",
            "PR-AUC": f"{r.cv['pr_auc'].mean():.3f} ± {r.cv['pr_auc'].std():.3f}",
            "Brier": f"{r.cv['brier'].mean():.3f}",
            "Ajuste (s)": f"{r.fit_seconds:.0f}",
        } for n, r in results.items()}).T


def test_table(results, y_te, boot: pd.DataFrame, reference: str, k: float) -> pd.DataFrame:
    rows = {}
    k_label = f"{k:.0%}".replace("%", "\\%")
    delta_col = f"$\\Delta$ PR-AUC vs. {MODEL_NAMES[reference]}"
    for n, r in results.items():
        b = boot.loc[n]
        row = {
            "ROC-AUC [IC 95\\%]": format_ci(b["roc_auc"], b.get("roc_auc_lo"), b.get("roc_auc_hi")),
            "PR-AUC [IC 95\\%]": format_ci(b["pr_auc"], b.get("pr_auc_lo"), b.get("pr_auc_hi")),
            "Brier": brier_score_loss(y_te, r.p_test),
            "Log-loss": log_loss(y_te, np.clip(r.p_test, 1e-6, 1 - 1e-6)),
        }
        tk = top_k(y_te, r.p_test, k)
        row[f"Lift@{k_label}"] = tk["lift_topk"]
        row[f"Recall@{k_label}"] = tk["recall_topk"]
        if n != reference and "d_pr_auc" in b and not pd.isna(b["d_pr_auc"]):
            row[delta_col] = format_ci(b["d_pr_auc"], b["d_pr_auc_lo"], b["d_pr_auc_hi"])
        else:
            row[delta_col] = "--"
        rows[MODEL_NAMES[n]] = row
    return pd.DataFrame(rows).T


def operating_points(results, y_te, beta: float) -> pd.DataFrame:
    """Metrics on the test set at every candidate threshold of every model (long format)."""
    rows = []
    for n, r in results.items():
        for point, thr in r.thresholds.items():
            m = threshold_metrics(y_te, r.p_test, thr, beta)
            m.update(model=n, point=point, threshold=thr)
            rows.append(m)
    return pd.DataFrame(rows)


def operating_point_table(points: pd.DataFrame, beta: float, point: str = "recall_target") -> pd.DataFrame:
    """Main table: precision at a fixed recall is comparable across models (the F-beta
    threshold degenerates into "flag everything" when the model is weak)."""
    sel = points[points["point"] == point].copy()
    sel.index = sel["model"].map(MODEL_NAMES)
    out = sel[["threshold", "precision", "recall", "specificity", "f1", "f_beta", "mcc"]]
    out.columns = ["Limiar", "Precisão", "Recall", "Especificidade", "F1", f"F{beta:g}", "MCC"]
    return out


def best_model_table(points: pd.DataFrame, best: str, beta: float) -> pd.DataFrame:
    sel = points[points["model"] == best].copy()
    sel.index = sel["point"].map(OPERATING_POINTS)
    out = sel[["threshold", "precision", "recall", "specificity", "f_beta", "tp", "fp", "fn", "tn"]]
    out.columns = ["Limiar", "Precisão", "Recall", "Especificidade", f"F{beta:g}", "VP", "FP", "FN", "VN"]
    return out


def ablation_table(abl: pd.DataFrame) -> pd.DataFrame:
    delta = [format_ci(r["d_pr_auc"], r["d_pr_auc_lo"], r["d_pr_auc_hi"])
             if "d_pr_auc" in r and not pd.isna(r["d_pr_auc"]) else "--" for _, r in abl.iterrows()]
    return pd.DataFrame({
        "Variáveis": abl["n_features"].astype(int),
        "ROC-AUC": abl["roc_auc"], "PR-AUC": abl["pr_auc"],
        "$\\Delta$ PR-AUC vs. sem clima": delta,
    }, index=[ABLATION_LABELS[i] for i in abl.index])


def group_importance_table(imp_groups: pd.DataFrame) -> pd.DataFrame:
    out = imp_groups.set_index("variable")[["roc_auc_drop", "pr_auc_drop"]]
    out.index = [GROUP_LABELS.get(i, i) for i in out.index]
    out.columns = ["Queda ROC-AUC", "Queda PR-AUC"]
    return out
