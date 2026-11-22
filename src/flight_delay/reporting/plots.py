"""Figures of the paper (static, light background, Portuguese labels).

Each model keeps the same colour in every figure (colour follows the entity, not
its rank), taken in fixed order from a colour-blind-validated categorical palette.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from ..modeling.catalog import MODEL_KEYS
from .labels import MODEL_NAMES, label

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#d9d8d2"
MODEL_COLORS = {k: CATEGORICAL[i % len(CATEGORICAL)] for i, k in enumerate(m for m in MODEL_KEYS if m != "dummy")}

plt.rcParams.update({
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2.0, "legend.frameon": False,
})


class FigureWriter:
    def __init__(self, out_dir: str | Path, show: bool = False, dpi: int = 200):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.show = show
        self.dpi = dpi

    def save(self, fig, name: str) -> Path:
        fig.tight_layout()
        path = self.out_dir / name
        fig.savefig(path, dpi=self.dpi)
        if self.show:
            plt.show()
        plt.close(fig)
        return path


def _models(results):
    return [n for n in results if n != "dummy"]


def roc_pr(results, y_te, fw: FigureWriter):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for n in _models(results):
        p = results[n].p_test
        fpr, tpr, _ = roc_curve(y_te, p)
        pr, rc, _ = precision_recall_curve(y_te, p)
        a1.plot(fpr, tpr, color=MODEL_COLORS[n], label=MODEL_NAMES[n])
        a2.plot(rc, pr, color=MODEL_COLORS[n], label=MODEL_NAMES[n])
    a1.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=0.8)
    a2.axhline(y_te.mean(), color=MUTED, ls="--", lw=0.8, label="Prevalência")
    a1.set(xlabel="Taxa de falsos positivos", ylabel="Taxa de verdadeiros positivos (Recall)", title="Curva ROC")
    a2.set(xlabel="Recall", ylabel="Precisão", title="Curva Precisão-Recall")
    a2.legend(fontsize=8)
    return fw.save(fig, "curvas_roc_pr.png")


def calibration(results, y_te, balance_classes: bool, fw: FigureWriter):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for n in _models(results):
        frac, mean_p = calibration_curve(y_te, results[n].p_test, n_bins=10, strategy="quantile")
        ax.plot(mean_p, frac, marker="o", ms=4, color=MODEL_COLORS[n], label=MODEL_NAMES[n])
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=0.8)
    title = "Calibração" + (" (com pesos de classe: probabilidades infladas)" if balance_classes else "")
    ax.set(xlabel="Probabilidade prevista", ylabel="Fração observada de atrasos", title=title)
    ax.title.set_fontsize(9)
    ax.legend(fontsize=8)
    return fw.save(fig, "calibracao.png")


def confusion_matrices(results, y_te, point: str, fw: FigureWriter):
    names = _models(results)
    cols = min(4, len(names))
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.6 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, n in zip(axes.ravel(), names):
        ax.axis("on")
        ax.grid(False)
        pred = (results[n].p_test >= results[n].thresholds[point]).astype(int)
        cm = confusion_matrix(y_te, pred, labels=[0, 1])
        pct = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.1%})", ha="center", va="center", fontsize=9,
                        color="white" if pct[i, j] > 0.5 else INK)
        ax.set_xticks([0, 1], ["No horário", "Atrasado"])
        ax.set_yticks([0, 1], ["No horário", "Atrasado"])
        ax.set_title(MODEL_NAMES[n], fontsize=10)
        ax.set_xlabel("Previsto")
        ax.set_ylabel("Real")
    return fw.save(fig, "matrizes_confusao.png")


def cv_boxplot(results, cv_folds: int, fw: FigureWriter):
    names = _models(results)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    data, labels = [results[n].cv["pr_auc"] for n in names], [MODEL_NAMES[n] for n in names]
    try:
        ax.boxplot(data, tick_labels=labels)
    except TypeError:  # matplotlib < 3.9
        ax.boxplot(data, labels=labels)
    ax.set(ylabel="PR-AUC por fold", title=f"Validação cruzada ({cv_folds} folds)")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
    return fw.save(fig, "cv_pr_auc.png")


def threshold_curves(name, result, y_te, beta: float, fw: FigureWriter):
    pr, rc, thr = precision_recall_curve(y_te, result.p_test)
    b2 = beta ** 2
    fb = (1 + b2) * pr[:-1] * rc[:-1] / np.maximum(b2 * pr[:-1] + rc[:-1], 1e-12)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(thr, pr[:-1], color=CATEGORICAL[0], label="Precisão")
    ax.plot(thr, rc[:-1], color=CATEGORICAL[1], label="Recall")
    ax.plot(thr, fb, color=CATEGORICAL[2], label=f"F{beta:g}")
    for k, v in result.thresholds.items():
        if k != "default_0.5":
            ax.axvline(v, ls=":", lw=1, color=MUTED)
            ax.text(v, 1.01, label(k), rotation=90, fontsize=7, va="bottom", color=MUTED)
    ax.set(xlabel="Limiar de decisão", title=f"{MODEL_NAMES[name]}: precisão, recall e F{beta:g} vs. limiar")
    ax.legend(fontsize=8)
    return fw.save(fig, "limiar_melhor_modelo.png")


def importance_bars(imp, file_name: str, title: str, fw: FigureWriter, top_n: int = 15):
    d = imp.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(d) + 1.5))
    ax.barh([label(v) for v in d["variable"]], d["pr_auc_drop"], xerr=d["pr_auc_drop_sd"],
            color=CATEGORICAL[0], error_kw={"ecolor": MUTED, "lw": 1})
    ax.set(xlabel="Queda de PR-AUC ao embaralhar")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", visible=False)
    return fw.save(fig, file_name)


PDP_CANDIDATES = ("origin_wind_speed_kmh", "destination_wind_speed_kmh", "origin_precip_mm",
                  "destination_precip_mm", "dep_hour", "origin_hourly_flow")


def partial_dependence(model, X_te, features, random_state: int, fw: FigureWriter):
    cand = [c for c in PDP_CANDIDATES if c in features]
    if not cand:
        return None
    sample = X_te.sample(min(len(X_te), 5000), random_state=random_state)
    ncols = min(3, len(cand))
    nrows = int(np.ceil(len(cand) / 3))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), squeeze=False)
    disp = PartialDependenceDisplay.from_estimator(
        model, sample, cand, kind="average", grid_resolution=25, percentiles=(0.02, 0.98),
        ax=axes.ravel()[: len(cand)], line_kw={"color": CATEGORICAL[0]})
    for a, c in zip(disp.axes_.ravel(), cand):
        a.set_ylabel("Prob. média de atraso")
        a.set_xlabel(label(c))
    fig.suptitle("Dependência parcial: probabilidade média de atraso", fontsize=10)
    return fw.save(fig, "dependencia_parcial.png")


def learning_curve_plot(lc, name, fw: FigureWriter):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lc["n_train"], lc["roc_auc"], marker="o", color=CATEGORICAL[0], label="ROC-AUC")
    ax.plot(lc["n_train"], lc["pr_auc"], marker="s", color=CATEGORICAL[1], label="PR-AUC")
    ax.set(xscale="log", xlabel="Tamanho do conjunto de treino", ylabel="Métrica no teste",
           title=f"Curva de aprendizado: {MODEL_NAMES[name]}")
    ax.legend()
    return fw.save(fig, "curva_aprendizado.png")
