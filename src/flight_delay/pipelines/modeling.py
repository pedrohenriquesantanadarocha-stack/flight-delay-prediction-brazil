"""Modelling pipeline: train and compare models, diagnose the best one, write the report.

Models are not persisted, so training, evaluation and diagnostics run in the
same process; their outputs are the predictions (07_model_output) and the
report (08_reporting).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ..adapters.storage import DataCatalog
from ..config import Settings
from ..domain.columns import TARGET
from ..evaluation.bootstrap import paired_bootstrap
from ..evaluation.diagnostics import ablation, learning_curve, permutation_importance, segment_performance
from ..features.contract import to_model_frame
from ..modeling.catalog import ModelOptions
from ..modeling.split import train_test_indices
from ..modeling.training import ModelResult, select_best, train_models
from ..reporting import latex, plots, tables
from ..reporting.labels import MODEL_NAMES
from .features import load_model_input

log = logging.getLogger(__name__)

MAIN_OPERATING_POINT = "recall_target"


@dataclass
class Split:
    df: pd.DataFrame
    groups: dict[str, list[str]]
    num: list[str]
    cat: list[str]
    X_tr: pd.DataFrame
    X_te: pd.DataFrame
    y_tr: np.ndarray
    y_te: np.ndarray
    g_tr: np.ndarray
    df_te: pd.DataFrame

    @property
    def features(self) -> list[str]:
        return self.num + self.cat


def run_name(settings: Settings) -> str:
    years = settings.scope.years
    span = f"{min(years)}-{max(years)}" if len(years) > 1 else f"{years[0]}"
    return f"{settings.scope.sample_tag}_{settings.modeling.split_strategy}_{span}"


def model_options(settings: Settings) -> ModelOptions:
    m = settings.modeling
    return ModelOptions(random_state=settings.random_state, n_jobs=m.n_jobs, balance_classes=m.balance_classes,
                        target_encoding_smoothing=m.target_encoding_smoothing)


def make_split(df: pd.DataFrame, settings: Settings) -> Split:
    df, groups, num, cat = to_model_frame(df)
    features = num + cat
    m = settings.modeling
    tr, te, g = train_test_indices(df, m.split_strategy, m.test_size, settings.random_state)
    X_tr = df.iloc[tr][features].reset_index(drop=True)
    X_te = df.iloc[te][features].reset_index(drop=True)
    y = df[TARGET].to_numpy()
    s = Split(df, groups, num, cat, X_tr, X_te, y[tr], y[te], g[tr], df.iloc[te].reset_index(drop=True))
    log.info("Split '%s': train %s (%s to %s) | test %s (%s to %s)", m.split_strategy,
             f"{len(s.y_tr):,}", f"{df.iloc[tr]['date'].min():%Y-%m-%d}", f"{df.iloc[tr]['date'].max():%Y-%m-%d}",
             f"{len(s.y_te):,}", f"{s.df_te['date'].min():%Y-%m-%d}", f"{s.df_te['date'].max():%Y-%m-%d}")
    log.info("Delay prevalence: train %.2f%% | test %.2f%%", 100 * s.y_tr.mean(), 100 * s.y_te.mean())
    log.info("Features: %d numeric + %d categorical | groups: %s", len(num), len(cat),
             ", ".join(f"{k}({len(v)})" for k, v in groups.items()))
    return s


def write_predictions(results: dict[str, ModelResult], s: Split, catalog: DataCatalog, name: str) -> None:
    pred = s.df_te[["scheduled_departure", "airline_icao", "origin_icao", "destination_icao", TARGET]].copy()
    for n, r in results.items():
        pred[f"p_{n}"] = r.p_test
    catalog.write("model_output", f"{name}/test_predictions", pred)


def run_modeling(settings: Settings, catalog: DataCatalog, show_plots: bool = False,
                 df: pd.DataFrame | None = None) -> dict:
    """Train, evaluate and report. `df` overrides the model input (used by tests)."""
    m, ev = settings.modeling, settings.evaluation
    rs = settings.random_state
    opts = model_options(settings)
    name = run_name(settings)
    out_dir: Path = catalog.dir("reporting", name)
    fw = plots.FigureWriter(out_dir, show=show_plots)

    def csv(df_: pd.DataFrame, file: str, index: bool = True):
        df_.to_csv(out_dir / file, sep=";", index=index)

    # --- data and split ------------------------------------------------------------
    s = make_split(load_model_input(settings, catalog) if df is None else df, settings)

    # --- training, CV and test predictions -------------------------------------------
    results = train_models(s.X_tr, s.y_tr, s.g_tr, s.X_te, s.num, s.cat, m, opts,
                           ev.beta, ev.recall_target, ev.cost_ratio_fn_fp)
    for n, r in results.items():
        log.info("[%s] test ROC-AUC %.4f, PR-AUC %.4f", n, roc_auc_score(s.y_te, r.p_test),
                 average_precision_score(s.y_te, r.p_test))
    write_predictions(results, s, catalog, name)

    best = select_best(results)
    log.info("Best model by CV (PR-AUC): %s", MODEL_NAMES[best])
    ref = m.reference_model if m.reference_model in results else best

    # --- comparison tables ----------------------------------------------------------
    t_cv = tables.cv_table(results)
    csv(t_cv, "tabela_cv.csv")
    boot = paired_bootstrap(s.y_te, {n: r.p_test for n, r in results.items()}, ref, ev.n_boot, rs)
    csv(boot.rename(index=MODEL_NAMES), "bootstrap_teste.csv")
    t_test = tables.test_table(results, s.y_te, boot, ref, ev.top_k)
    csv(t_test, "tabela_teste.csv")
    points = tables.operating_points(results, s.y_te, ev.beta)
    csv(points, "pontos_de_operacao.csv", index=False)
    t_op = tables.operating_point_table(points, ev.beta, MAIN_OPERATING_POINT)
    t_best = tables.best_model_table(points, best, ev.beta)

    # --- comparison figures -----------------------------------------------------------
    plots.roc_pr(results, s.y_te, fw)
    plots.calibration(results, s.y_te, m.balance_classes, fw)
    plots.confusion_matrices(results, s.y_te, MAIN_OPERATING_POINT, fw)
    plots.cv_boxplot(results, m.cv_folds, fw)
    plots.threshold_curves(best, results[best], s.y_te, ev.beta, fw)

    # --- diagnostics of the best model ---------------------------------------------------
    model, params = results[best].model, results[best].params
    idx = np.random.RandomState(rs).choice(len(s.X_te), min(len(s.X_te), ev.n_diagnostic_sample), replace=False)
    Xd, yd = s.X_te.iloc[idx].reset_index(drop=True), s.y_te[idx]

    log.info("Permutation importance...")
    imp_var = permutation_importance(model, Xd, yd, {c: [c] for c in s.features}, ev.n_permutation_repeats, rs)
    imp_grp = permutation_importance(model, Xd, yd, s.groups, ev.n_permutation_repeats, rs)
    csv(imp_var, "importancia_variaveis.csv", index=False)
    csv(imp_grp, "importancia_grupos.csv", index=False)
    plots.importance_bars(imp_var, "importancia_variaveis.png", f"Importância por permutação: {MODEL_NAMES[best]}", fw)
    plots.importance_bars(imp_grp, "importancia_grupos.png", "Importância por grupo de variáveis", fw)

    log.info("Ablation by feature group...")
    abl = ablation(best, params, s.groups, s.X_tr, s.y_tr, s.X_te, s.y_te, opts, ev.n_boot, rs)
    csv(abl, "ablacao_grupos.csv")

    log.info("Partial dependence...")
    try:
        plots.partial_dependence(model, s.X_te, s.features, rs, fw)
    except Exception as e:  # noqa: BLE001
        log.warning("  partial dependence not produced: %s: %s", type(e).__name__, e)

    lc = pd.DataFrame()
    if ev.learning_curve:
        log.info("Learning curve...")
        lc = learning_curve(best, params, s.X_tr, s.y_tr, s.X_te, s.y_te, s.num, s.cat, opts)
        csv(lc, "curva_aprendizado.csv", index=False)
        plots.learning_curve_plot(lc, best, fw)

    seg = segment_performance(s.df_te, s.y_te, results[best].p_test, results[best].thresholds[MAIN_OPERATING_POINT])
    csv(seg, "desempenho_segmentos.csv", index=False)

    # --- LaTeX ------------------------------------------------------------------------------
    recall_pct = f"{ev.recall_target:.0%}".replace("%", "\\%")
    tex = [
        latex.table(t_cv, f"Validação cruzada ({m.cv_folds} folds, média $\\pm$ desvio-padrão) no conjunto de treino.",
                    "tab:cv_modelos"),
        latex.table(t_test, "Desempenho no conjunto de teste com IC 95\\% por \\textit{bootstrap} pareado.",
                    "tab:teste_modelos"),
        latex.table(t_op, f"Desempenho no teste com o limiar calibrado no treino para recall $\\geq {recall_pct}$.",
                    "tab:ponto_operacao"),
        latex.table(t_best, f"{MODEL_NAMES[best]}: diferentes pontos de operação no teste.", "tab:melhor_modelo",
                    index_name="Ponto"),
        latex.table(tables.ablation_table(abl), f"Ablação por grupo de variáveis ({MODEL_NAMES[best]}, teste).",
                    "tab:ablacao", index_name="Configuração"),
        latex.table(tables.group_importance_table(imp_grp),
                    f"Importância por permutação de grupos de variáveis ({MODEL_NAMES[best]}).",
                    "tab:importancia_grupos", index_name="Grupo"),
    ]
    (out_dir / "tabelas_modelagem.tex").write_text(
        "% Generated by flight_delay (requires booktabs)\n\n" + "\n".join(tex), encoding="utf-8")

    summary = {
        "run": name, "split_strategy": m.split_strategy, "n_train": int(len(s.y_tr)), "n_test": int(len(s.y_te)),
        "prevalence_train": float(s.y_tr.mean()), "prevalence_test": float(s.y_te.mean()),
        "best_model_cv": best, "reference_model": ref,
        "hyperparameters": {n: r.params for n, r in results.items()},
        "thresholds": {n: r.thresholds for n, r in results.items()},
        "settings": settings.to_dict(),
    }
    catalog.write_json("reporting", f"{name}/summary", summary)

    log.info("\n--- Test: model ranking ---\n%s", t_test.to_string())
    log.info("\n--- %s: operating points ---\n%s", MODEL_NAMES[best], t_best.round(3).to_string())
    log.info("\n--- Importance by group ---\n%s", imp_grp.round(4).to_string(index=False))
    log.info("Report written to %s", out_dir)

    return {"results": results, "best": best, "split": s, "cv_table": t_cv, "test_table": t_test,
            "operating_points": points, "ablation": abl, "importance_variables": imp_var,
            "importance_groups": imp_grp, "segments": seg, "learning_curve": lc, "report_dir": out_dir}
