import numpy as np
import pandas as pd
import pytest

from flight_delay.evaluation.bootstrap import format_ci, paired_bootstrap
from flight_delay.evaluation.metrics import threshold_metrics, top_k
from flight_delay.evaluation.thresholds import candidate_thresholds
from flight_delay.modeling.catalog import MODEL_KEYS, ModelOptions, available, build_model, set_class_weight
from flight_delay.modeling.split import train_test_indices
from flight_delay.reporting import latex


@pytest.fixture
def days_frame():
    rng = np.random.default_rng(0)
    dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 60, 3000), unit="D")
    return pd.DataFrame({"date": dates, "target": (rng.random(3000) < 0.2).astype(int)})


def test_by_day_split_never_shares_a_day(days_frame):
    tr, te, _ = train_test_indices(days_frame, "by_day", 0.2, 42)
    assert not set(days_frame["date"].iloc[tr]) & set(days_frame["date"].iloc[te])
    assert 0.1 < len(te) / len(days_frame) < 0.3


def test_temporal_split_tests_on_the_future(days_frame):
    tr, te, _ = train_test_indices(days_frame, "temporal", 0.2, 42)
    assert days_frame["date"].iloc[tr].max() < days_frame["date"].iloc[te].min()


@pytest.mark.parametrize("name", [m for m in MODEL_KEYS if available(m)])
def test_every_model_builds_and_fits(name):
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400),
                      "c": rng.choice(["x", "y", "z"], 400).astype(object)})
    X.loc[::17, "a"] = np.nan
    y = (X["b"] + rng.normal(size=400) > 0.8).astype(int).to_numpy()
    est, space = build_model(name, ModelOptions(n_jobs=1), ["a", "b"], ["c"])
    set_class_weight(est, y, ModelOptions())
    est.fit(X, y)
    p = est.predict_proba(X)[:, 1]
    assert p.shape == (400,) and np.all((p >= 0) & (p <= 1))
    assert isinstance(space, dict)


def test_scale_pos_weight_only_for_boosting():
    est, _ = build_model("random_forest", ModelOptions(), ["a"], [])
    set_class_weight(est, np.array([0, 0, 0, 1]), ModelOptions())
    assert "clf__scale_pos_weight" not in est.get_params()


def test_thresholds_respect_the_recall_target():
    rng = np.random.default_rng(2)
    y = (rng.random(2000) < 0.2).astype(int)
    p = np.clip(0.2 + 0.3 * y + rng.normal(0, 0.2, 2000), 0, 1)
    thr = candidate_thresholds(y, p, beta=2.0, recall_target=0.7, cost_ratio_fn_fp=3.0)
    assert set(thr) == {"default_0.5", "f_beta", "recall_target", "cost"}
    assert threshold_metrics(y, p, thr["recall_target"], 2.0)["recall"] >= 0.7


def test_top_k_lift_of_a_perfect_ranking():
    y = np.array([1] * 10 + [0] * 90)
    out = top_k(y, y.astype(float), 0.10)
    assert out["precision_topk"] == 1 and out["recall_topk"] == 1 and out["lift_topk"] == pytest.approx(10)


def test_paired_bootstrap_reference_has_no_delta():
    rng = np.random.default_rng(3)
    y = (rng.random(500) < 0.3).astype(int)
    probs = {"a": rng.random(500), "b": np.clip(y * 0.5 + rng.random(500) * 0.5, 0, 1)}
    boot = paired_bootstrap(y, probs, "a", n_boot=50, random_state=0)
    assert pd.isna(boot.loc["a"].get("d_pr_auc", np.nan))
    assert boot.loc["b", "d_pr_auc_lo"] <= boot.loc["b", "d_pr_auc"] <= boot.loc["b", "d_pr_auc_hi"]
    assert format_ci(0.7123, 0.70, 0.72) == "0.712 [0.700; 0.720]"


def test_latex_uses_decimal_comma_and_escapes():
    df = pd.DataFrame({"PR-AUC": [0.2551], "IC": ["0.712 [0.700; 0.720]"]}, index=["Random_Forest"])
    tex = latex.table(df, "Legenda", "tab:x")
    assert "0,255" in tex and "0,712 [0,700; 0,720]" in tex and "Random\\_Forest" in tex
