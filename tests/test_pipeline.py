"""End-to-end: ETL -> features -> modelling with fake sources (no network)."""

import json

import pandas as pd

from flight_delay.adapters.storage import DataCatalog
from flight_delay.pipelines.etl import ingest_weather, run_etl
from flight_delay.pipelines.features import run_features
from flight_delay.pipelines.run import make_catalog, run

from synthetic import FakeFlightSource, FakeWeatherSource


def test_full_run_produces_every_layer_and_report(settings_factory):
    s = settings_factory()
    out = run(s, flight_source=FakeFlightSource(1100), weather_source=FakeWeatherSource())
    cat = make_catalog(s)
    root = cat.root
    for layer in ("02_intermediate", "03_primary", "04_feature", "05_model_input", "07_model_output"):
        assert any((root / layer).rglob("*")), layer
    report = out["report_dir"]
    for f in ("tabelas_modelagem.tex", "tabela_teste.csv", "curvas_roc_pr.png", "importancia_grupos.png",
              "ablacao_grupos.csv", "summary.json"):
        assert (report / f).exists(), f
    summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
    assert summary["best_model_cv"] in {"logistic_regression", "decision_tree"}
    assert "\\begin{table}" in (report / "tabelas_modelagem.tex").read_text(encoding="utf-8")
    assert not any((root / "01_raw").rglob("*.csv"))  # keep_raw: false


def test_rerun_resumes_without_downloading_again(settings_factory):
    s = settings_factory()
    cat = make_catalog(s)
    fs, ws = FakeFlightSource(600), FakeWeatherSource()
    run_etl(s, cat, fs, ws)
    run_features(s, cat)
    n_flight, n_weather = len(fs.calls), len(ws.calls)
    run_etl(s, cat, fs, ws)
    run_features(s, cat)
    assert (len(fs.calls), len(ws.calls)) == (n_flight, n_weather)


def test_unpublished_month_is_skipped(settings_factory):
    s = settings_factory()
    cat = make_catalog(s)
    run_etl(s, cat, FakeFlightSource(600, missing={(2025, 2)}), FakeWeatherSource())
    assert cat.exists("primary", "flights/flights_2025_01")
    assert not cat.exists("primary", "flights/flights_2025_02")
    assert len(run_features(s, cat)) == 1


def test_weather_is_queried_again_only_for_new_airports(tmp_path):
    cat = DataCatalog(tmp_path, "pickle")
    ws = FakeWeatherSource()
    ingest_weather(2025, ["SBGR", "SBRJ"], cat, ws)
    ingest_weather(2025, ["SBGR"], cat, ws)
    assert len(ws.calls) == 1
    ingest_weather(2025, ["SBGR", "SBKP"], cat, ws)
    assert len(ws.calls) == 2 and set(ws.calls[-1][0]) == {"SBGR", "SBRJ", "SBKP"}


def test_catalog_roundtrip_is_atomic(tmp_path):
    cat = DataCatalog(tmp_path, "pickle")
    df = pd.DataFrame({"a": [1, 2], "t": pd.to_datetime(["2025-01-01", "2025-01-02"])})
    cat.write("primary", "x/y", df)
    assert cat.read("primary", "x/y").equals(df)
    assert not list(tmp_path.rglob("*.part"))
