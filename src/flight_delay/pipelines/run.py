"""Composition root: builds the adapters from the settings and runs the stages."""

from __future__ import annotations

import logging

from ..adapters.anac import AnacVraSource
from ..adapters.noaa_bigquery import BigQueryGsodSource
from ..adapters.storage import DataCatalog
from ..config import Settings
from ..ports import FlightDataSource, WeatherSource
from .etl import run_etl
from .features import run_features
from .modeling import run_modeling

log = logging.getLogger(__name__)

STAGES = ("etl", "features", "modeling")


def make_catalog(settings: Settings) -> DataCatalog:
    p = settings.paths
    return DataCatalog(p.data_root, settings.storage.format, raw_dir=p.raw_tmp)


def make_sources(settings: Settings) -> tuple[FlightDataSource, WeatherSource]:
    src, w = settings.source, settings.weather
    return (AnacVraSource(src.anac_base_url, src.timeout_s, src.retries),
            BigQueryGsodSource(w.gcp_project, w.gcp_project_fallback))


def run(settings: Settings, stages=STAGES, force: bool = False, show_plots: bool = False,
        catalog: DataCatalog | None = None, flight_source: FlightDataSource | None = None,
        weather_source: WeatherSource | None = None) -> dict | None:
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise ValueError(f"unknown stages {sorted(unknown)}; valid: {STAGES}")
    catalog = catalog or make_catalog(settings)
    if "etl" in stages:
        if flight_source is None or weather_source is None:
            fs, ws = make_sources(settings)
            flight_source, weather_source = flight_source or fs, weather_source or ws
        run_etl(settings, catalog, flight_source, weather_source, force)
    if "features" in stages:
        run_features(settings, catalog, force)
    if "modeling" in stages:
        return run_modeling(settings, catalog, show_plots=show_plots)
    return None
