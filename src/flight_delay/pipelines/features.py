"""Feature pipeline: primary layer to features (all flights) and the modelling sample."""

from __future__ import annotations

import logging

import pandas as pd

from ..adapters.storage import DataCatalog
from ..config import Settings
from ..domain.columns import TARGET
from ..domain.sampling import sample_per_month
from ..domain.schemas import validate
from ..features.build import build_month_features
from .etl import flights_name, weather_name

log = logging.getLogger(__name__)


def feature_name(year: int, month: int, threshold: int) -> str:
    return f"flights/features_{year}_{month:02d}_thr{threshold}"


def sample_name(year: int, month: int, tag: str) -> str:
    return f"samples/sample_{year}_{month:02d}_{tag}"


def _weather_for(years, catalog: DataCatalog) -> pd.DataFrame:
    """Weather of every available year in scope, so that a flight departing on
    31 Dec and arriving on 1 Jan finds the destination weather of the next year."""
    parts = [catalog.read("primary", weather_name(y)) for y in years if catalog.exists("primary", weather_name(y))]
    if not parts:
        raise FileNotFoundError("No weather table in 03_primary: run the ETL first.")
    return pd.concat(parts, ignore_index=True).drop_duplicates(["airport_icao", "date"])


def run_features(settings: Settings, catalog: DataCatalog, force: bool = False) -> list[str]:
    """Build features for every available month and the per-month samples. Returns sample names."""
    scope = settings.scope
    weather = None
    samples = []
    for year in scope.years:
        for month in scope.months:
            if not catalog.exists("primary", flights_name(year, month)):
                continue
            s_name = sample_name(year, month, scope.sample_tag)
            if catalog.exists("model_input", s_name) and not force:
                samples.append(s_name)
                continue
            f_name = feature_name(year, month, scope.delay_threshold_min)
            if catalog.exists("feature", f_name) and not force:
                feats = catalog.read("feature", f_name)
            else:
                if weather is None:
                    weather = _weather_for(scope.years, catalog)
                flights = catalog.read("primary", flights_name(year, month))
                feats = validate(build_month_features(flights, weather, scope.delay_threshold_min), "model_input")
                catalog.write("feature", f_name, feats)
            sample = sample_per_month(feats, scope.max_flights_per_month, scope.random_state)
            catalog.write("model_input", s_name, sample)
            samples.append(s_name)
            log.info("[%d-%02d] in month %s | sample %s | delayed %.1f%% | origin weather coverage %.1f%%",
                     year, month, f"{len(feats):,}", f"{len(sample):,}", 100 * sample[TARGET].mean(),
                     100 * sample["origin_precip_mm"].notna().mean())
    return samples


def load_model_input(settings: Settings, catalog: DataCatalog) -> pd.DataFrame:
    scope = settings.scope
    parts = [catalog.read("model_input", sample_name(y, m, scope.sample_tag))
             for y in scope.years for m in scope.months
             if catalog.exists("model_input", sample_name(y, m, scope.sample_tag))]
    if not parts:
        raise FileNotFoundError("No modelling sample in 05_model_input: run the feature pipeline first.")
    return pd.concat(parts, ignore_index=True)
