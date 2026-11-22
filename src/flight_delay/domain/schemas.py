"""Data contracts between layers (Pandera).

Each pipeline validates what it writes, so a change in the ANAC or NOAA files
fails at the boundary with a clear report instead of silently degrading the model.
"""

from __future__ import annotations

import logging
import warnings
from functools import cache

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from .columns import DELAY_MIN, REQUIRED_VRA_COLUMNS, TARGET, WEATHER_MEASURES

log = logging.getLogger(__name__)


@cache
def _pa():
    try:
        import pandera.pandas as pa  # pandera >= 0.24
    except ImportError:
        try:
            import pandera as pa
        except ImportError:
            return None
    return pa


def _is_datetime(s: pd.Series) -> bool:
    return is_datetime64_any_dtype(s)


def _is_numeric(s: pd.Series) -> bool:
    return is_numeric_dtype(s)


@cache
def _schemas() -> dict:
    pa = _pa()
    if pa is None:
        return {}
    Check, Column = pa.Check, pa.Column
    dt = Check(_is_datetime, element_wise=False, error="must be a datetime column")
    num = Check(_is_numeric, element_wise=False, error="must be numeric")
    icao = Check.str_matches(r"^[A-Z0-9]{3,4}$")

    intermediate_flights = pa.DataFrameSchema(
        {c: Column(nullable=True, required=True) for c in REQUIRED_VRA_COLUMNS}
        | {c: Column(nullable=True, checks=dt) for c in
           ("scheduled_departure", "actual_departure", "scheduled_arrival")},
        strict=False, coerce=False,
    )
    primary_flights = pa.DataFrameSchema(
        {
            "origin_icao": Column(nullable=False),
            "destination_icao": Column(nullable=False),
            "scheduled_departure": Column(nullable=False, checks=dt),
            "actual_departure": Column(nullable=False, checks=dt),
            DELAY_MIN: Column(nullable=False, checks=num),
            "departure_date": Column(nullable=False, checks=dt),
        },
        strict=False,
    )
    weather = pa.DataFrameSchema(
        {
            "airport_icao": Column(nullable=False, checks=icao),
            "date": Column(nullable=False, checks=dt),
            "precip_mm": Column(nullable=True, checks=[num, Check.ge(0)]),
            "temp_mean_c": Column(nullable=True, checks=[num, Check.in_range(-60, 60)]),
            "wind_speed_kmh": Column(nullable=True, checks=[num, Check.ge(0)]),
        }
        | {c: Column(nullable=True, checks=num) for c in WEATHER_MEASURES
           if c not in {"precip_mm", "temp_mean_c", "wind_speed_kmh"}},
        unique=["airport_icao", "date"], strict=False,
    )
    model_input = pa.DataFrameSchema(
        {
            TARGET: Column(nullable=False, checks=Check.isin([0, 1])),
            "scheduled_departure": Column(nullable=False, checks=dt),
            "origin_hourly_flow": Column(nullable=False, checks=[num, Check.ge(1)]),
            "dep_hour": Column(nullable=False, checks=Check.in_range(0, 23)),
        },
        strict=False,
    )
    return {"intermediate_flights": intermediate_flights, "primary_flights": primary_flights,
            "weather": weather, "model_input": model_input}


def validate(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Validate `df` against the named contract; all failures are reported at once."""
    schemas = _schemas()
    if not schemas:
        warnings.warn("pandera is not installed: data contracts are NOT being checked", stacklevel=2)
        return df
    return schemas[name].validate(df, lazy=True)
