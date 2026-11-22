"""Label and pre-departure features for one month of realised flights."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..domain.columns import DELAY_MIN, TARGET
from ..domain.weather import join_weather

HIGH_SEASON_MONTHS = (1, 7, 12)


def add_target(df: pd.DataFrame, threshold_min: int) -> pd.DataFrame:
    out = df.copy()
    out[TARGET] = (out[DELAY_MIN] >= threshold_min).astype(int)
    return out


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dep = out["scheduled_departure"]
    out["year"] = dep.dt.year
    out["month"] = dep.dt.month
    out["dep_hour"] = dep.dt.hour
    out["day_of_week"] = dep.dt.dayofweek  # 0 = Monday
    out["high_season"] = out["month"].isin(HIGH_SEASON_MONTHS).astype(int)
    out["weekend"] = (out["day_of_week"] >= 5).astype(int)
    for name, value, period in (("hour", out["dep_hour"], 24),
                                ("dow", out["day_of_week"], 7),
                                ("month", out["month"] - 1, 12)):
        out[f"{name}_sin"] = np.sin(2 * np.pi * value / period)
        out[f"{name}_cos"] = np.cos(2 * np.pi * value / period)
    return out


def add_traffic(df: pd.DataFrame) -> pd.DataFrame:
    """Scheduled flights per origin airport, day and hour, plus scheduled duration.

    IMPORTANT: must run on the FULL month, before sampling; otherwise the flow is
    underestimated and no longer measures the real congestion.
    """
    out = df.copy()
    hour = out["scheduled_departure"].dt.hour
    out["origin_hourly_flow"] = (
        out.assign(_h=hour).groupby(["origin_icao", "departure_date", "_h"])["origin_icao"].transform("count")
    )
    dur = (out["scheduled_arrival"] - out["scheduled_departure"]).dt.total_seconds() / 60.0
    out["scheduled_duration_min"] = dur.where((dur > 0) & (dur < 1440))
    return out


def add_route(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["route"] = out["origin_icao"].astype(str) + "-" + out["destination_icao"].astype(str)
    return out


def build_month_features(flights: pd.DataFrame, weather: pd.DataFrame, threshold_min: int) -> pd.DataFrame:
    """Feature table for every flight of one month (feature layer)."""
    df = add_target(flights, threshold_min)
    df = add_calendar(df)
    df = add_traffic(df)
    df = add_route(df)
    df = join_weather(df, weather)
    df["n_flights_month_total"] = len(df)  # lets the paper describe/weight the sampling
    return df
