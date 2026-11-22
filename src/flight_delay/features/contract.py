"""Feature contract: groups, types, and what may never be a feature.

The groups drive the permutation importance by group and the ablation study.
Their order is part of the contract (it fixes the column order seen by the models).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..domain.columns import DELAY_MIN, TARGET, WEATHER_FEATURES

FEATURE_GROUPS: dict[str, list[str]] = {
    "calendar": ["dep_hour", "day_of_week", "high_season", "weekend",
                 "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"],
    "traffic": ["origin_hourly_flow", "scheduled_duration_min"],
    "airline_route": ["airline_icao", "origin_icao", "destination_icao", "route"],
    "weather_origin": [f"origin_{c}" for c in WEATHER_FEATURES],
    "weather_destination": [f"destination_{c}" for c in WEATHER_FEATURES],
}
CATEGORICAL = frozenset({"airline_icao", "origin_icao", "destination_icao", "route"})

# Known only AFTER the departure: must never become a feature.
FORBIDDEN = frozenset({
    "actual_departure", "actual_arrival", DELAY_MIN, TARGET,
    "justification_code", "flight_status",
})


class LeakageError(ValueError):
    pass


def check_no_leakage(features: list[str]) -> None:
    leaked = sorted(set(features) & FORBIDDEN)
    if leaked:
        raise LeakageError(f"Variables only known after departure: {leaked}")


def resolve_groups(df: pd.DataFrame, groups: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
    """Groups restricted to the columns present in `df` (empty groups dropped)."""
    groups = groups or FEATURE_GROUPS
    out = {g: [c for c in cols if c in df.columns] for g, cols in groups.items()}
    return {g: cols for g, cols in out.items() if cols}


def to_model_frame(df: pd.DataFrame, groups: dict[str, list[str]] | None = None):
    """Typed modelling frame. Returns (df, groups, numeric, categorical).

    Numeric columns become float64 (partial dependence rejects integers) and
    categoricals become strings with missing values as "NA". Adds `date`, the
    day used to group the train/test split.
    """
    df = df.copy()
    df["scheduled_departure"] = pd.to_datetime(df["scheduled_departure"], errors="coerce")
    df = df.dropna(subset=[TARGET, "scheduled_departure"]).reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)
    df["date"] = df["scheduled_departure"].dt.normalize()

    groups = resolve_groups(df, groups)
    used = [c for cols in groups.values() for c in cols]
    check_no_leakage(used)
    cat = [c for c in used if c in CATEGORICAL]
    num = [c for c in used if c not in CATEGORICAL]
    for c in cat:
        df[c] = df[c].fillna("NA").astype(str).astype(object)
    df[num] = df[num].apply(pd.to_numeric, errors="coerce").astype(np.float64)
    return df, groups, num, cat
