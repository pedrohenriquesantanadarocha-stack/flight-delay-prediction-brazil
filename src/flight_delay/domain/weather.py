"""Daily weather: derived variables and the join to flights."""

from __future__ import annotations

import pandas as pd

from .columns import WEATHER_FEATURES, WEATHER_KEYS, WEATHER_MEASURES


def standardize_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Types the source table and adds derived variables (primary layer)."""
    out = df[list(WEATHER_KEYS + WEATHER_MEASURES)].copy()
    out["airport_icao"] = out["airport_icao"].astype("string")
    out["date"] = pd.to_datetime(out["date"]).astype("datetime64[ns]")
    for c in WEATHER_MEASURES:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    out["temp_range_c"] = out["temp_max_c"] - out["temp_min_c"]
    out["dewpoint_depression_c"] = out["temp_mean_c"] - out["dewpoint_c"]
    return out.sort_values(list(WEATHER_KEYS)).reset_index(drop=True)


def join_weather(flights: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Origin weather on the departure day and destination weather on the arrival day.

    Same-day daily means: they include hours after the departure, i.e. the model
    assumes a perfect weather forecast for the day (state this in the paper).
    Row order of `flights` is preserved.
    """
    w = weather[list(WEATHER_KEYS) + list(WEATHER_FEATURES)].copy()
    w["date"] = w["date"].astype("datetime64[ns]")
    out = flights.copy()
    for side, airport, day in (("origin", "origin_icao", "departure_date"),
                               ("destination", "destination_icao", "arrival_date")):
        prefixed = w.rename(columns={c: f"{side}_{c}" for c in WEATHER_FEATURES})
        prefixed = prefixed.rename(columns={"airport_icao": "_airport", "date": "_date"})
        left_day = out[day].astype("datetime64[ns]")
        out = (out.assign(_airport=out[airport].astype("string"), _date=left_day)
                  .merge(prefixed, on=["_airport", "_date"], how="left", validate="many_to_one")
                  .drop(columns=["_airport", "_date"]))
    return out
