import numpy as np
import pandas as pd
import pytest

from flight_delay.domain.columns import TARGET
from flight_delay.domain.sampling import sample_per_month
from flight_delay.domain.vra import clean_flights, standardize_vra
from flight_delay.domain.weather import join_weather, standardize_weather
from flight_delay.features.build import add_target, add_traffic, build_month_features
from flight_delay.features.contract import (
    FEATURE_GROUPS,
    FORBIDDEN,
    LeakageError,
    check_no_leakage,
    to_model_frame,
)

from synthetic import AIRPORTS, vra_frame, weather_frame


@pytest.fixture(scope="module")
def flights():
    return clean_flights(standardize_vra(vra_frame(2025, 1, 1200)), 2025, 1)


@pytest.fixture(scope="module")
def weather():
    return standardize_weather(weather_frame(AIRPORTS, 2025))


def test_no_feature_group_contains_post_departure_information():
    used = [c for cols in FEATURE_GROUPS.values() for c in cols]
    assert not set(used) & FORBIDDEN
    check_no_leakage(used)


def test_leakage_guard_raises():
    with pytest.raises(LeakageError):
        check_no_leakage(["dep_hour", "actual_departure"])


def test_target_threshold_is_inclusive(flights):
    df = add_target(flights, 30)
    assert (df.loc[df["departure_delay_min"] >= 30, TARGET] == 1).all()
    assert (df.loc[df["departure_delay_min"] < 30, TARGET] == 0).all()


def test_hourly_flow_is_computed_before_sampling(flights):
    full = add_traffic(flights)
    sample = sample_per_month(full, 100, 0)
    hour = sample["scheduled_departure"].dt.hour
    recount = sample.assign(_h=hour).groupby(["origin_icao", "departure_date", "_h"])["origin_icao"] \
        .transform("count")
    assert (sample["origin_hourly_flow"] >= recount).all()
    assert (sample["origin_hourly_flow"] > recount).any()


def test_weather_join_is_left_and_keeps_row_order(flights, weather):
    out = join_weather(flights, weather)
    assert len(out) == len(flights)
    assert out["scheduled_departure"].equals(flights["scheduled_departure"])
    no_station = out["origin_icao"] == "SBXX"
    assert out.loc[no_station, "origin_precip_mm"].isna().all()
    assert out.loc[~no_station, "origin_precip_mm"].notna().all()


def test_destination_weather_uses_arrival_day_even_across_years(weather):
    weather_2026 = standardize_weather(weather_frame(AIRPORTS, 2026))
    both = pd.concat([weather, weather_2026], ignore_index=True)
    f = pd.DataFrame({"origin_icao": ["SBGR"], "destination_icao": ["SBRJ"],
                      "departure_date": [pd.Timestamp("2025-12-31")], "arrival_date": [pd.Timestamp("2026-01-01")]})
    out = join_weather(f, both)
    w = weather_2026
    expected = w.loc[(w["airport_icao"] == "SBRJ") & (w["date"] == pd.Timestamp("2026-01-01")), "precip_mm"].item()
    assert out["destination_precip_mm"].item() == expected


def test_model_frame_types(flights, weather):
    feats = build_month_features(flights, weather, 30)
    df, groups, num, cat = to_model_frame(feats)
    assert list(groups) == list(FEATURE_GROUPS)
    assert set(cat) == {"airline_icao", "origin_icao", "destination_icao", "route"}
    assert all(df[c].dtype == np.float64 for c in num)
    assert all(df[c].map(type).eq(str).all() for c in cat)
    assert "date" in df and df[TARGET].isin([0, 1]).all()


def test_derived_weather_variables(weather):
    row = weather.iloc[0]
    assert row["temp_range_c"] == pytest.approx(row["temp_max_c"] - row["temp_min_c"])
    assert row["dewpoint_depression_c"] == pytest.approx(row["temp_mean_c"] - row["dewpoint_c"])
