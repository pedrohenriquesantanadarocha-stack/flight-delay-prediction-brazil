import pandas as pd
import pytest

from flight_delay.adapters.anac import AnacVraSource, read_vra_csv
from flight_delay.domain.columns import DELAY_MIN
from flight_delay.domain.sampling import sample_per_month
from flight_delay.domain.vra import clean_flights, fix_encoding, parse_datetimes, standardize_vra

from synthetic import vra_frame, write_vra_csv


def test_iso_dates_are_not_parsed_day_first():
    s = pd.Series(["2025-01-05 14:30:00", "2025-02-10 08:00:00"])
    out = parse_datetimes(s)
    assert out.dt.month.tolist() == [1, 2] and out.dt.day.tolist() == [5, 10]


def test_brazilian_dates_are_parsed_day_first():
    out = parse_datetimes(pd.Series(["05/01/2025 14:30", "10/02/2025 08:00"]))
    assert out.dt.month.tolist() == [1, 2] and out.dt.day.tolist() == [5, 10]


def test_mixed_seconds_do_not_become_nat():
    out = parse_datetimes(pd.Series(["05/01/2025 14:30"] * 50 + ["06/01/2025 14:30:15"] * 50))
    assert out.notna().all()


def test_fix_encoding_repairs_mojibake():
    assert fix_encoding("ICAO Empresa AÃ©rea") == "ICAO Empresa Aérea"
    assert fix_encoding("Aérea") == "Aérea"


def test_standardize_maps_both_header_generations():
    new = vra_frame(2025, 1, 50)
    old = new.rename(columns={"ICAO Aeródromo Origem": "Sigla ICAO Aeroporto Origem",
                              "ICAO Aeródromo Destino": "Sigla ICAO Aeroporto Destino"})
    for raw in (new, old):
        df = standardize_vra(raw)
        assert {"origin_icao", "destination_icao", "airline_icao", "scheduled_departure"} <= set(df.columns)
        assert pd.api.types.is_datetime64_any_dtype(df["scheduled_departure"])


def test_missing_required_column_fails_loudly():
    raw = vra_frame(2025, 1, 20).drop(columns=["Partida Real"])
    with pytest.raises(KeyError, match="actual_departure"):
        standardize_vra(raw)


def test_read_csv_skips_preamble_and_detects_header(tmp_path):
    path = write_vra_csv(vra_frame(2025, 3, 30), tmp_path / "VRA_20253.csv")
    df = read_vra_csv(path)
    assert "Partida Prevista" in df.columns and len(df) == 30


def test_clean_flights_drops_cancelled_and_other_months():
    typed = standardize_vra(vra_frame(2025, 1, 400))
    out = clean_flights(typed, 2025, 1)
    assert out["actual_departure"].notna().all()
    assert (out["scheduled_departure"].dt.month == 1).all()
    assert len(out) < len(typed)
    expected = (out["actual_departure"] - out["scheduled_departure"]).dt.total_seconds() / 60
    assert (out[DELAY_MIN] == expected).all()


def test_sampling_is_reproducible_and_capped():
    df = clean_flights(standardize_vra(vra_frame(2025, 1, 500)), 2025, 1)
    a = sample_per_month(df, 100, 42)
    b = sample_per_month(df, 100, 42)
    assert len(a) == 100 and a.equals(b)
    assert len(sample_per_month(df, None, 42)) == len(df)


def test_anac_url_has_no_leading_zero_in_file_name():
    url = AnacVraSource("https://x/VRA").url(2017, 8)
    assert url.endswith("/2017/08%20-%20Agosto/VRA_20178.csv")
