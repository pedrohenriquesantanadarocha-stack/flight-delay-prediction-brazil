"""Synthetic ANAC VRA files and NOAA weather, plus fake sources for the ports.

Delays depend on the hour, the airport and the origin precipitation, so models
have a real signal to learn and the full pipeline can be exercised offline.
"""

from __future__ import annotations

import calendar
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

AIRPORTS = ["SBGR", "SBSP", "SBRJ", "SBKP", "SBBR", "SBCF", "SBPA", "SBSV"]
AIRLINES = ["GLO", "TAM", "AZU"]
HEADER_NOTE = "Dados abertos ANAC - arquivo sintético para testes"


def _seed(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode())


def _precip(airport: str, day: pd.Timestamp) -> float:
    rng = np.random.default_rng(_seed(airport, day.toordinal()))
    return float(np.round(rng.gamma(0.6, 6.0) * (rng.random() < 0.45), 2))


def weather_frame(airports, year: int) -> pd.DataFrame:
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    rows = []
    for a in airports:
        if a.endswith("X"):  # an airport without a weather station
            continue
        for d in days:
            rng = np.random.default_rng(_seed(a, d.toordinal(), "w"))
            t = 22 + 6 * np.sin(2 * np.pi * (d.dayofyear - 20) / 365) + rng.normal(0, 2)
            rows.append({"airport_icao": a, "date": d.strftime("%Y-%m-%d"),
                         "temp_mean_c": round(t, 1), "dewpoint_c": round(t - abs(rng.normal(5, 2)), 1),
                         "temp_max_c": round(t + 5, 1), "temp_min_c": round(t - 5, 1),
                         "precip_mm": _precip(a, d), "wind_speed_kmh": round(abs(rng.normal(12, 5)), 1),
                         "wind_gust_kmh": round(abs(rng.normal(30, 8)), 1) if rng.random() < 0.6 else np.nan})
    return pd.DataFrame(rows)


def vra_frame(year: int, month: int, n: int, seed: int = 0, iso_dates: bool = False) -> pd.DataFrame:
    """Raw VRA table with ANAC (Portuguese) headers and string timestamps."""
    rng = np.random.default_rng(seed + year * 100 + month)
    n_days = calendar.monthrange(year, month)[1]
    day = rng.integers(1, n_days + 1, n)
    hour = rng.choice(np.arange(5, 24), n, p=np.r_[np.full(4, 0.08), np.full(15, 0.68 / 15)])
    minute = rng.choice([0, 15, 30, 45], n)
    sched = pd.to_datetime(dict(year=year, month=month, day=day, hour=hour, minute=minute))
    orig = rng.choice(AIRPORTS + ["SBXX"], n)
    dest = np.array([rng.choice([a for a in AIRPORTS if a != o]) for o in orig])
    dur = rng.integers(50, 240, n)
    precip = np.array([_precip(o, d.normalize()) for o, d in zip(orig, sched)])
    logit = -2.3 + 0.08 * (hour - 12) + 0.07 * precip + 0.5 * (orig == "SBGR") + rng.normal(0, 0.6, n)
    delayed = rng.random(n) < 1 / (1 + np.exp(-logit))
    delay = np.where(delayed, rng.integers(30, 180, n), rng.integers(-10, 25, n))
    actual = sched + pd.to_timedelta(delay, unit="min")
    cancelled = rng.random(n) < 0.03
    fmt = "%Y-%m-%d %H:%M:%S" if iso_dates else "%d/%m/%Y %H:%M"
    df = pd.DataFrame({
        "Sigla ICAO Empresa Aérea": rng.choice(AIRLINES, n),
        "Número Voo": rng.integers(1000, 9999, n).astype(str),
        "Código Autorização (DI)": "0",
        "Código Tipo Linha": "N",
        "ICAO Aeródromo Origem": orig,
        "ICAO Aeródromo Destino": dest,
        "Partida Prevista": sched.dt.strftime(fmt),
        "Partida Real": pd.Series(actual.dt.strftime(fmt)).where(~cancelled),
        "Chegada Prevista": (sched + pd.to_timedelta(dur, unit="min")).dt.strftime(fmt),
        "Chegada Real": (actual + pd.to_timedelta(dur, unit="min")).dt.strftime(fmt),
        "Situação Voo": np.where(cancelled, "CANCELADO", "REALIZADO"),
        "Código Justificativa": "",
    })
    # A few flights scheduled in the previous month appear in the file (they must be dropped)
    df.loc[: max(1, n // 200), "Partida Prevista"] = (pd.Timestamp(year, month, 1)
                                                      - pd.Timedelta(hours=2)).strftime(fmt)
    return df


def write_vra_csv(df: pd.DataFrame, path: Path) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        f.write(HEADER_NOTE + "\n")
        df.to_csv(f, sep=";", index=False)
    return path


class FakeFlightSource:
    def __init__(self, n_per_month: int = 1500, missing=()):
        self.n = n_per_month
        self.missing = set(missing)
        self.calls: list[tuple[int, int]] = []

    def download_month(self, year, month, dest_dir):
        self.calls.append((year, month))
        if (year, month) in self.missing:
            return None
        return write_vra_csv(vra_frame(year, month, self.n), Path(dest_dir) / f"VRA_{year}{month}.csv")


class FakeWeatherSource:
    def __init__(self):
        self.calls: list[tuple[tuple[str, ...], int]] = []

    def fetch(self, airports, year):
        self.calls.append((tuple(airports), year))
        return weather_frame(airports, year)
