"""Ports: the interfaces the pipelines depend on (hexagonal architecture).

Concrete implementations live in `flight_delay.adapters`. Tests use fakes that
satisfy the same protocols, so no pipeline needs the network or a cloud account.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class FlightDataSource(Protocol):
    def download_month(self, year: int, month: int, dest_dir: Path) -> Path | None:
        """Download the raw monthly file. Return its path, or None if it is not published."""


@runtime_checkable
class WeatherSource(Protocol):
    def fetch(self, airports: list[str], year: int) -> pd.DataFrame:
        """Daily weather per airport for one year, with columns
        `columns.WEATHER_KEYS + columns.WEATHER_MEASURES` (date as 'YYYY-MM-DD')."""
