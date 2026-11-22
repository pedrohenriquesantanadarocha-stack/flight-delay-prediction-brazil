"""ETL pipeline: raw sources to the primary layer."""

from __future__ import annotations

import logging

import pandas as pd

from ..adapters.anac import read_vra_csv
from ..adapters.storage import DataCatalog
from ..config import Settings
from ..domain.schemas import validate
from ..domain.vra import clean_flights, standardize_vra
from ..domain.weather import standardize_weather
from ..ports import FlightDataSource, WeatherSource

log = logging.getLogger(__name__)


def intermediate_name(year: int, month: int) -> str:
    return f"vra/vra_{year}_{month:02d}"


def flights_name(year: int, month: int) -> str:
    return f"flights/flights_{year}_{month:02d}"


def weather_name(year: int) -> str:
    return f"weather/weather_{year}"


def ingest_month(year: int, month: int, settings: Settings, catalog: DataCatalog,
                 source: FlightDataSource, force: bool = False) -> bool:
    """Download one month and write its intermediate and primary tables. Returns True if available."""
    if catalog.exists("primary", flights_name(year, month)) and not force:
        return True

    if catalog.exists("intermediate", intermediate_name(year, month)) and not force:
        typed = catalog.read("intermediate", intermediate_name(year, month))
    else:
        raw_path = source.download_month(year, month, catalog.dir("raw", "vra"))
        if raw_path is None:
            return False
        typed = validate(standardize_vra(read_vra_csv(raw_path)), "intermediate_flights")
        catalog.write("intermediate", intermediate_name(year, month), typed)
        if not settings.source.keep_raw:
            raw_path.unlink(missing_ok=True)

    flights = validate(clean_flights(typed, year, month), "primary_flights")
    if flights.empty:
        log.warning("[%d-%02d] no valid flight (check date format/columns)", year, month)
        return False
    catalog.write("primary", flights_name(year, month), flights)
    log.info("[%d-%02d] raw %s | realised in month %s", year, month, f"{len(typed):,}", f"{len(flights):,}")
    return True


def airports_of_year(year: int, months, catalog: DataCatalog) -> list[str]:
    airports: set[str] = set()
    for m in months:
        if catalog.exists("primary", flights_name(year, m)):
            f = catalog.read("primary", flights_name(year, m))
            airports |= set(f["origin_icao"].dropna().astype(str)) | set(f["destination_icao"].dropna().astype(str))
    return sorted(airports)


def ingest_weather(year: int, airports: list[str], catalog: DataCatalog, source: WeatherSource,
                   force: bool = False) -> pd.DataFrame:
    """Daily weather of the year, cached. Queries again only when new airports appear."""
    name, meta = weather_name(year), f"{weather_name(year)}_airports"
    queried: set[str] = set()
    if catalog.exists("primary", name) and catalog.json_exists("primary", meta) and not force:
        queried = set(catalog.read_json("primary", meta))
        if set(airports) <= queried:
            return catalog.read("primary", name)
    target = sorted(queried | set(airports))
    weather = validate(standardize_weather(source.fetch(target, year)), "weather")
    catalog.write("primary", name, weather)
    catalog.write_json("primary", meta, target)
    log.info("[%d] weather: %d airports queried, %d with data", year, len(target), weather["airport_icao"].nunique())
    return weather


def run_etl(settings: Settings, catalog: DataCatalog, flight_source: FlightDataSource,
            weather_source: WeatherSource, force: bool = False) -> None:
    for year in settings.scope.years:
        log.info("ETL %d...", year)
        available = [m for m in settings.scope.months
                     if ingest_month(year, m, settings, catalog, flight_source, force)]
        if not available:
            log.warning("[%d] no data; skipping", year)
            continue
        ingest_weather(year, airports_of_year(year, available, catalog), catalog, weather_source, force)
