"""Column contract: how ANAC/NOAA names map to the internal (English) names.

This is the single place where source names appear. Everything downstream uses
the internal names; reports translate them back to Portuguese with
`flight_delay.reporting.labels`.
"""

from __future__ import annotations

# --- ANAC VRA: source name -> internal name ------------------------------------
# Several source names map to the same internal name because ANAC changed
# headers across years.
VRA_RENAME: dict[str, str] = {
    "ICAO Empresa Aérea": "airline_icao",
    "Sigla ICAO Empresa Aérea": "airline_icao",
    "Número Voo": "flight_number",
    "Código Autorização (DI)": "authorization_code",
    "Código Tipo Linha": "line_type_code",
    "ICAO Aeródromo Origem": "origin_icao",
    "Sigla ICAO Aeroporto Origem": "origin_icao",
    "ICAO Aeródromo Destino": "destination_icao",
    "Sigla ICAO Aeroporto Destino": "destination_icao",
    "Partida Prevista": "scheduled_departure",
    "Partida Real": "actual_departure",
    "Chegada Prevista": "scheduled_arrival",
    "Chegada Real": "actual_arrival",
    "Situação Voo": "flight_status",
    "Situacao Voo": "flight_status",
    "Código Justificativa": "justification_code",
}

DATETIME_COLUMNS = ("scheduled_departure", "actual_departure", "scheduled_arrival", "actual_arrival")
REQUIRED_VRA_COLUMNS = ("origin_icao", "destination_icao", "scheduled_departure",
                        "actual_departure", "scheduled_arrival")

# --- NOAA GSOD (after the SQL in adapters/noaa_bigquery.py) -------------------
WEATHER_KEYS = ("airport_icao", "date")
WEATHER_MEASURES = (
    "temp_mean_c", "dewpoint_c", "temp_max_c", "temp_min_c",
    "precip_mm", "wind_speed_kmh", "wind_gust_kmh",
)
# Weather variables actually used as features (derived ones included)
WEATHER_FEATURES = (
    "precip_mm", "wind_speed_kmh", "wind_gust_kmh",
    "dewpoint_depression_c", "temp_range_c", "temp_mean_c",
)

TARGET = "target"
DELAY_MIN = "departure_delay_min"
