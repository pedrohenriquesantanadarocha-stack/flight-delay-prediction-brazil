"""NOAA GSOD daily weather from BigQuery public datasets.

Authentication uses Google Application Default Credentials (ADC):
  * locally:  `gcloud auth application-default login` once (see README);
  * Colab:    `google.colab.auth.authenticate_user()` (done by adapters.colab.setup).
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

GSOD_QUERY = """
WITH clean AS (
  SELECT
    s.call AS airport_icao,
    DATE(CAST(g.year AS INT64), CAST(g.mo AS INT64), CAST(g.da AS INT64)) AS date,
    IF(SAFE_CAST(g.temp AS FLOAT64) = 9999.9, NULL, SAFE_CAST(g.temp AS FLOAT64)) AS temp_f,
    IF(SAFE_CAST(g.dewp AS FLOAT64) = 9999.9, NULL, SAFE_CAST(g.dewp AS FLOAT64)) AS dewp_f,
    IF(SAFE_CAST(g.max  AS FLOAT64) = 9999.9, NULL, SAFE_CAST(g.max  AS FLOAT64)) AS max_f,
    IF(SAFE_CAST(g.min  AS FLOAT64) = 9999.9, NULL, SAFE_CAST(g.min  AS FLOAT64)) AS min_f,
    IF(SAFE_CAST(g.prcp AS FLOAT64) = 99.99,  NULL, SAFE_CAST(g.prcp AS FLOAT64)) AS prcp_in,
    IF(SAFE_CAST(g.wdsp AS FLOAT64) = 999.9,  NULL, SAFE_CAST(g.wdsp AS FLOAT64)) AS wdsp_knots,
    IF(SAFE_CAST(g.gust AS FLOAT64) = 999.9,  NULL, SAFE_CAST(g.gust AS FLOAT64)) AS gust_knots
  FROM `bigquery-public-data.noaa_gsod.gsod*` g
  JOIN `bigquery-public-data.noaa_gsod.stations` s
    ON g.stn = s.usaf AND SAFE_CAST(g.wban AS STRING) = SAFE_CAST(s.wban AS STRING)
  WHERE _TABLE_SUFFIX IN UNNEST(@years)
    AND s.call IS NOT NULL
    AND s.call IN UNNEST(@airports)
)
SELECT
  airport_icao,
  CAST(date AS STRING) AS date,
  ROUND((AVG(temp_f) - 32) * 5 / 9, 1) AS temp_mean_c,
  ROUND((AVG(dewp_f) - 32) * 5 / 9, 1) AS dewpoint_c,
  ROUND((AVG(max_f)  - 32) * 5 / 9, 1) AS temp_max_c,
  ROUND((AVG(min_f)  - 32) * 5 / 9, 1) AS temp_min_c,
  ROUND(AVG(prcp_in) * 25.4, 2)        AS precip_mm,
  ROUND(AVG(wdsp_knots) * 1.852, 1)    AS wind_speed_kmh,
  ROUND(AVG(gust_knots) * 1.852, 1)    AS wind_gust_kmh
FROM clean
GROUP BY airport_icao, date
"""


class BigQueryGsodSource:
    """Implements `ports.WeatherSource`."""

    def __init__(self, project: str | None = None, fallback_project: str | None = None, credentials=None):
        self._project = project
        self._fallback = fallback_project
        self._credentials = credentials

    def _client(self):
        try:
            import google.auth
            from google.cloud import bigquery
        except ImportError as e:
            raise ImportError("Weather needs the 'gcp' extra: uv sync --extra gcp") from e
        credentials, adc_project = self._credentials, None
        if credentials is None:
            credentials, adc_project = google.auth.default()
        project = self._project or adc_project or self._fallback
        if not project:
            raise RuntimeError("No GCP project: set weather.gcp_project in the config "
                               "or run `gcloud config set project <id>`.")
        return bigquery, bigquery.Client(credentials=credentials, project=project)

    def fetch(self, airports: list[str], year: int) -> pd.DataFrame:
        bigquery, client = self._client()
        job_config = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ArrayQueryParameter("airports", "STRING", sorted(airports)),
            bigquery.ArrayQueryParameter("years", "STRING", [str(year)]),
        ])
        log.info("BigQuery GSOD: %d airports, year %d", len(airports), year)
        return client.query(GSOD_QUERY, job_config=job_config).to_dataframe()
