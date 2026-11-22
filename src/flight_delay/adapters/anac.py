"""ANAC open data (Voo Regular Ativo, VRA): download and read the monthly CSV."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd

log = logging.getLogger(__name__)

MONTHS_PT = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril", 5: "Maio", 6: "Junho",
    7: "Julho", 8: "Agosto", 9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro",
}


class AnacVraSource:
    """Implements `ports.FlightDataSource` over HTTP."""

    def __init__(self, base_url: str, timeout_s: int = 120, retries: int = 3):
        self.base_url = base_url
        self.timeout_s = timeout_s
        self.retries = retries

    def url(self, year: int, month: int) -> str:
        """e.g. .../VRA/2017/08 - Agosto/VRA_20178.csv (file name has no leading zero)."""
        path = f"{self.base_url}/{year}/{month:02d} - {MONTHS_PT[month]}/VRA_{year}{month}.csv"
        return quote(path, safe=":/")

    def download_month(self, year: int, month: int, dest_dir: Path) -> Path | None:
        import requests

        dest = Path(dest_dir) / f"VRA_{year}{month}.csv"
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        url = self.url(year, month)
        for attempt in range(1, self.retries + 1):
            try:
                r = requests.get(url, stream=True, timeout=self.timeout_s,
                                 headers={"User-Agent": "Mozilla/5.0 (pesquisa academica)"})
                if r.status_code == 404:
                    log.warning("[%d-%02d] not published by ANAC (404): %s", year, month, url)
                    return None
                r.raise_for_status()
                tmp = dest.with_suffix(".csv.part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                os.replace(tmp, dest)
                return dest
            except requests.RequestException as e:
                log.warning("[%d-%02d] download failed (%d/%d): %s", year, month, attempt, self.retries, e)
                time.sleep(2 ** attempt)
        return None


def read_vra_csv(path: str | Path) -> pd.DataFrame:
    """Read a VRA CSV, detecting the encoding and the header line (as source names)."""
    for enc in ("utf-8-sig", "latin-1"):
        try:
            header_line = None
            with open(path, encoding=enc) as f:
                for i, line in enumerate(f):
                    if "partida prevista" in line.lower():
                        header_line = i
                        break
                    if i > 15:
                        break
            if header_line is None:
                continue
            return pd.read_csv(path, sep=";", encoding=enc, skiprows=header_line,
                               dtype=str, keep_default_na=True, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"VRA header not found in {path}")
