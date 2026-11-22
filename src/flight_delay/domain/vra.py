"""ANAC VRA rules: standardise the raw table and keep valid, realised flights."""

from __future__ import annotations

import logging

import pandas as pd

from .columns import DATETIME_COLUMNS, DELAY_MIN, REQUIRED_VRA_COLUMNS, VRA_RENAME

log = logging.getLogger(__name__)


def fix_encoding(text):
    """Undo UTF-8 read as Latin-1 ('AÃ©rea' -> 'Aérea'); leave other values untouched."""
    if isinstance(text, str):
        try:
            return text.encode("latin1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text
    return text


def parse_datetimes(series: pd.Series) -> pd.Series:
    """Parse VRA timestamps, whose format changes across years.

    ISO (2025-01-05 14:30:00) must NOT use dayfirst=True, which swaps day and month.
    Brazilian (05/01/2025 14:30) needs dayfirst=True. If a column mixes values with
    and without seconds, the format inferred from the first row would turn the rest
    into NaT, so it retries with format="mixed" when many values fail.
    """
    s = series.astype("string").str.strip()
    head = s.dropna().head(500)
    if len(head) and head.str.match(r"^\d{4}-\d{2}-\d{2}").all():
        return pd.to_datetime(s, format="ISO8601", errors="coerce")
    out = pd.to_datetime(s, dayfirst=True, errors="coerce")
    n_valid = s.notna().sum()
    if n_valid and out.isna().sum() - s.isna().sum() > 0.01 * n_valid:
        out = pd.to_datetime(s, dayfirst=True, format="mixed", errors="coerce")
    return out


def standardize_vra(raw: pd.DataFrame) -> pd.DataFrame:
    """Typed mirror of the raw table (intermediate layer).

    Renames source headers to internal names, keeps only the mapped columns and
    parses timestamps. No rows are dropped here.
    """
    df = raw.copy()
    df.columns = [fix_encoding(c).strip() for c in df.columns]
    present = {c: VRA_RENAME[c] for c in df.columns if c in VRA_RENAME}
    dropped = sorted(set(df.columns) - set(present))
    if dropped:
        log.debug("VRA columns not in the contract (dropped): %s", dropped)
    df = df[list(present)].rename(columns=present)
    df = df.loc[:, ~df.columns.duplicated()]
    missing = [c for c in REQUIRED_VRA_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"VRA file lacks required columns {missing}; got {list(df.columns)}")
    for c in DATETIME_COLUMNS:
        if c in df.columns:
            df[c] = parse_datetimes(df[c])
    for c in df.columns:
        if c not in DATETIME_COLUMNS:
            df[c] = df[c].astype("string")
    return df.reset_index(drop=True)


IMPLAUSIBLE_DELAY_MIN = 7 * 24 * 60


def count_implausible_delays(flights: pd.DataFrame, limit_min: float = IMPLAUSIBLE_DELAY_MIN) -> int:
    """Flights whose |departure delay| exceeds `limit_min` (usually registration errors)."""
    return int((flights[DELAY_MIN].abs() > limit_min).sum())


def drop_extreme_delays(df: pd.DataFrame, max_abs_delay_min: float | None) -> pd.DataFrame:
    """Remove flights with |delay| above the limit; None keeps every flight."""
    if max_abs_delay_min is None:
        return df
    return df[df[DELAY_MIN].abs() <= max_abs_delay_min].reset_index(drop=True)


def clean_flights(df: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    """Realised flights scheduled in (year, month), with the departure delay (primary layer).

    Flights without an actual departure (e.g. cancelled) are removed: report this
    in the paper.
    """
    out = df.dropna(subset=["scheduled_departure", "actual_departure", "origin_icao", "destination_icao"])
    out = out[(out["scheduled_departure"].dt.year == year) & (out["scheduled_departure"].dt.month == month)]
    out = out.copy()
    out[DELAY_MIN] = (out["actual_departure"] - out["scheduled_departure"]).dt.total_seconds() / 60.0
    out["departure_date"] = out["scheduled_departure"].dt.normalize()
    out["arrival_date"] = out["scheduled_arrival"].dt.normalize()
    return out.reset_index(drop=True)
