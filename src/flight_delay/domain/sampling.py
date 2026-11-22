"""Sampling to control the computational cost."""

from __future__ import annotations

import pandas as pd


def sample_per_month(df: pd.DataFrame, max_per_month: int | None, random_state: int) -> pd.DataFrame:
    """Uniform random sample of at most `max_per_month` flights per calendar month.

    Stratifying by month keeps the seasonality. Rows are shuffled even when the
    month has fewer flights than the cap (same behaviour as the original code).
    """
    if max_per_month is None:
        return df.reset_index(drop=True)
    month = df["scheduled_departure"].dt.month
    parts = [g.sample(min(len(g), max_per_month), random_state=random_state)
             for _, g in df.groupby(month)]
    return pd.concat(parts).reset_index(drop=True)
