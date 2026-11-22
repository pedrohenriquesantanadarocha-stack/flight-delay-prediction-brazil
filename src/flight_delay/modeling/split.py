"""Train/test split and CV splitters.

"by_day":   no day appears in both train and test (flights of the same day and
            airport influence each other and would leak across partitions).
"temporal": train on the past, test on the future.
"random":   stratified random split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

from ..domain.columns import TARGET

STRATEGIES = ("by_day", "temporal", "random")


def train_test_indices(df: pd.DataFrame, strategy: str, test_size: float, random_state: int):
    """Return (train_idx, test_idx, group_per_row); groups are day ids."""
    y = df[TARGET].to_numpy()
    groups = pd.factorize(df["date"])[0]
    if strategy == "temporal":
        days = np.sort(df["date"].unique())
        cut = days[int(len(days) * (1 - test_size))]
        train = np.where(df["date"] < cut)[0]
        test = np.where(df["date"] >= cut)[0]
    elif strategy == "by_day":
        k = max(2, round(1 / test_size))
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=random_state)
        train, test = next(sgkf.split(df, y, groups))
    elif strategy == "random":
        train, test = train_test_split(np.arange(len(df)), test_size=test_size,
                                       stratify=y, random_state=random_state)
    else:
        raise ValueError(f"split strategy must be one of {STRATEGIES}")
    return train, test, groups


def cv_splitter(strategy: str, n_splits: int, random_state: int):
    if strategy == "random":
        return StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
    return StratifiedGroupKFold(n_splits, shuffle=True, random_state=random_state)
