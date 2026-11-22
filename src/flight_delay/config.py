"""Typed, immutable configuration loaded from one or more YAML files.

Files are deep-merged in order, so a profile only lists what it changes. Unknown
keys raise an error instead of being silently ignored (a typo in a YAML would
otherwise run the wrong experiment).
"""

from __future__ import annotations

import dataclasses
import typing
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathsConfig:
    data_root: str = "data"
    raw_tmp: str | None = None


@dataclass(frozen=True)
class StorageConfig:
    format: str = "parquet"


@dataclass(frozen=True)
class SourceConfig:
    anac_base_url: str = (
        "https://sistemas.anac.gov.br/dadosabertos/Voos e operações aéreas/Voo Regular Ativo (VRA)"
    )
    timeout_s: int = 120
    retries: int = 3
    keep_raw: bool = False


@dataclass(frozen=True)
class ScopeConfig:
    years: tuple[int, ...] = (2024, 2025)
    months: tuple[int, ...] = tuple(range(1, 13))
    delay_threshold_min: int = 30
    max_flights_per_month: int | None = 20_000
    random_state: int = 42

    @property
    def sample_tag(self) -> str:
        """Identifies the parameters that change the modelling sample."""
        n = "all" if self.max_flights_per_month is None else self.max_flights_per_month
        return f"thr{self.delay_threshold_min}_n{n}_rs{self.random_state}"


@dataclass(frozen=True)
class WeatherConfig:
    gcp_project: str | None = None
    gcp_project_fallback: str | None = None


@dataclass(frozen=True)
class TuningConfig:
    enabled: bool = True
    n_iter: int = 10
    cv: int = 3
    max_sample: int = 120_000
    scoring: str = "average_precision"


@dataclass(frozen=True)
class ModelingConfig:
    split_strategy: str = "by_day"
    test_size: float = 0.2
    cv_folds: int = 5
    n_jobs: int = -1
    models: tuple[str, ...] = (
        "dummy", "logistic_regression", "decision_tree", "random_forest",
        "extra_trees", "hist_gradient_boosting", "xgboost", "lightgbm",
    )
    reference_model: str = "logistic_regression"
    target_encoding_smoothing: float = 10.0
    balance_classes: bool = True
    tuning: TuningConfig = field(default_factory=TuningConfig)


@dataclass(frozen=True)
class EvaluationConfig:
    beta: float = 2.0
    recall_target: float = 0.70
    cost_ratio_fn_fp: float = 3.0
    n_boot: int = 200
    n_diagnostic_sample: int = 30_000
    n_permutation_repeats: int = 5
    learning_curve: bool = True
    top_k: float = 0.10


@dataclass(frozen=True)
class Settings:
    paths: PathsConfig = field(default_factory=PathsConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    scope: ScopeConfig = field(default_factory=ScopeConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    modeling: ModelingConfig = field(default_factory=ModelingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @property
    def random_state(self) -> int:
        return self.scope.random_state

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ------------------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------------------
def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce(tp: Any, value: Any, where: str) -> Any:
    origin = typing.get_origin(tp)
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, Mapping):
            raise TypeError(f"{where}: expected a mapping, got {type(value).__name__}")
        return _build(tp, value, where)
    if origin is tuple:
        return tuple(value)
    return value


def _build(cls: type, data: Mapping[str, Any], where: str = "config"):
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise KeyError(f"{where}: unknown keys {sorted(unknown)} (valid: {sorted(names)})")
    kwargs = {k: _coerce(hints[k], v, f"{where}.{k}") for k, v in data.items()}
    return cls(**kwargs)


def settings_from_dict(data: Mapping[str, Any] | None) -> Settings:
    settings = _build(Settings, data or {})
    validate(settings)
    return settings


def load_settings(paths: str | Path | Iterable[str | Path], overrides: Mapping[str, Any] | None = None) -> Settings:
    """Load and deep-merge YAML files (later files win), then apply `overrides`."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    merged: dict[str, Any] = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            merged = deep_merge(merged, yaml.safe_load(f) or {})
    if overrides:
        merged = deep_merge(merged, overrides)
    return settings_from_dict(merged)


def validate(s: Settings) -> None:
    if s.storage.format not in {"parquet", "pickle"}:
        raise ValueError("storage.format must be 'parquet' or 'pickle'")
    if s.modeling.split_strategy not in {"by_day", "temporal", "random"}:
        raise ValueError("modeling.split_strategy must be 'by_day', 'temporal' or 'random'")
    if not 0 < s.modeling.test_size < 1:
        raise ValueError("modeling.test_size must be in (0, 1)")
    bad_months = [m for m in s.scope.months if not 1 <= m <= 12]
    if bad_months:
        raise ValueError(f"scope.months out of range: {bad_months}")
