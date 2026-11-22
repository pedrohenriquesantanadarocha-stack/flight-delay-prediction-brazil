"""Layered data catalogue (Kedro data-engineering convention).

    01_raw           source files exactly as downloaded
    02_intermediate  typed mirror of raw (internal column names, parsed dates)
    03_primary       domain tables: realised flights, daily weather
    04_feature       features + label for every flight of a month
    05_model_input   the sampled modelling table
    06_models        (unused: models are not persisted)
    07_model_output  out-of-sample predictions
    08_reporting     tables, figures, LaTeX, run summary

Writes are atomic (temporary file + rename), so an interrupted run never leaves
a half-written file that a later run would mistake for a finished step.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

LAYERS = {
    "raw": "01_raw",
    "intermediate": "02_intermediate",
    "primary": "03_primary",
    "feature": "04_feature",
    "model_input": "05_model_input",
    "models": "06_models",
    "model_output": "07_model_output",
    "reporting": "08_reporting",
}
_EXT = {"parquet": ".parquet", "pickle": ".pkl"}


class DataCatalog:
    def __init__(self, root: str | Path, fmt: str = "parquet", raw_dir: str | Path | None = None):
        if fmt not in _EXT:
            raise ValueError(f"unsupported storage format {fmt!r}")
        self.root = Path(root)
        self.fmt = fmt
        self._raw_dir = Path(raw_dir) if raw_dir else None

    # --- paths -------------------------------------------------------------------
    def dir(self, layer: str, *parts: str) -> Path:
        base = self._raw_dir if (layer == "raw" and self._raw_dir) else self.root / LAYERS[layer]
        p = base.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def table_path(self, layer: str, name: str) -> Path:
        path = self.root / LAYERS[layer] / f"{name}{_EXT[self.fmt]}"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def exists(self, layer: str, name: str) -> bool:
        return self.table_path(layer, name).exists()

    # --- tables ------------------------------------------------------------------
    def read(self, layer: str, name: str) -> pd.DataFrame:
        path = self.table_path(layer, name)
        if self.fmt == "parquet":
            return pd.read_parquet(path)
        return pd.read_pickle(path)

    def write(self, layer: str, name: str, df: pd.DataFrame) -> Path:
        path = self.table_path(layer, name)
        tmp = path.with_suffix(path.suffix + ".part")
        if self.fmt == "parquet":
            df.to_parquet(tmp, index=False)
        else:
            df.to_pickle(tmp)
        os.replace(tmp, path)
        return path

    # --- small JSON side files (metadata, summaries) ------------------------------
    def read_json(self, layer: str, name: str) -> Any:
        with open(self.root / LAYERS[layer] / f"{name}.json", encoding="utf-8") as f:
            return json.load(f)

    def write_json(self, layer: str, name: str, obj: Any) -> Path:
        path = self.root / LAYERS[layer] / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.part")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)
        os.replace(tmp, path)
        return path

    def json_exists(self, layer: str, name: str) -> bool:
        return (self.root / LAYERS[layer] / f"{name}.json").exists()


def _json_default(o):
    import numpy as np

    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)
