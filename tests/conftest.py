import importlib.util
import sys
import warnings
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore", message="pandera is not installed")

ROOT = Path(__file__).resolve().parents[1]
STORAGE_FORMAT = "parquet" if importlib.util.find_spec("pyarrow") else "pickle"


@pytest.fixture
def settings_factory(tmp_path):
    from flight_delay.config import load_settings

    def make(**overrides):
        base = {
            "paths": {"data_root": str(tmp_path / "data")},
            "storage": {"format": STORAGE_FORMAT},
            "scope": {"years": [2025], "months": [1, 2], "max_flights_per_month": 900},
            "modeling": {"models": ["dummy", "logistic_regression", "decision_tree"], "cv_folds": 3,
                         "n_jobs": 1, "tuning": {"n_iter": 2}},
            "evaluation": {"n_boot": 10, "n_diagnostic_sample": 500, "n_permutation_repeats": 1},
        }
        from flight_delay.config import deep_merge

        return load_settings([ROOT / "configs" / "base.yaml"], overrides=deep_merge(base, overrides))

    return make
