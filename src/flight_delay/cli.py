"""Command line interface.

    flight-delay run                                   # full experiment (configs/base.yaml)
    flight-delay run -c configs/base.yaml -c configs/test.yaml
    flight-delay run --stage features --stage modeling
    flight-delay check-env
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings

from .config import load_settings
from .pipelines.run import STAGES, run

DEFAULT_CONFIG = "configs/base.yaml"


def check_env() -> int:
    import pandas as pd
    import sklearn

    from .modeling.catalog import LGBMClassifier, XGBClassifier

    version = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    ok = version >= (1, 4)
    status = {
        "python": sys.version.split()[0], "scikit-learn": sklearn.__version__ + ("" if ok else "  (needs >= 1.4)"),
        "pandas": pd.__version__,
        "xgboost": "ok" if XGBClassifier else "MISSING (uv sync --extra boosting)",
        "lightgbm": "ok" if LGBMClassifier else "MISSING (uv sync --extra boosting)",
    }
    for mod, hint in (("pyarrow", "uv sync"), ("pandera", "uv sync"), ("google.cloud.bigquery", "uv sync --extra gcp")):
        try:
            __import__(mod)
            status[mod] = "ok"
        except ImportError:
            status[mod] = f"MISSING ({hint})"
    for k, v in status.items():
        print(f"{k:>22}: {v}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="flight-delay", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="run pipeline stages")
    p_run.add_argument("-c", "--config", action="append",
                       help=f"YAML file(s), deep-merged in order (default: {DEFAULT_CONFIG})")
    p_run.add_argument("-s", "--stage", action="append", choices=STAGES, help="stage(s) to run (default: all)")
    p_run.add_argument("--force", action="store_true", help="recompute outputs that already exist")
    p_run.add_argument("-v", "--verbose", action="store_true")
    sub.add_parser("check-env", help="show library versions and missing extras")
    args = parser.parse_args(argv)

    if args.command == "check-env":
        return check_env()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    warnings.filterwarnings("ignore", category=FutureWarning)
    settings = load_settings(args.config or [DEFAULT_CONFIG])
    run(settings, stages=tuple(args.stage or STAGES), force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
