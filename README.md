# flight-delay-prediction-brazil

Predicting 30+ minute departure delays in Brazilian commercial aviation using ANAC flight data (VRA, 2024–2025)
and daily weather (GSOD/NOAA). Compares Logistic Regression, Decision Tree, Random Forest, and XGBoost on ~480K
flights; Random Forest performs best (ROC-AUC 0.71, PR-AUC 0.255). Includes paper and code.

The code compares up to eight classifiers (baseline, logistic regression, decision tree, random forest, extra
trees, histogram gradient boosting, XGBoost, LightGBM) with grouped cross-validation, thresholds chosen on
training data only, paired-bootstrap confidence intervals, permutation importance, an ablation by feature group,
partial dependence, per-segment performance and LaTeX tables for the paper.

## Architecture

The project separates **pipelines** (feature → training → evaluation), **data layers** and **infrastructure**,
following the literature on ML systems engineering:

| Idea | Source | Where in the code |
|---|---|---|
| Avoid glue code and "pipeline jungles" | Sculley et al., *Hidden Technical Debt in ML Systems*, NeurIPS 2015 | one module per responsibility |
| Independent feature / training / inference pipelines | Dowling, FTI pipelines (Hopsworks) | `pipelines/` |
| Layered data (raw → intermediate → primary → feature → model input) | Kedro data-engineering convention; medallion architecture | `adapters/storage.py`, `data/` |
| Project layout | Cookiecutter Data Science v2 | repository tree |
| Ports and adapters (hexagonal architecture) | Cockburn; Ferreira et al. 2025 (MLOps) | `ports.py`, `adapters/` |
| Data contracts at layer boundaries | Pandera | `domain/schemas.py` |
| Tests of data, features and models | Breck et al., *The ML Test Score*, IEEE BigData 2017 | `tests/` |
| Reproducible pipeline (MLOps level 1) | Google Cloud, *MLOps: continuous delivery and automation pipelines* | `configs/`, `cli.py` |

```
src/flight_delay/
├── config.py          typed, immutable settings loaded from YAML (no globals)
├── ports.py           interfaces the pipelines depend on (FlightDataSource, WeatherSource)
├── adapters/          outside world: ANAC HTTP, BigQuery (GSOD), layered storage, Colab
├── domain/            pure rules: column contract, VRA cleaning, weather, sampling, schemas
├── features/          every model input defined in one place + anti-leakage contract
├── modeling/          split, model catalogue, tuning, cross-validation
├── evaluation/        metrics, thresholds, bootstrap, importance, ablation, segments
├── reporting/         figures, LaTeX tables, Portuguese labels for the paper
├── pipelines/         etl → features → modeling, and the composition root (run.py)
└── cli.py             `flight-delay` command
```

The core (`domain`, `features`, `modeling`, `evaluation`) never imports Colab, Google Drive or BigQuery, so it
runs and is tested locally without network or cloud credentials.

### Data layers

| Layer | Content |
|---|---|
| `01_raw` | ANAC CSV as downloaded (deleted after conversion unless `source.keep_raw: true`) |
| `02_intermediate` | typed mirror of the CSV: internal column names, parsed timestamps, all rows |
| `03_primary` | realised flights of the month with the departure delay; daily weather per airport |
| `04_feature` | label and pre-departure features for **every** flight of the month |
| `05_model_input` | the per-month sample used for modelling |
| `07_model_output` | test-set predictions of every model |
| `08_reporting` | tables (CSV and LaTeX), figures and `summary.json`, one folder per run |

Every stage skips outputs that already exist, so an interrupted run resumes where it stopped (`--force`
recomputes).

## Running locally

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/). On Windows (PowerShell):

   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. Install the project with every extra (boosting libraries and Google Cloud client):

   ```bash
   uv sync --all-extras
   uv run flight-delay check-env
   ```

   `uv sync` creates `uv.lock`; commit it so every run uses the same library versions.

3. Configure Google Cloud once (weather comes from the BigQuery public dataset `noaa_gsod`):

   1. Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install).
   2. `gcloud auth application-default login` (opens the browser; creates Application Default Credentials).
   3. `gcloud config set project <your-project-id>`, or set `weather.gcp_project` in a config file.
   4. Make sure the BigQuery API is enabled in that project. Queries on public datasets are billed to it;
      check BigQuery pricing and the free tier for your account.

4. Run:

   ```bash
   # quick validation: 2 months, small sample, fewer tuning/bootstrap iterations
   uv run flight-delay run -c configs/base.yaml -c configs/test.yaml

   # full experiment of the paper
   uv run flight-delay run

   # only some stages (e.g. re-model without touching the data)
   uv run flight-delay run --stage modeling
   ```

Outputs go to `data/` (git-ignored).

## Running on Google Colab

Open `notebooks/colab_runner.ipynb` in Colab. It clones this repository, installs the package, mounts Google
Drive, authenticates BigQuery and runs the same pipeline with `configs/colab.yaml`, which stores the data layers
in `/content/drive/MyDrive/Dados_ANAC`.

## Configuration

`configs/base.yaml` is the full experiment. Other files are deep-merged on top of it and only list what they
change (`test.yaml`, `colab.yaml`). Unknown keys are rejected, so a typo cannot silently run the wrong
experiment. Changing `scope.delay_threshold_min`, `scope.max_flights_per_month` or `scope.random_state` writes new
feature/sample files instead of reusing stale ones.

## Tests

```bash
uv run pytest
uv run ruff check .
```

The tests build synthetic ANAC files and fake weather, so the whole pipeline (ETL → features → modelling →
report) runs offline in a few seconds. They cover date parsing (ISO vs. Brazilian formats), the column contract,
the anti-leakage guard, traffic computed before sampling, the weather join, grouped splits, thresholds, bootstrap,
LaTeX formatting and resuming an interrupted run.

## Methodological notes

- **Target:** departure delay ≥ `scope.delay_threshold_min` minutes (30 by default). Flights without an actual
  departure (e.g. cancelled) are removed.
- **Weather:** GSOD values are **daily means of the same day** (origin: departure day; destination: arrival day).
  They include hours after the departure, so the model assumes a perfect weather forecast for the day.
- **Traffic:** `origin_hourly_flow` counts scheduled departures per airport, day and hour on the *full* month,
  before sampling.
- **Split:** by default no day appears in both train and test (`modeling.split_strategy: by_day`); the best model
  is chosen by cross-validated PR-AUC and the decision thresholds on out-of-fold training predictions, so the
  test set plays no part in either choice.
- **Class weights** are on by default, so predicted probabilities are not calibrated (see `calibracao.png`).
