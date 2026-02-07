# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ScoreCardUtils is a Python package for credit scorecard development and risk modeling. It provides utilities for SHAP-based feature selection, variable binning, WoE transformation, bivariate analysis with Excel export, model evaluation metrics (KS, AUC, Gini), and scorecard scaling.

## Build & Development Commands

```bash
# Install in development mode (UV recommended)
uv pip install -e ".[dev]"

# Run all tests with coverage
python -m pytest tests/ -v --cov=scorecardutils --cov-report=term-missing

# Run a single test file
python -m pytest tests/unit/test_evaluation_metrics.py -v

# Run a specific test
python -m pytest tests/unit/test_evaluation_metrics.py::test_gainTable -v

# Pre-commit hooks (nbstripout for notebook cleaning)
pre-commit run --all-files
```

## Architecture

The package lives in `src/scorecardutils/` with four modules, all re-exported from `__init__.py`:

- **`feature_selection.py`** — SHAP-based feature selection (`shap_feature_selection`), correlation group detection (`find_correlation_groups`, `select_best_features_from_corr_groups`), and Variable Stability Index calculation (`vsi_check`) using parallel `ThreadPoolExecutor`. Trains XGBoost internally with automatic class imbalance handling via `scale_pos_weight`.

- **`BivariatePlot.py`** — Bivariate analysis and visualization (`unified_bivariate_analysis`). Operates in three modes: train-only, OOT-only, and OOT-comparison. Generates styled Excel workbooks with embedded matplotlib PNG charts. Calculates Event Rate, WoE (Jeffrey divergence), and IV metrics per variable. Creates and cleans up a `temp_plots/` directory for intermediate PNGs.

- **`evaluation_metrics.py`** — Model performance evaluation. `gainTable()` produces gains/lift tables with KS statistic. `calculate_breaks()` computes weighted quantile-based score bins. `calculate_performance_metrics()` returns KS, ROC-AUC, and Gini by segment. Handles weighted samples and imbalanced data.

- **`utils.py`** — Scorecard scaling (`eqLinear` for alpha/beta parameters, `three_digit_score` for probability-to-score conversion), pickle serialization (`save_object`/`load_object`), and YAML config reading.

## Code Style

- Always create a git branch before making changes.
- Use comments sparingly — only comment complex or non-obvious code.

## Key Details

- **Python >= 3.12** required; build system is `hatchling`
- **optbinning is pinned to 0.20.1** — this is intentional, do not upgrade
- Tests use pytest with fixtures in `tests/conftest.py` providing `sample_data` and `feature_names`
- CI runs on GitHub Actions (`.github/workflows/tests.yml`): Ubuntu, Python 3.12, UV-based install
- Notebooks in `notebooks/` demonstrate the full scorecard workflow (data creation, EDA, modeling)
- Sample datasets in `data/` are large CSV files used by notebooks
