# ScoreCardUtils

**A Python toolkit for building, evaluating, and monitoring credit risk scorecards.**

ScoreCardUtils provides end-to-end utilities for credit scorecard development — from SHAP-based feature selection and optimal binning to WoE transformation, bivariate analysis, model evaluation (KS, AUC, Gini), scorecard scaling, and population stability monitoring (CSI/VSI). Built for credit risk analysts, data scientists, and quantitative teams working in banking, lending, and financial services.

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/sandipkhaire/scorecardutils/actions/workflows/tests.yml/badge.svg)](https://github.com/sandipkhaire/scorecardutils/actions)

---

## Why ScoreCardUtils?

Building credit scorecards involves a repetitive pipeline — feature selection, binning, WoE analysis, model evaluation, and ongoing monitoring. Most teams end up writing the same boilerplate code across projects. ScoreCardUtils packages these common steps into a single, tested library so you can focus on modeling decisions rather than infrastructure.

- **SHAP-driven feature selection** with automatic class imbalance handling and correlation group resolution
- **Bivariate analysis** exported directly to styled Excel workbooks with embedded charts — ready for stakeholder review
- **Gains tables and KS statistics** with weighted sample support for production scoring pipelines
- **Population stability monitoring** (CSI/VSI) with parallel processing for fast OOT comparisons
- **Scorecard scaling** utilities to convert model probabilities into standard 3-digit credit scores

---

## Installation

```bash
# Using pip
pip install git+https://github.com/sandipkhaire/scorecardutils.git

# Using UV (recommended for faster installs)
uv pip install git+https://github.com/sandipkhaire/scorecardutils.git
```

Requires **Python 3.12** or higher.

---

## Quick Start

### 1. Feature Selection with SHAP

Select the most predictive features using SHAP importance values from an XGBoost model. Automatically handles class imbalance via `scale_pos_weight` and supports native categorical features.

```python
from scorecardutils import shap_feature_selection

selected_features, importance_df, _ = shap_feature_selection(
    train_data=train_df,
    feature_names=feature_list,
    target_name='default_flag',
    importance_threshold=0.95,
    verbose=True
)
```

### 2. Resolve Correlated Features

After SHAP selection, remove redundant correlated features by keeping the most important one from each correlation group.

```python
from scorecardutils import find_correlation_groups, select_best_features_from_corr_groups

corr_groups = find_correlation_groups(train_df[selected_features], corr_threshold=0.8)
result_df, final_features = select_best_features_from_corr_groups(
    corr_groups, importance_df,
    feature_name_col='feature_name',
    feature_importance_col='importance'
)
```

### 3. Bivariate Analysis with Excel Export

Generate professional bivariate analysis reports as styled Excel workbooks. Each variable gets its own sheet with a binning table and embedded chart. Supports train-only, OOT-only, and train-vs-OOT comparison modes.

```python
from scorecardutils import unified_bivariate_analysis

# Training data only
unified_bivariate_analysis(
    binning_process=binning_process,
    filename='bivariate_train',
    metric='event_rate',
    variables=final_features,
    show_bar_values=True,
    verbose=True
)

# Train vs OOT comparison
unified_bivariate_analysis(
    binning_process=binning_process,
    filename='bivariate_comparison',
    metric='woe',
    variables=final_features,
    oot_data=oot_df,
    target_column='default_flag',
    compare_data=True,
    show_bar_values=True
)
```

### 4. Model Evaluation — Gains Table and KS

Build gains/lift tables with KS statistics. Supports weighted samples and custom break points.

```python
from scorecardutils import gainTable, calculate_performance_metrics

# Gains table with KS
result = gainTable(
    data=test_df,
    score='predicted_score',
    ground_truth='default_flag',
    response_name=['Good', 'Bad'],
    numOfIntervals=10
)

gains_df = result['GainTable']
ks_value = result['KS']
breaks = result['breaks']

# KS, AUC, Gini — overall and by segment
metrics_df = calculate_performance_metrics(
    data=test_df,
    score_col='predicted_score',
    target_col='default_flag',
    segments='region'
)
```

### 5. Population Stability Monitoring (CSI/VSI)

Monitor feature distribution drift between training and out-of-time (OOT) data using Characteristic Stability Index. Uses parallel processing for fast computation across all variables.

```python
from scorecardutils import vsi_check

# Summary: one CSI value per variable
csi_summary = vsi_check(
    X_oot=oot_df,
    X_train=train_df,
    binning_process=binning_process,
    style='summary'
)

# Detailed: CSI breakdown by bin
csi_detailed = vsi_check(
    X_oot=oot_df,
    X_train=train_df,
    binning_process=binning_process,
    style='detailed'
)
```

### 6. Scorecard Scaling

Convert model probabilities into standard 3-digit credit scores using log-odds linear scaling.

```python
from scorecardutils import eqLinear, three_digit_score

# Calculate scaling parameters: score of 600 at 50:1 odds, 20 points to double the odds
params = eqLinear(OddsAtAnchor=50, Anchor=600, PDO=20)

# Convert probabilities to scores
test_df['credit_score'] = three_digit_score(
    test_df['predicted_probability'],
    alpha=params['alpha'],
    beta=params['beta']
)
```

---

## API Reference

### Feature Selection (`scorecardutils.feature_selection`)

| Function | Description |
|---|---|
| `shap_feature_selection(train_data, target_name, ...)` | Select features by cumulative SHAP importance from XGBoost |
| `find_correlation_groups(X, corr_threshold)` | Group correlated features using graph-based clustering |
| `select_best_features_from_corr_groups(groups, importance_df)` | Pick the highest-importance feature from each correlation group |
| `vsi_check(X_oot, X_train, binning_process, ...)` | Calculate Variable Stability Index (CSI) with parallel processing |

### Bivariate Analysis (`scorecardutils.BivariatePlot`)

| Function | Description |
|---|---|
| `unified_bivariate_analysis(binning_process, ...)` | Generate styled Excel workbook with binning tables and charts per variable |

### Evaluation Metrics (`scorecardutils.evaluation_metrics`)

| Function | Description |
|---|---|
| `gainTable(data, score, ground_truth, ...)` | Build gains/lift table with KS statistic |
| `calculate_breaks(data, score, ...)` | Compute weighted quantile-based score bin break points |
| `calculate_performance_metrics(data, score_col, target_col, ...)` | Calculate KS, ROC-AUC, and Gini — overall or by segment |
| `calculate_ks_statistic(data, score_col, target_col, ...)` | Standalone KS statistic calculation with optional weights |

### Utilities (`scorecardutils.utils`)

| Function | Description |
|---|---|
| `eqLinear(OddsAtAnchor, Anchor, PDO)` | Calculate alpha/beta parameters for scorecard scaling |
| `three_digit_score(prob_series, alpha, beta)` | Convert default probabilities to 3-digit credit scores |
| `save_object(file_path, obj)` | Serialize a Python object to disk (pickle) |
| `load_object(file_path)` | Load a previously saved Python object |
| `read_yaml_file(file_path)` | Parse a YAML configuration file |

---

## Example Notebooks

The `notebooks/` directory contains end-to-end worked examples:

| Notebook | Description |
|---|---|
| `0.data_creation_script.ipynb` | Generate synthetic credit risk datasets for experimentation |
| `1.EDA.ipynb` | Exploratory data analysis on credit risk data |
| `Modelling_script.ipynb` | Full scorecard development workflow — feature selection, binning, model training, evaluation, and monitoring |

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| numpy | >= 2.2.5 | Numerical computation |
| pandas | >= 2.2.3 | Data manipulation |
| scikit-learn | >= 1.6.1 | Model evaluation metrics |
| xgboost | >= 3.0.0 | Gradient boosting for SHAP feature selection |
| shap | >= 0.47.2 | SHAP value computation |
| optbinning | == 0.20.1 | Optimal binning and WoE transformation |
| feature-engine | >= 1.8.3 | Feature engineering utilities |
| openpyxl | >= 3.1.5 | Excel workbook generation |
| seaborn | >= 0.13.2 | Statistical visualization |
| pyyaml | >= 6.0.1 | YAML configuration parsing |

---

## Development Setup

```bash
git clone https://github.com/sandipkhaire/scorecardutils.git
cd scorecardutils
uv pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v --cov=scorecardutils --cov-report=term-missing

# Pre-commit hooks (notebook output stripping)
pre-commit install
```

---

## Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change, then submit a pull request.

## License

[MIT License](LICENSE)

## Author

**Sandip Khaire** — [GitHub](https://github.com/sandipkhaire)
