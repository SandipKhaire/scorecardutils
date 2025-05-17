# ScoreCardUtils

A comprehensive Python package for credit scorecard development and risk modeling. This package provides utilities for feature selection, binning, WoE transformation, and model evaluation specifically designed for credit risk assessment.

## Features

- Feature selection using SHAP values
- Variable binning with Information Value (IV) calculation
- Weight of Evidence (WoE) transformation
- Characteristic Stability Index (CSI) analysis
- Bivariate analysis with customizable plotting
- Model performance evaluation metrics
- Scorecard scaling utilities

## Installation

### Using pip

```bash
pip install git+https://github.com/sandipkhaire/scorecardutils.git
```

### Using UV (recommended)

```bash
uv pip install git+https://github.com/sandipkhaire/scorecardutils.git
```

## Quick Start

```python
import pandas as pd
from scorecardutils.feature_selection import shap_feature_selection
from scorecardutils.BivariatePlot import unified_bivariate_analysis

# Load your data
data = pd.read_csv('your_data.csv')

# Feature selection using SHAP
selected_features, importance_df, _ = shap_feature_selection(
    train_data=data,
    feature_names=features,
    target_name='target',
    verbose=True
)

# Bivariate analysis
unified_bivariate_analysis(
    binning_process=binning_process,
    filename='event_rate_analysis',
    metric='woe',
    variables=selected_features,
    show_bar_values=True
)
```

## Requirements

- Python >= 3.12
- numpy >= 2.2.5
- pandas >= 2.2.3
- scikit-learn >= 1.6.1
- xgboost >= 3.0.0
- optbinning == 0.20.1
- shap >= 0.47.2
- feature-engine >= 1.8.3
- seaborn >= 0.13.2
- openpyxl >= 3.1.5

## Development

To set up the development environment:

```bash
# Clone the repository
git clone https://github.com/sandipkhaire/scorecardutils.git
cd scorecardutils

# Install development dependencies using UV
uv pip install -e ".[dev]"
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Authors

- **Sandip Khaire** - *Initial work* - [sandipkhaire](https://github.com/sandipkhaire)