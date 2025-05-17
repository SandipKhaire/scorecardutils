"""
ScoreCardUtils - A comprehensive Python package for credit scorecard development
"""

__version__ = "1.0.1"

from .feature_selection import (
    shap_feature_selection,
    find_correlation_groups,
    select_best_features_from_corr_groups,
    vsi_check
)
from .BivariatePlot import unified_bivariate_analysis
from .evaluation_metrics import (
    calculate_breaks,
    calculate_performance_metrics,
    gainTable
)
from .utils import (
    save_object,
    load_object,
    eqLinear,
    three_digit_score
)

__all__ = [
    'shap_feature_selection',
    'find_correlation_groups',
    'select_best_features_from_corr_groups',
    'vsi_check',
    'unified_bivariate_analysis',
    'calculate_breaks',
    'calculate_performance_metrics',
    'gainTable',
    'save_object',
    'load_object',
    'eqLinear',
    'three_digit_score',
]
