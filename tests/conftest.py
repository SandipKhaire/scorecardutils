"""
Pytest configuration file for scorecardutils tests.
"""

import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_data():
    """Create a sample dataset for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.lognormal(10, 0.5, n_samples),
        'credit_score': np.random.normal(700, 50, n_samples),
        'default': np.random.binomial(1, 0.2, n_samples)
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def feature_names():
    """List of feature names for testing."""
    return ['age', 'income', 'credit_score']
