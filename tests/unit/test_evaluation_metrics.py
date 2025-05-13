import pytest
import numpy as np
import pandas as pd
import sys
sys.path.append('.')  # Add current directory to path

# Import the functions to be tested
from scorecardutils.evaluation_metrics import (
    gainTable, 
    determine_minority_class,
    calculate_breaks,
    prepare_weighted_data,
    check_score_correlation,
    calculate_cumulative_metrics,
    calculate_ks_statistic,
    calculate_performance_metrics
)

@pytest.fixture
def sample_dataframe():
    """
    Create a sample dataframe for testing credit risk utility functions
    """
    np.random.seed(42)
    data = pd.DataFrame({
        'score': np.random.uniform(0, 1, 1000),
        'ground_truth': np.random.choice([0, 1], 1000, p=[0.7, 0.3]),
        'weights': np.ones(1000)
    })
    return data

def test_determine_minority_class(sample_dataframe):
    """
    Test determine_minority_class function
    """
    # Scenario 1: Imbalanced dataset with fewer 1s
    minority_index = determine_minority_class(sample_dataframe, 'ground_truth')
    assert minority_index == 0  # Expecting positive class (1) to be minority

    # Edge case: Perfectly balanced dataset
    balanced_df = pd.DataFrame({
        'ground_truth': [0, 0, 1, 1,1],
        'score': [0.1, 0.2, 0.3, 0.4,0.5]
    })
    minority_index = determine_minority_class(balanced_df, 'ground_truth')
    assert minority_index == 1  

def test_calculate_breaks(sample_dataframe):
    """
    Test calculate_breaks function with various scenarios
    """
    # Scenario 1: Default number of intervals
    breaks = calculate_breaks(sample_dataframe, 'score')
    assert len(breaks) == 11  # 10 intervals + start/end infinite bounds
    assert breaks[0] == -np.inf
    assert breaks[-1] == np.inf

    # Scenario 2: Custom number of intervals
    custom_breaks = calculate_breaks(sample_dataframe, 'score', numOfIntervals=5)
    assert len(custom_breaks) == 6  # 5 intervals + start/end infinite bounds

    # Scenario 3: With predefined weights
    breaks_with_weights = calculate_breaks(sample_dataframe, 'score', weights='weights')
    assert len(breaks_with_weights) == 11

def test_check_score_correlation(sample_dataframe):
    """
    Test check_score_correlation function
    """
    # Scenarios: Positive and negative correlations
    correlations = [
        # Perfectly correlated
        pd.DataFrame({
            'score': [0.1, 0.2, 0.3, 0.4],
            'ground_truth': [0, 0, 1, 1]
        }),
        # Negatively correlated
        pd.DataFrame({
            'score': [0.9, 0.8, 0.2, 0.1],
            'ground_truth': [0, 0, 1, 1]
        })
    ]

    
    correlation = check_score_correlation(correlations[0], 'score', 'ground_truth')
    assert isinstance(correlation, float)
    assert correlation>0 # Positive correlation

    correlation = check_score_correlation(correlations[1], 'score', 'ground_truth')
    assert isinstance(correlation, float)
    assert correlation<0 # Negative correlation



def test_calculate_ks_statistic(sample_dataframe):
    """
    Test KS statistic calculation with various scenarios
    """
    # Scenario 1: Basic KS statistic calculation
    ks_result = calculate_ks_statistic(
        sample_dataframe, 
        score_col='score', 
        target_col='ground_truth'
    )
    assert 0 <= ks_result <= 100  # KS should be between 0-100

    # Scenario 2: With explicit weights
    ks_with_weights = calculate_ks_statistic(
        sample_dataframe, 
        score_col='score', 
        target_col='ground_truth', 
        weights_col='weights'
    )
    assert 0 <= ks_with_weights <= 100

    # Edge Case: Perfectly separated classes
    perfect_sep_df = pd.DataFrame({
        'score': [0.1, 0.2, 0.3, 0.4] * 25,
        'ground_truth': [0, 0, 1, 1] * 25,
        'weights': [1] * 100
    })
    perfect_ks = calculate_ks_statistic(
        perfect_sep_df, 
        score_col='score', 
        target_col='ground_truth'
    )
    assert perfect_ks == 100  # Perfect separation

def test_gainTable(sample_dataframe):
    """
    Test gainTable function with multiple scenarios
    """
    # Scenario 1: Basic gain table
    result = gainTable(
        sample_dataframe, 
        score='score', 
        ground_truth='ground_truth'
    )
    assert 'GainTable' in result
    assert 'breaks' in result
    assert 'KS' in result

    # Scenario 2: With custom response names
    custom_result = gainTable(
        sample_dataframe, 
        score='score', 
        ground_truth='ground_truth',
        response_name=['Pass', 'Fail']
    )
    assert 'Pass' in custom_result['GainTable'].columns
    assert 'Fail' in custom_result['GainTable'].columns

    # Scenario 3: With weights
    weighted_result = gainTable(
        sample_dataframe, 
        score='score', 
        ground_truth='ground_truth',
        weights='weights'
    )
    assert 'GainTable' in weighted_result


    

    
