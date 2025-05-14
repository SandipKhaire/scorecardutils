import numpy as np
import pandas as pd
from typing import List, Tuple,Dict, Union
from scipy import stats
from sklearn.metrics import roc_auc_score

def gainTable(data, score, ground_truth, weights=None, 
              response_name=['Good','Bad'], numOfIntervals=10, 
              breaks=None, is_score_prob=False):
    """
    Generate a Gains Table with KS Statistics.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    score : str
        Column name with predicted probabilities or scores
    ground_truth : str
        Column name containing actual labels
    weights : str or None, optional
        Column name with sample weights (default: None)
    response_name : list, optional
        Names for positive(1) and negative(0) classes (default: ['Good','Bad'])
    numOfIntervals : int, optional
        Number of intervals to create (default: 10)
    breaks : array or None, optional
        Predefined break points (default: None)
    
    Returns:
    --------
    dict
        Dictionary containing gains table, break points, and KS statistic
    """
    # Sort the data in ascending order of predicted probabilities
    data = data.sort_values(by=score, ascending=True)
    
    # Determine minority class for lift calculation
    lift_index = determine_minority_class(data, ground_truth)
    
    # Handle weights
    if weights is None:
        data['wgt'] = 1
        weights = 'wgt'
    
    # Calculate breaks if not provided
    if breaks is None:
        breaks = calculate_breaks(data, score, weights, numOfIntervals)
    
    # Create score range column
    data['Score_Range'] = pd.cut(
        data[score], 
        bins=breaks, 
        include_lowest=True, 
        duplicates='drop'
    )
    
    # Prepare weighted data
    KS_data = prepare_weighted_data(
        data, score, ground_truth, weights, response_name
    )
    

    
    # Calculate cumulative metrics
    KS_data = calculate_cumulative_metrics(
        KS_data, response_name, is_score_prob, lift_index
    )
    
    correlation=check_score_correlation(data,score,ground_truth)

    KS_data= calculate_cumulative_metrics(
            KS_data, response_name, correlation, lift_index
        )

    # Calculate lift
    KS_data = calculate_lift(KS_data, response_name, lift_index)
    
    
    # Add total row
    KS_data = add_total_row(KS_data, data, score, weights, response_name)
    
    # Get KS statistic
    KS = KS_data['KS'].max()
    
    return {
        'GainTable': KS_data, 
        'breaks': breaks, 
        'KS': KS
    }


def determine_minority_class(data, ground_truth):
    """
    Determine the minority class in the ground truth column.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    ground_truth : str
        Column name containing actual labels
    
    Returns:
    --------
    int
        Index of the lift calculation (0 or 1)
    """
    class_counts = data[ground_truth].value_counts()
    minority_class = class_counts.idxmin()
    return 0 if minority_class == 1 else 1


def calculate_breaks(
    data: pd.DataFrame, 
    score: str, 
    weights: str = None, 
    numOfIntervals: int = 10
) -> np.ndarray:
    """
    Calculate breaks for score intervals using weighted or unweighted ECDF.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    score : str
        Column name with predicted probabilities or scores
    weights : str, optional
        Column name with sample weights
        If None, a weight column of 1s will be created
    numOfIntervals : int, optional
        Number of intervals to create (default: 10)
    
    Returns:
    --------
    numpy.ndarray
        Array of break points
    """
    # Create default weight column if not provided
    if weights is None:
        weights = 'wgt'
        data = data.copy()
        data[weights] = 1.0
    
    # Compute grouped weights
    result = data.groupby(score)[weights].sum().sort_index(ascending=True)
    result = pd.DataFrame(result).rename(columns={weights: 'TotalWeight'})
    result = result.reset_index()
    result['cum_weights'] = result['TotalWeight'].cumsum()
    
    # Calculate the weighted ECDF
    x_ = result[score].values
    y_ = result['cum_weights'] / result['TotalWeight'].sum()
    
    # Use interpolation to find the values corresponding to the quantiles
    quantiles = np.linspace(0, 1, num=numOfIntervals+1)
    breaks = np.unique(np.interp(quantiles, y_, x_))
    
    # Extend breaks to cover entire range
    breaks[0] = -np.inf
    breaks[-1] = np.inf
    
    # Round breaks if score values are large
    if data[score].max() > 1:
        breaks = np.round(breaks)
    
    return breaks

def prepare_weighted_data(data, score, ground_truth, weights, response_name):
    """
    Prepare data for gains table calculation with weighted aggregation.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    score : str
        Column name with predicted probabilities or scores
    ground_truth : str
        Column name containing actual labels
    weights : str
        Column name with sample weights
    response_name : list
        Names for positive(1) and negative(0) classes repectively
    
    Returns:
    --------
    pandas.DataFrame
        Prepared gains table data
    """
    # Weighted mean function
    wm = lambda x: np.average(x, weights=data.loc[x.index, weights])
    
    KS_data = data.assign(
        temp_good=np.where(data[ground_truth] == 1, data[weights], 0),
        temp_bad=np.where(data[ground_truth] == 0, data[weights], 0)
    ).groupby('Score_Range').agg(
        Totals=(weights, sum),
        Good=('temp_good', sum),
        Bad=('temp_bad', sum),
        AvgScore=(score, wm)
    ).sort_index(ascending=True)
    
    KS_data.columns = ['Totals', response_name[0], response_name[1], 'AvgScore']
    
    # Calculate rates
    KS_data[response_name[1]+'_Rate'] = (KS_data[response_name[1]] / KS_data['Totals']).apply(lambda x: round(100*x, 2))
    KS_data[response_name[0]+'_Rate'] = (KS_data[response_name[0]] / KS_data['Totals']).apply(lambda x: round(100*x, 2))
    
    return KS_data


def check_score_correlation(data, score, ground_truth):
    """
    Check the correlation between score and ground truth to determine sorting order.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    score : str
        Column name with predicted probabilities or scores
    ground_truth : str
        Column name containing actual labels
    
    Returns:
    --------
    bool
        True if ascending sort is needed, False otherwise
    
    Notes:
    ------
    - Uses Pearson correlation to determine relationship
    - Returns True if score is negatively correlated with ground truth
    - Handles potential cases of different correlation methods
    """
    # Calculate Pearson correlation
    try:
        # Method 1: Pandas correlation
        correlation = data[[score, ground_truth]].corr().iloc[0, 1]
    except Exception:
        try:
            # Method 2: SciPy correlation as fallback
            correlation, _ = stats.pearsonr(data[score], data[ground_truth])
        except Exception:
            # Fallback to a simple calculation if other methods fail
            correlation = np.corrcoef(data[score], data[ground_truth])[0, 1]
    
    # Determine sorting order based on correlation
    # Sort ascending if score is negatively correlated with ground truth
    return correlation 




def calculate_cumulative_metrics(KS_data, response_name, correlation, lift_index):
    """
    Calculate cumulative metrics for the gains table.
    
    Parameters:
    -----------
    KS_data : pandas.DataFrame
        Prepared gains table data
    response_name : list
        Names for positive and negative classes
    is_score_prob : bool
        Whether the score is a probability for class 1 if not please provide the prob for class 1
    lift_index : int
        Index for lift calculation
    
    Returns:
    --------
    pandas.DataFrame
        Gains table with cumulative metrics
    """
    # Sort index based on is_score_prob
    if lift_index==0 and correlation > 0 :
        KS_data = KS_data.sort_index(ascending=False)
    
    if lift_index==1 and correlation < 0 :
        KS_data = KS_data.sort_index(ascending=False)

    # Calculate Pop% for each bucket
    total_population = KS_data['Totals'].sum()
    KS_data['Pop%'] = np.round((KS_data['Totals'] / total_population) * 100, 2)
    
    # Calculate CumPop% - cumulative population percentage
    KS_data['CumPop%'] = np.round(
        (KS_data['Totals'] / total_population).cumsum() * 100, 2
    )
    
    # Cumulative calculations
    KS_data['Cum'+response_name[1]] = np.round(
        (KS_data[response_name[1]] / KS_data[response_name[1]].sum()).cumsum(), 4
    ) * 100
    
    KS_data['Cum'+response_name[0]] = np.round(
        (KS_data[response_name[0]] / KS_data[response_name[0]].sum()).cumsum(), 4
    ) * 100
    
    # KS Statistic
    KS_data['KS'] = np.round(
        abs(KS_data['Cum'+response_name[1]] - KS_data['Cum'+response_name[0]]), 4
    )

    return KS_data




def calculate_lift(KS_data, response_name, lift_index):
    """
    Calculate lift metric for the gains table.
    
    Parameters:
    -----------
    KS_data : pandas.DataFrame
        Gains table data
    response_name : list
        Names for positive and negative classes
    lift_index : int
        Index for lift calculation
    
    Returns:
    --------
    pandas.DataFrame
        Gains table with lift metric
    """
    KS_data['Lift'] = (
        KS_data['Cum'+response_name[lift_index]] / 
        ((KS_data['Totals'].cumsum() / KS_data['Totals'].sum()) * 100)
    ).apply(lambda x: round(x, 2))

    KS_data=KS_data[['Totals',response_name[1],response_name[1]+'_Rate','Cum'+response_name[1],
                            response_name[0],response_name[0]+'_Rate','Cum'+response_name[0],
                            'KS','AvgScore','Pop%', 'CumPop%','Lift']]
    KS_data.reset_index(inplace=True)
    
    return KS_data


def add_total_row(KS_data, data, score, weights, response_name):
    """
    Add a total row to the gains table.
    
    Parameters:
    -----------
    KS_data : pandas.DataFrame
        Gains table data
    data : pandas.DataFrame
        Original input dataframe
    score : str
        Column name with predicted probabilities or scores
    weights : str
        Column name with sample weights
    response_name : list
        Names for positive and negative classes
    
    Returns:
    --------
    pandas.DataFrame
        Gains table with total row
    """
    sum_row = {
        'Score_Range': 'Total',
        'Totals': KS_data['Totals'].sum(),
        response_name[1]: KS_data[response_name[1]].sum(),
        response_name[1]+'_Rate': np.round(
            (KS_data[response_name[1]].sum() / KS_data['Totals'].sum()) * 100, 2
        ),
        response_name[0]: KS_data[response_name[0]].sum(),
        response_name[0]+'_Rate': np.round(
            (KS_data[response_name[0]].sum() / KS_data['Totals'].sum()) * 100, 2
        ),
        'KS': KS_data['KS'].max(),
        'AvgScore': np.average(data[score], weights=data[weights]),
        'Pop%': 100.0 # Total represents 100% of population
    }
    
    KS_data.loc[len(KS_data)] = sum_row
    
    # Round numeric columns
    numeric_cols = KS_data.select_dtypes(include=[np.number]).columns
    KS_data[numeric_cols] = KS_data[numeric_cols].round(2)
    
    return KS_data



def calculate_ks_statistic(
    data: pd.DataFrame, 
    score_col: str, 
    target_col: str, 
    weights_col: str = None,
    breaks: np.ndarray = None,
    num_intervals: int = 10
) -> float:
    """
    Calculate KS statistic with flexible weight handling.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe
    score_col : str
        Column name with predicted probabilities
    target_col : str
        Column name with binary target variable
    weights_col : str, optional
        Column name with sample weights. 
        If None, a column of 1s will be created.
    breaks : np.ndarray, optional
        Predefined break points for binning. 
        If None, breaks will be calculated using weighted ECDF.
    num_intervals : int, optional
        Number of intervals for break calculation if breaks are not provided
    
    Returns:
    --------
    - KS statistic (float)
    """
    # Create default weights column if not provided
    if weights_col is None:
        # Create a new column of 1s with the same index as the original dataframe
        weights_col = 'wgt'
        data = data.copy()
        data[weights_col] = 1.0
    
    # Generate breaks if not provided
    if breaks is None:
        breaks = calculate_breaks(
            data, 
            score=score_col, 
            weights=weights_col, 
            numOfIntervals=num_intervals
        )
    
    # Validate breaks
    if len(breaks) < 2:
        raise ValueError("Breaks must contain at least two values")
    
    # Ensure breaks cover the entire range
    if breaks[0] != -np.inf:
        breaks[0] = -np.inf
    if breaks[-1] != np.inf:
        breaks[-1] = np.inf
    
    # Separate good and bad populations
    bad_data = data[data[target_col] == 1]
    good_data = data[data[target_col] == 0]
    
    # Calculate weighted histograms for bad and good populations
    bad_counts, _ = np.histogram(
        bad_data[score_col].values, 
        bins=breaks, 
        weights=bad_data[weights_col].values
    )
    
    good_counts, _ = np.histogram(
        good_data[score_col].values, 
        bins=breaks, 
        weights=good_data[weights_col].values
    )
    
    # Total weighted populations
    total_bad_weight = bad_data[weights_col].sum()
    total_good_weight = good_data[weights_col].sum()
    
    # Cumulative rates
    cum_bad_rate = np.cumsum(bad_counts) / total_bad_weight
    cum_good_rate = np.cumsum(good_counts) / total_good_weight
    
    # Pad cumulative rates
    cum_bad_rate = np.pad(cum_bad_rate, (1, 0), mode='constant')
    cum_good_rate = np.pad(cum_good_rate, (1, 0), mode='constant')
    # Calculate KS statistic
    ks_curve = np.abs(cum_bad_rate - cum_good_rate)
    ks_statistic = np.round(np.max(ks_curve)*100,2)
    
    return ks_statistic




def _calculate_segment_metrics(
    data: pd.DataFrame, 
    score_col: str, 
    target_col: str, 
    breaks: np.ndarray = None
) -> Dict[str, float]:
    """
    Internal function to calculate performance metrics for a single segment.

    Parameters:
    -----------
    data : pd.DataFrame
        Segment-specific dataframe
    score_col : str
        Column name with predicted probabilities
    target_col : str
        Column name with binary target variable
    breaks : np.ndarray, optional
        Predefined break points for KS statistic calculation

    Returns:
    --------
    Dict[str, float]
        Dictionary of performance metrics
    """

    
    # Calculate KS Statistic
    KS=calculate_ks_statistic(data=data,score_col=score_col,target_col=target_col,breaks=breaks)
    
    # Calculate ROC AUC and GINI
    AUC_ROC = roc_auc_score(
        y_true=data[target_col].values,
        y_score=data[score_col].values
    )
    
    GINI = 2 * AUC_ROC - 1 
    
    # Round metrics
    return {
        'KS': np.round(KS, 2),
        'ROC_AUC': np.round(AUC_ROC * 100, 2),
        'GINI': np.round(GINI * 100, 2)
    }


def calculate_performance_metrics(
    data: pd.DataFrame, 
    score_col: str, 
    target_col: str, 
    breaks: np.ndarray = None,
    segments: Union[str, List[str]] = None
) -> pd.DataFrame:
    """
    Calculate comprehensive performance metrics for binary classification models.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe containing model predictions and true labels
    score_col : str
        Column name with predicted probabilities
    target_col : str
        Column name with binary target variable
    breaks : np.ndarray, optional
        Predefined break points for KS statistic calculation
    segments : str or list of str, optional
        Segment column name or list of segment names to evaluate
        If None, uses entire dataset

    Returns:
    --------
    pd.DataFrame
        Performance metrics for specified segments
    """
    # Prepare performance results
    perf_results = {}

    # Handle segments
    if segments is None:
        # Use entire dataset if no segments specified
        perf_results['overall'] = _calculate_segment_metrics(
            data=data, 
            score_col=score_col, 
            target_col=target_col, 
            breaks=breaks
        )
    elif isinstance(segments, str):
        # If segments is a column name, iterate through unique values
        for seg in data[segments].unique():
            segment_data = data[data[segments] == seg]
            perf_results[seg] = _calculate_segment_metrics(
                data=segment_data, 
                score_col=score_col, 
                target_col=target_col, 
                breaks=breaks
            )
    elif isinstance(segments, list):
        # If segments is a list of names, filter for each
        for seg in segments:
            segment_data = data[data['dataset'] == seg]
            perf_results[seg] = _calculate_segment_metrics(
                data=segment_data, 
                score_col=score_col, 
                target_col=target_col, 
                breaks=breaks
            )
    else:
        raise ValueError("Segments must be None, a string, or a list of strings")

    # Convert to transposed DataFrame
    return pd.DataFrame(perf_results).T

