import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import defaultdict
from typing import Dict, List, Set
from concurrent.futures import ThreadPoolExecutor


def shap_feature_selection(
    train_data: pd.DataFrame,
    target_name: str,
    feature_names: list = None,
    split_data: bool = True,
    test_size: float = 0.3,
    importance_threshold: float = 0.95,
    random_state: int = 42,
    model_params: dict = None,
    use_train_for_shap: bool = True,
    enable_categorical: bool = True,
    create_shap_df: bool = False,
    plot_shap: bool = False,
    verbose: bool = False
):
    """
    Performs feature selection using SHAP values and XGBoost.
    
    Parameters:
    -----------
    train_data : pd.DataFrame
        DataFrame containing features and target variable
    target_name : str
        Name of the target column (required)
    feature_names : list, optional
        List of feature column names. If None, all columns except target will be used
    split_data : bool, default=True
        Whether to split data into train/test sets
    test_size : float, default=0.3
        Proportion of data to use for testing when split_data=True
    importance_threshold : float, default=0.95
        Cumulative importance threshold for feature selection
    random_state : int, default=42
        Random seed for reproducibility
    model_params : dict, optional
        Parameters for XGBoost model. If None, default parameters will be used
    use_train_for_shap : bool, default=True
        Whether to calculate SHAP values on training data (True) or test data (False)
    enable_categorical : bool, default=True
        Whether to enable XGBoost's native categorical feature support
    create_shap_df : bool, default=False
        Whether to create and return a DataFrame with SHAP values
    plot_shap : bool, default=True
        Whether to plot SHAP summary plot
    verbose : bool, default=False
        Whether to print progress information
    
    Returns:
    --------
    tuple
        (selected_features, shap_importance_df, shap_values, shap_df)
        - selected_features: List of selected feature names
        - shap_importance_df: DataFrame with feature importance information
        - shap_values: Raw SHAP values
        - shap_df: DataFrame with SHAP values for each sample and feature (if create_shap_df=True, else None)
    """
    # Validate that target_name exists in the dataframe
    if target_name not in train_data.columns:
        raise ValueError(f"Target column '{target_name}' not found in the provided DataFrame")
    
    # Handle feature names
    if feature_names is None:
        feature_names = [col for col in train_data.columns if col != target_name]
    
    # Extract features and target
    X = train_data[feature_names].copy()
    y = train_data[target_name].copy()
    
    # Default model parameters
    default_params = {
        'n_estimators': 100,
        'learning_rate': 0.05,
        'max_depth': 4,
        'random_state': random_state,
        'eval_metric': 'auc',
        'early_stopping_rounds': 20,
        'enable_categorical': enable_categorical
    }
    
    # Update with user-provided parameters if available
    if model_params is not None:
        default_params.update(model_params)

    print(f"Using model parameters: {default_params}")
    
    # Calculate class balance for imbalanced datasets
    if len(y.unique()) == 2 and 'scale_pos_weight' not in default_params:
        scale_pos_weight = sum(y == 0) / max(sum(y == 1), 1)  # Avoid division by zero
        default_params['scale_pos_weight'] = scale_pos_weight
        if verbose:
            print(f"scale_pos_weight: {scale_pos_weight:.4f}")
    
    # Split data if requested
    if split_data:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if len(y.unique()) < 10 else None
        )
        eval_set = [(X_test, y_test)]
    else:
        X_train, y_train = X, y
        eval_set = None
        # Create a copy of params without early stopping
        if 'early_stopping_rounds' in default_params:
            del default_params['early_stopping_rounds']


    
    # Initialize and train XGBoost model
    model = xgb.XGBClassifier(**default_params)
    
    # If enable_categorical is True, identify categorical columns
    if default_params.get('enable_categorical', False):
        categorical_columns = X_train.select_dtypes(include=['category', 'object']).columns.tolist()
        if categorical_columns and verbose:
            print(f"Detected {len(categorical_columns)} categorical features: {categorical_columns}")
        
        # Ensure categorical columns are properly typed
        for col in categorical_columns:
            X_train[col] = X_train[col].astype('category')
            if split_data:
                X_test[col] = X_test[col].astype('category')
    
    if eval_set is not None:
        model.fit(X_train, y_train, eval_set=eval_set, verbose=verbose)
    else:
        model.fit(X_train, y_train, verbose=verbose)
    
    # Use TreeExplainer for faster computation with tree-based models
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values on either training or test data
    if use_train_for_shap:
        if verbose:
            print("Calculating SHAP values on training data...")
        data_for_shap = X_train
        shap_values = explainer.shap_values(X_train)
    else:
        if split_data:
            if verbose:
                print("Calculating SHAP values on test data...")
            data_for_shap = X_test
            shap_values = explainer.shap_values(X_test)
        else:
            if verbose:
                print("Warning: use_train_for_shap=False but split_data=False, using training data for SHAP values")
            data_for_shap = X_train
            shap_values = explainer.shap_values(X_train)
    
    # For multi-class, take the mean absolute value across all classes
    if isinstance(shap_values, list):
        # For multi-class models, shap_values is a list of arrays, one per class
        mean_abs_shap_values = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        # For binary classification, shap_values is a single array
        mean_abs_shap_values = np.abs(shap_values).mean(axis=0)
    
    # Create feature importance DataFrame
    shap_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'SHAP_Importance': mean_abs_shap_values
    })
    
    # Sort features by importance
    shap_importance_df = shap_importance_df.sort_values(by='SHAP_Importance', ascending=False)
    
    # Calculate relative and cumulative importance
    total_importance = shap_importance_df['SHAP_Importance'].sum()
    shap_importance_df['Relative_Importance'] = shap_importance_df['SHAP_Importance'] / total_importance
    shap_importance_df['Cumulative_Importance'] = shap_importance_df['Relative_Importance'].cumsum()
    
    # Select features based on cumulative importance threshold
    selected_features = shap_importance_df[
        shap_importance_df['Cumulative_Importance'] <= importance_threshold
    ]['Feature'].tolist()
    
    # Add the next feature to cross the threshold to ensure we meet or exceed the threshold
    if len(selected_features) < len(feature_names) and importance_threshold < 1.0:
        next_feature = shap_importance_df.iloc[len(selected_features)]['Feature']
        selected_features.append(next_feature)
    
    if verbose:
        print(f"Selected {len(selected_features)} features out of {len(feature_names)} "
              f"that explain at least {importance_threshold*100:.1f}% of model predictions")
    
    # Create SHAP DataFrame if requested
    shap_df = None
    if create_shap_df:
        if isinstance(shap_values, list):
            # For multi-class problems, create a separate DataFrame for each class
            shap_dfs = []
            for class_idx, class_shap_values in enumerate(shap_values):
                class_shap_df = pd.DataFrame(
                    class_shap_values, 
                    columns=[f"{col}_shap" for col in feature_names]
                )
                class_shap_df['class'] = class_idx
                shap_dfs.append(class_shap_df)
            
            # Combine all class DataFrames
            shap_df = pd.concat(shap_dfs, axis=0)
        else:
            # For binary classification
            shap_df = pd.DataFrame(
                shap_values,
                columns=[f"{col}" for col in feature_names]
            )
    
    # Plot SHAP summary plot if requested
    if plot_shap:
        plt.figure(figsize=(10, 8))
        if isinstance(shap_values, list):
            # For multi-class, show the mean absolute SHAP values
            mean_abs_shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            shap.summary_plot(mean_abs_shap_values, data_for_shap, plot_type="bar")
        else:
            shap.summary_plot(shap_values, data_for_shap)
        plt.tight_layout()
        plt.show()
    
    return selected_features, shap_importance_df,shap_df





def find_correlation_groups(
    X: pd.DataFrame,
    corr_threshold: float = 0.8
) -> Dict[int, List[str]]:
    """
    Find groups of correlated features based on a correlation threshold.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame containing only numeric feature columns.
    corr_threshold : float, optional (default=0.8)
        Absolute correlation threshold above which features are considered correlated.

    Returns
    -------
    Dict[int, List[str]]
        Dictionary where each key is a group ID and each value is a list of correlated feature names.
        Features not correlated with any other above the threshold are returned in individual groups.
    """
    corrmat = X.corr().abs()
    corr_pairs = corrmat.unstack()
    filtered_pairs = corr_pairs[
        (corr_pairs > corr_threshold) & (corr_pairs < 1)
    ].reset_index()
    filtered_pairs.columns = ['feature1', 'feature2', 'corr']

    correlation_groups: Dict[int, Set[str]] = defaultdict(set)
    features_assigned: Set[str] = set()

    # First pass: create groups based on correlated pairs
    for _, row in filtered_pairs.iterrows():
        f1, f2 = row['feature1'], row['feature2']
        group_found = False
        for group in correlation_groups.values():
            if f1 in group or f2 in group:
                group.update([f1, f2])
                group_found = True
                break
        if not group_found:
            group_id = len(correlation_groups)
            correlation_groups[group_id] = {f1, f2}
        features_assigned.update([f1, f2])

    # Second pass: merge overlapping groups
    merged = True
    while merged:
        merged = False
        keys = list(correlation_groups.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                g1, g2 = keys[i], keys[j]
                if g1 in correlation_groups and g2 in correlation_groups:
                    if correlation_groups[g1] & correlation_groups[g2]:
                        correlation_groups[g1].update(correlation_groups[g2])
                        del correlation_groups[g2]
                        merged = True
                        break
            if merged:
                break

    # Add non-correlated features
    ungrouped_features = set(X.columns) - features_assigned
    for feature in ungrouped_features:
        correlation_groups[len(correlation_groups)] = {feature}

    return {k: sorted(list(v)) for k, v in correlation_groups.items()}


def select_best_features_from_corr_groups(correlated_groups, feature_importance_df,
                                     feature_name_col='feature_name',
                                     feature_importance_col='importance'):
    """
    Select the best feature from each correlation group based on feature importance
    
    Parameters:
    -----------
    correlated_groups : dict
        Dictionary where keys are group IDs and values are lists of feature names
    feature_importance_df : pandas DataFrame
        DataFrame with at least two columns for feature names and importance values
    feature_name_col : str
        Name of the feature column to be used for importance comparison
    feature_importance_col : str
        Name of the importance column to be used for comparison
        
    Returns:
    --------
    result_df : pandas DataFrame
        DataFrame with columns: 'feature', 'group', 'importance', 'keep'
    selected_features : list
        List of features to keep
    """
    # Create a dictionary for quick lookup of feature importance
    importance_dict = pd.Series(
        feature_importance_df[feature_importance_col].values,
        index=feature_importance_df[feature_name_col]
    ).to_dict()
    
    # Create a list to hold rows for the result dataframe
    result_rows = []
    
    # Process each correlation group
    for group_id, features in correlated_groups.items():
        # Get importance for each feature in the group
        group_features_data = [(f, importance_dict.get(f, float('nan'))) for f in features]
        
        # For groups with multiple features, find the one with highest importance
        if len(features) > 1:
            # Find feature with max importance
            best_feature, _ = max(group_features_data, key=lambda x: x[1])
        else:
            # If only one feature, keep it
            best_feature = features[0]
        
        # Add all features from this group to results
        for feature, importance in group_features_data:
            result_rows.append({
                'feature': feature,
                'group': group_id,
                'importance': importance,
                'keep': feature == best_feature
            })
    
    # Create result dataframe from rows
    result_df = pd.DataFrame(result_rows)
    
    # Sort by group and importance (descending)
    if not result_df.empty:
        result_df = result_df.sort_values(['group', 'importance'], ascending=[True, False])
    
    # Get list of features to keep
    selected_features = result_df.loc[result_df['keep'], 'feature'].tolist()
    
    return result_df, selected_features



def jeffrey(p_1, p_2, return_sum=True):
    """Calculate Jeffrey divergence between two distributions, multiplied by 100 and rounded to 2 decimal places."""
    # Handle zeros to avoid division by zero
    p_1_safe = np.maximum(p_1, 1e-10)
    p_2_safe = np.maximum(p_2, 1e-10)
    
    # Calculate Jeffrey divergence and multiply by 100
    divergence = ((p_1_safe - p_2_safe) * np.log(p_1_safe / p_2_safe)) * 100
    
    # Round to 2 decimal places
    divergence = np.round(divergence, 2)
    
    # Return sum if requested, otherwise return the array
    return np.sum(divergence) if return_sum else divergence


def process_variable(name, X_oot, X_train, binning_process, psi_min_bin_size=0.01):
    """Process a single variable to calculate VSI metrics."""
    # Get the binned variable object
    optb = binning_process.get_binned_variable(name)
    sc_table = optb.binning_table.build()
    
    # Transform data to bins
    ta = optb.transform(X_oot[name], metric="bins")
    te = optb.transform(X_train[name], metric="bins")
    
    # Get unique bins
    unique_bins = sc_table["Bin"].iloc[:-1].values
    n_bins = len(unique_bins)
    
    # Vectorized counts
    bin_counts_a = np.array([np.sum(ta == str(bin_val)) for bin_val in unique_bins[:n_bins]])
    bin_counts_e = np.array([np.sum(te == str(bin_val)) for bin_val in unique_bins[:n_bins]])
    
    # Calculate proportions
    t_records_a = bin_counts_a.sum()
    t_records_e = bin_counts_e.sum()
    prop_a = np.round(bin_counts_a / t_records_a,2)
    prop_e = np.round(bin_counts_e / t_records_e,2)
    
    # Calculate PSI
    psi = jeffrey(prop_a, prop_e, return_sum=False)
    
    # Create result dataframe
    df_psi = pd.DataFrame({
        "Variable": [name] * n_bins,
        "Bin": unique_bins[:n_bins],
        "Count OOT": bin_counts_a,
        "Count Train": bin_counts_e,
        "Count OOT (%)": prop_a,
        "Count Train (%)": prop_e,
        "CSI": psi
    })
    
    return df_psi


def vsi_check(X_oot, X_train, binning_process, style='summary', psi_min_bin_size=0.01, max_workers=None):
    """
    Calculate Variable Stability Index (VSI) by comparing OOT and training data distributions.
    
    Parameters:
    -----------
    X_oot : pandas.DataFrame
        Out-of-time data for comparison
    X_train : pandas.DataFrame
        Training data as baseline
    binning_process : object
        Binning process object with required methods for transformation
    style : str, default 'summary'
        Output style - 'summary' for variable-level CSI sums or 'detailed' for bin-level details
    psi_min_bin_size : float, default 0.01
        Minimum bin size threshold for inclusion in CSI calculations
    max_workers : int, optional
        Maximum number of worker threads for parallel processing
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing CSI values either at variable level (summary) or bin level (detailed)
    """
    # Get variables with support
    variables = binning_process.get_support(names=True)
    
    # Process variables in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(
            lambda var: process_variable(var, X_oot, X_train, binning_process, psi_min_bin_size),
            variables
        ))
    
    # Combine results
    df_psi_variable = pd.concat(results, ignore_index=True)
    
    # Return results based on requested style
    if style == "summary":
        summary_df = (df_psi_variable[df_psi_variable["Count Train (%)"] >= psi_min_bin_size]
                     .groupby(['Variable'])['CSI']
                     .sum()
                     .reset_index())
        # Ensure the summed CSI values are also rounded to 2 decimal places
        summary_df['CSI'] = summary_df['CSI'].round(2)
        return summary_df
    else:  # style == "detailed"
        return df_psi_variable