import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from openpyxl.drawing.image import Image
from openpyxl import Workbook
from typing import List, Optional, Union, Dict, Any



def enhanced_bivariate_plot(
    binning_process,
    filename: str = 'bivariate',
    metric: str = 'event_rate',
    variables: Optional[List[str]] = None,
    figsize: tuple = (12, 6),
    dpi: int = 100,
    style: str = 'whitegrid'
) -> None:
    """
    Create enhanced bivariate plots and export to Excel with special handling for 
    Totals (in index column), Special, and Missing bins.
    
    Parameters:
    -----------
    binning_process : OptimalBinning process object
        The binning process containing the variables to plot
    filename : str, default='bivariate'
        Base name for the output Excel file
    metric : str, default='event_rate'
        Metric to plot. Options: 'event_rate', 'woe'
    variables : list of str, optional
        List of specific variable names to process. If None, all variables are processed.
    figsize : tuple, default=(12, 6)
        Figure size for plots in inches (width, height)
    dpi : int, default=100
        Resolution of saved images
    style : str, default='whitegrid'
        Seaborn style for plots
    
    Returns:
    --------
    None
        Saves Excel file with embedded plots and data tables
    """
    # Validate metric parameter
    metric = metric.lower()
    if metric not in ['event_rate', 'woe']:
        raise ValueError("metric must be either 'event_rate' or 'woe'")
    
    # Get all available variables from binning process
    all_variables = binning_process.variable_names
    
    # Filter variables if specified
    if variables is not None:
        # Check if all specified variables exist in the binning process
        invalid_vars = set(variables) - set(all_variables)
        if invalid_vars:
            raise ValueError(f"Variables not found in binning process: {', '.join(invalid_vars)}")
        selected_vars = variables
    else:
        selected_vars = all_variables
    
    # Set up the visualization style
    plt.style.use('default')  # Clean matplotlib style

    # Create a temporary directory for images if it doesn't exist
    if not os.path.exists('temp_plots'):
        os.makedirs('temp_plots')
    
    # Dictionary to store plot paths
    plot_paths = {}
    
    # Determine column names based on metric
    metric_column = 'Event rate' if metric == 'event_rate' else 'WoE'
    y_axis_label = 'Event Rate' if metric == 'event_rate' else 'Weight of Evidence (WoE)'
    
    # Create Excel writer
    with pd.ExcelWriter(f'{filename}.xlsx', engine='openpyxl', mode='w') as writer:
        # Process each selected variable
        for var in selected_vars:
            # Sanitize variable name for Excel sheet name (remove invalid characters)
            # Excel sheet names cannot contain: [ ] : * ? / \
            sheet_name = str(var)
            for char in ['[', ']', ':', '*', '?', '/', '\\']:
                sheet_name = sheet_name.replace(char, '_')
            # Ensure sheet name is not longer than 31 characters (Excel limit)
            if len(sheet_name) > 31:
                sheet_name = sheet_name[:31]
            # Get the binned variable
            optb = binning_process.get_binned_variable(var)
            
            # Get DataFrame from binning table and clean it up
            df = optb.binning_table.build()
            if 'JS' in df.columns:
                df = df.drop(columns='JS')
            
            # Keep the original DataFrame for Excel output including Totals row
            df_for_excel = df.reset_index()
            
            # Extract the Totals row and remove it from the main DataFrame for plotting
            totals_row = None
            if 'Totals' in df.index:
                totals_row = df.loc['Totals']
                df = df.drop('Totals')
            
            # Reset index after removing Totals, making the index into a column
            df = df.reset_index()
            
            # The actual bin column is 'Bin', but if it doesn't exist, use the first column as fallback
            bin_col_name = 'Bin' if 'Bin' in df.columns else df.columns[0]
            
            # Write original DataFrame (including Totals) to Excel
            df_for_excel.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Create custom plot using Seaborn and Matplotlib
            plot_image_path = f'temp_plots/variable_{var}_{metric}.png'
            
            # Create figure with proper size to avoid cropping
            fig, ax1 = plt.subplots(figsize=figsize)
            
            # Fix: Convert bin column to string to safely compare with string values
            df[bin_col_name] = df[bin_col_name].astype(str)
            
            # Identify regular, special, and missing bins - safely using string comparisons
            regular_bins = df[~df[bin_col_name].str.contains('Special|Missing', regex=True, na=False)]
            special_bin = df[df[bin_col_name].str.contains('Special', regex=False, na=False)]
            missing_bin = df[df[bin_col_name].str.contains('Missing', regex=False, na=False)]
            
            # Create indices for x-axis
            x_indices = np.arange(len(df))
            
            # Format bin labels for better display
            bin_labels = []
            for _, row in df.iterrows():
                bin_name = row[bin_col_name]
                if 'Special' in bin_name or 'Missing' in bin_name:
                    bin_labels.append(bin_name)
                else:
                    # Use the actual bin values but truncate if too long
                    shortened_name = str(bin_name)
                    if len(shortened_name) > 15:
                        shortened_name = shortened_name[:12] + "..."
                    bin_labels.append(shortened_name)
            
            # Plot Count (%) as blue bars for all bins
            bars = ax1.bar(x_indices, df['Count (%)'] * 100, color='#0a3f7d', alpha=0.7)
            ax1.set_xticks(x_indices)
            ax1.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=9)
            ax1.set_xlabel('Bins', fontsize=10)
            ax1.set_ylabel('Count (%)', color='blue', fontsize=10)
            ax1.tick_params(axis='y', labelcolor='blue', labelsize=9)
            
            # Create second y-axis for Event Rate or WoE
            ax2 = ax1.twinx()
            line_color = 'darkgoldenrod'
            
            # Plot the metric line for regular bins only (connecting them)
            if not regular_bins.empty:
                # Get indices of regular bins in the full dataframe
                regular_mask = ~df[bin_col_name].str.contains('Special|Missing', regex=True, na=False)
                regular_indices = np.where(regular_mask)[0]
                
                if len(regular_indices) > 0:
                    # Plot connected line for regular bins
                    ax2.plot(regular_indices, regular_bins[metric_column], 
                             marker='o', color=line_color, linewidth=2, label=y_axis_label)
                    
                    # Add value annotations for regular bins with smaller font
                    for idx, val in zip(regular_indices, regular_bins[metric_column]):
                        ax2.annotate(f'{val:.3f}', 
                                     xy=(idx, val), 
                                     xytext=(0, 5),
                                     textcoords='offset points',
                                     ha='center', 
                                     fontsize=7)
            
            # Plot Special bin point (if exists) without connecting
            if not special_bin.empty:
                special_indices = df[df[bin_col_name].str.contains('Special', regex=False, na=False)].index
                for idx in special_indices:
                    special_val = df.loc[idx, metric_column]
                    ax2.plot(idx, special_val,
                             marker='s', color='red', markersize=8, linestyle='None', label='Special' if idx == special_indices[0] else "")
                    # Reduce decimal places and font size for special bin annotation
                    ax2.annotate(f'{special_val:.3f}', 
                                 xy=(idx, special_val), 
                                 xytext=(0, 5),
                                 textcoords='offset points',
                                 ha='center', 
                                 fontsize=7)
            
            # Plot Missing bin point (if exists) without connecting
            if not missing_bin.empty:
                missing_indices = df[df[bin_col_name].str.contains('Missing', regex=False, na=False)].index
                for idx in missing_indices:
                    missing_val = df.loc[idx, metric_column]
                    ax2.plot(idx, missing_val,
                             marker='D', color='purple', markersize=8, linestyle='None', label='Missing' if idx == missing_indices[0] else "")
                    # Reduce decimal places and font size for missing bin annotation
                    ax2.annotate(f'{missing_val:.3f}', 
                                 xy=(idx, missing_val), 
                                 xytext=(0, 5),
                                 textcoords='offset points',
                                 ha='center', 
                                 fontsize=7)
            
            # Add horizontal line for Totals if available
            if totals_row is not None:
                total_metric_value = totals_row[metric_column]
                
                # Handle case where WoE might be a blank string in Totals row
                if isinstance(total_metric_value, str) and total_metric_value.strip() == '':
                    total_metric_value = 0.0
                else:
                    # Try to convert to float in case it's a string representation of a number
                    try:
                        total_metric_value = float(total_metric_value)
                    except (ValueError, TypeError):
                        total_metric_value = 0.0
                
                ax2.axhline(y=total_metric_value, color='green', linestyle='--', 
                           alpha=0.7, label=f'Total {y_axis_label}: {total_metric_value:.4f}')
            
            # Set y-axis label and title
            ax2.set_ylabel(y_axis_label, color=line_color, fontsize=11)
            ax2.tick_params(axis='y', labelcolor=line_color)
            
            # Set title with variable name but keep it concise
            plt.title(f'{var}: {y_axis_label} by Bin', fontsize=12)
            
            # Add legend with small font size and optimize position
            handles, labels = ax2.get_legend_handles_labels()
            # Remove duplicate labels in legend
            by_label = dict(zip(labels, handles))
            ax2.legend(by_label.values(), by_label.keys(), loc='best', fontsize=9)
            
            # Adjust layout to prevent cropping
            plt.tight_layout()
            
            # Save the figure with high quality
            plt.savefig(plot_image_path, dpi=dpi, bbox_inches='tight')
            plt.close()
            
            # Store the path for later cleanup
            plot_paths[var] = plot_image_path
            
            # Insert plot image into the Excel sheet
            img = Image(plot_image_path)
            
            # Calculate position based on data size (including Totals row)
            # Start plot after the data table with some margin
            row_position = len(df_for_excel) + 4  # +4 for margin
            img.anchor = f'A{row_position}'
            
            writer.sheets[sheet_name].add_image(img)
            
            # Adjust column widths for better readability
            for idx, col in enumerate(df_for_excel.columns):
                column_width = max(len(str(col)), df_for_excel[col].astype(str).map(len).max())
                writer.sheets[sheet_name].column_dimensions[chr(65 + idx)].width = column_width + 2
    
    # Clean up temporary image files
    for path in plot_paths.values():
        if os.path.exists(path):
            os.remove(path)
    
    # Remove temp directory if empty
    if os.path.exists('temp_plots') and not os.listdir('temp_plots'):
        os.rmdir('temp_plots')
    
    print(f"Enhanced bivariate analysis completed! Results saved to {filename}.xlsx")


def transform_and_plot_oot_bivariate_data(
    binning_process,
    oot_data: pd.DataFrame,
    target_column: str,
    filename: str = 'oot_bivariate',
    metric: str = 'event_rate',
    variables: Optional[List[str]] = None,
    figsize: tuple = (12, 6),
    dpi: int = 100,
    style: str = 'whitegrid',
    compare_with_train: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Transform OOT data using a trained binning process,
    calculate binning tables and create bivariate plots.
    
    Parameters:
    -----------
    binning_process : BinningProcess object
        The trained binning process to use for transformation
    oot_data : pandas DataFrame
        The OOT dataset to transform and analyze
    target_column : str
        Name of the target/outcome column in the OOT data
    filename : str, default='oot_bivariate'
        Base name for the output Excel file
    metric : str, default='event_rate'
        Metric to plot. Options: 'event_rate', 'woe'
    variables : list of str, optional
        List of specific variable names to process. If None, all variables are processed.
    figsize : tuple, default=(12, 6)
        Figure size for plots in inches (width, height)
    dpi : int, default=100
        Resolution of saved images
    style : str, default='whitegrid'
        Seaborn style for plots
    compare_with_train : bool, default=True
        Whether to plot training data metrics alongside OOT data metrics
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary containing the binning tables for each transformed variable
    """
    # Validate metric parameter
    metric = metric.lower()
    if metric not in ['event_rate', 'woe']:
        raise ValueError("metric must be either 'event_rate' or 'woe'")
    
    # Get all available variables from binning process
    try:
        # Fall back to variable_names if get_support is not available
        all_variables = binning_process.variable_names        
    except AttributeError:
        all_variables = binning_process.get_support(names=True)
    
    # Filter variables if specified
    if variables is not None:
        # Check if all specified variables exist in the binning process
        invalid_vars = set(variables) - set(all_variables)
        if invalid_vars:
            raise ValueError(f"Variables not found in binning process: {', '.join(invalid_vars)}")
        selected_vars = variables
    else:
        selected_vars = all_variables
    
    # Set up the visualization style
    plt.style.use('default')  # Clean matplotlib style
    
    # Create a temporary directory for images if it doesn't exist
    if not os.path.exists('temp_plots'):
        os.makedirs('temp_plots')
    
    # Dictionary to store plot paths and binning tables
    plot_paths = {}
    binning_tables = {}
    
    # Determine column names based on metric
    metric_column = 'Event rate' if metric == 'event_rate' else 'WoE'
    y_axis_label = 'Event Rate' if metric == 'event_rate' else 'Weight of Evidence (WoE)'
    
    # Create Excel writer
    with pd.ExcelWriter(f'{filename}.xlsx', engine='openpyxl', mode='w') as writer:
        # Process each selected variable
        for var in selected_vars:
            # Sanitize variable name for Excel sheet name (remove invalid characters)
            # Excel sheet names cannot contain: [ ] : * ? / \
            sheet_name = str(var)
            for char in ['[', ']', ':', '*', '?', '/', '\\']:
                sheet_name = sheet_name.replace(char, '_')
            # Ensure sheet name is not longer than 31 characters (Excel limit)
            if len(sheet_name) > 31:
                sheet_name = sheet_name[:31]
                
            # Get the binned variable from the trained process
            try:
                optb = binning_process.get_binned_variable(var)
            except Exception as e:
                print(f"Error getting binned variable {var}: {str(e)}")
                continue
            
            # Get original binning table from training data
            train_df = optb.binning_table.build()
            if 'JS' in train_df.columns:
                train_df = train_df.drop(columns='JS')
            
            # Transform the OOT data for this variable using the trained binning
            # First, extract the feature vector for this variable
            try:
                X_var = oot_data[var].values
                y_oot = oot_data[target_column].values
            except KeyError as e:
                print(f"Variable {var} or target column {target_column} not found in OOT data: {str(e)}")
                continue
            
            # Use the transform method with "bins" metric to get bin assignments
            try:
                oot_bins = optb.transform(X_var, metric="bins")
            except Exception as e:
                print(f"Error transforming OOT data for variable {var}: {str(e)}")
                continue
            
            # Create a mapping of bin labels to indices
            train_df_reset = train_df.reset_index() if 'Totals' not in train_df.index else train_df.drop('Totals').reset_index()
            bin_col_name = 'Bin' if 'Bin' in train_df_reset.columns else train_df_reset.columns[0]
            
            # Create a dictionary to store counts for each bin
            bin_n_event = {bin_name: 0 for bin_name in train_df_reset[bin_col_name]}
            bin_n_nonevent = {bin_name: 0 for bin_name in train_df_reset[bin_col_name]}
            
            # Count events and non-events for each bin in OOT data
            for i, bin_label in enumerate(oot_bins):
                if bin_label in bin_n_event:
                    if y_oot[i] == 1:
                        bin_n_event[bin_label] += 1
                    else:
                        bin_n_nonevent[bin_label] += 1
            
            # Calculate totals for percentages
            total_events = sum(bin_n_event.values())
            total_nonevents = sum(bin_n_nonevent.values())
            total_records = total_events + total_nonevents
            
            # Create a new DataFrame for OOT data metrics
            oot_stats = []
            for bin_name in train_df_reset[bin_col_name]:
                n_event = bin_n_event[bin_name]
                n_nonevent = bin_n_nonevent[bin_name]
                n_records = n_event + n_nonevent
                
                # Calculate event rate for this bin
                event_rate = n_event / n_records if n_records > 0 else 0
                
                # Calculate WoE similar to the binning_table.build() method
                if n_event > 0 and n_nonevent > 0 and total_events > 0 and total_nonevents > 0:
                    p_event = n_event / total_events
                    p_nonevent = n_nonevent / total_nonevents
                    woe = np.log(p_nonevent / p_event)
                else:
                    woe = 0  # or handle this differently if needed
                
                oot_stats.append({
                    bin_col_name: bin_name,
                    'Count': n_records,
                    'Count (%)': n_records / total_records if total_records > 0 else 0,
                    'Non-event': n_nonevent,
                    'Event': n_event,
                    'Event rate': event_rate,
                    'WoE': woe,
                })
            
            # Create DataFrame from collected statistics
            oot_df = pd.DataFrame(oot_stats)
            
            # Calculate total event rate
            total_event_rate = total_events / total_records if total_records > 0 else 0
            
            # Add totals row
            totals_row = {
                bin_col_name: 'Totals',
                'Count': total_records,
                'Count (%)': 1.0,
                'Non-event': total_nonevents,
                'Event': total_events,
                'Event rate': total_event_rate,
                'WoE': ''  # WoE doesn't make sense for totals
            }
            
            # Store OOT binning table for return
            oot_df_with_totals = oot_df.copy()
            oot_df_with_totals = pd.concat([oot_df_with_totals, pd.DataFrame([totals_row])], ignore_index=True)
            binning_tables[var] = oot_df_with_totals
            
            # Write transformed DataFrame to Excel - make sure at least one sheet is created
            oot_df_with_totals.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Create custom plot with comparison
            plot_image_path = f'temp_plots/variable_{var}_{metric}_oot.png'
            
            # Create figure with proper size
            fig, ax1 = plt.subplots(figsize=figsize)
            
            # Fix: Convert bin column to string to safely compare with string values
            oot_df[bin_col_name] = oot_df[bin_col_name].astype(str)
            
            # Identify regular, special, and missing bins - safely using string comparisons
            regular_bins = oot_df[~oot_df[bin_col_name].str.contains('Special|Missing', regex=True, na=False)]
            special_bin = oot_df[oot_df[bin_col_name].str.contains('Special', regex=False, na=False)]
            missing_bin = oot_df[oot_df[bin_col_name].str.contains('Missing', regex=False, na=False)]
            
            # Create indices for x-axis
            x_indices = np.arange(len(oot_df))
            
            # Format bin labels for better display
            bin_labels = []
            for _, row in oot_df.iterrows():
                bin_name = row[bin_col_name]
                if 'Special' in bin_name or 'Missing' in bin_name:
                    bin_labels.append(bin_name)
                else:
                    # Use the actual bin values but truncate if too long
                    shortened_name = str(bin_name)
                    if len(shortened_name) > 15:
                        shortened_name = shortened_name[:12] + "..."
                    bin_labels.append(shortened_name)
            
            # Width of each bar
            bar_width = 0.35
            
            # Create side-by-side bars instead of a single bar
            if compare_with_train:
                # Get training data percentages
                train_df_no_totals = train_df.drop('Totals') if 'Totals' in train_df.index else train_df
                train_df_reset = train_df_no_totals.reset_index()
                
                # Ensure bin column in train_df_reset is string for consistent comparison
                train_df_reset[bin_col_name] = train_df_reset[bin_col_name].astype(str)
                
                # Initialize array for train percentages
                train_percentages = np.zeros(len(oot_df))
                
                # Fill in train percentages for matching bins
                for idx, oot_row in oot_df.iterrows():
                    bin_value = oot_row[bin_col_name]
                    # Find matching bin in training data
                    train_match = train_df_reset[train_df_reset[bin_col_name] == bin_value]
                    if not train_match.empty:
                        train_percentages[idx] = train_match['Count (%)'].values[0] * 100
                
                # Plot train data bars on the left
                train_bars = ax1.bar(x_indices - bar_width/2, train_percentages, 
                                    bar_width, color='green', alpha=0.7, label='Train Count (%)')
                
                # Plot OOT data bars on the right
                oot_bars = ax1.bar(x_indices + bar_width/2, oot_df['Count (%)'] * 100, 
                                  bar_width, color='#0a3f7d', alpha=0.7, label='OOT Count (%)')
                
                # Add bar value labels
                for idx, (train_val, oot_val) in enumerate(zip(train_percentages, oot_df['Count (%)'] * 100)):
                    if train_val > 0:
                        ax1.text(idx - bar_width/2, train_val + 0.5, f'{train_val:.1f}%', 
                                ha='center', va='bottom', fontsize=7, rotation=90)
                    if oot_val > 0:
                        ax1.text(idx + bar_width/2, oot_val + 0.5, f'{oot_val:.1f}%', 
                                ha='center', va='bottom', fontsize=7, rotation=90)
            else:
                # Just plot OOT data bars centered
                oot_bars = ax1.bar(x_indices, oot_df['Count (%)'] * 100, 
                                  bar_width*1.5, color='#0a3f7d', alpha=0.7, label='OOT Count (%)')
                
                # Add bar value labels
                for idx, oot_val in enumerate(oot_df['Count (%)'] * 100):
                    if oot_val > 0:
                        ax1.text(idx, oot_val + 0.5, f'{oot_val:.1f}%', 
                                ha='center', va='bottom', fontsize=7, rotation=90)
            
            ax1.set_xticks(x_indices)
            ax1.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=9)
            ax1.set_xlabel('Bins', fontsize=10)
            ax1.set_ylabel('Count (%)', color='blue', fontsize=10)
            ax1.tick_params(axis='y', labelcolor='blue', labelsize=9)
            
            # Create second y-axis for Event Rate or WoE
            ax2 = ax1.twinx()
            line_color = 'darkgoldenrod'
            
            # Plot the metric line for regular bins only in OOT data
            if not regular_bins.empty:
                # Get indices of regular bins in the full dataframe
                regular_mask = ~oot_df[bin_col_name].str.contains('Special|Missing', regex=True, na=False)
                regular_indices = np.where(regular_mask)[0]
                
                if len(regular_indices) > 0:
                    # Plot connected line for regular bins
                    ax2.plot(regular_indices, regular_bins[metric_column], 
                             marker='o', color=line_color, linewidth=2, 
                             label=f'OOT {y_axis_label}')
                    
                    # Add value annotations for OOT data
                    for idx, val in zip(regular_indices, regular_bins[metric_column]):
                        ax2.annotate(f'{val:.3f}', 
                                     xy=(idx, val), 
                                     xytext=(0, 5),
                                     textcoords='offset points',
                                     ha='center', 
                                     fontsize=7)
            
            # Add comparison with training data if requested
            if compare_with_train:
                # Get training data binning table (excluding totals)
                train_df_no_totals = train_df.drop('Totals') if 'Totals' in train_df.index else train_df
                train_df_reset = train_df_no_totals.reset_index()
                
                # Ensure bin column in train_df_reset is string for consistent comparison
                train_df_reset[bin_col_name] = train_df_reset[bin_col_name].astype(str)
                
                # Prepare train metric values for regular bins only
                if not regular_bins.empty:
                    # Initialize arrays for train metric values
                    train_regular_indices = []
                    train_regular_values = []
                    
                    # Collect train metric values for regular bins
                    for idx, bin_val in enumerate(oot_df[bin_col_name]):
                        if not any(substring in bin_val for substring in ['Special', 'Missing']):
                            # Find matching bin in training data
                            train_match = train_df_reset[train_df_reset[bin_col_name] == bin_val]
                            if not train_match.empty:
                                train_regular_indices.append(idx)
                                train_regular_values.append(train_match[metric_column].values[0])
                    
                    # Plot train metric line for regular bins
                    if train_regular_indices and train_regular_values:
                        ax2.plot(train_regular_indices, train_regular_values, 
                               marker='x', color='green', linewidth=2, linestyle='-',
                               label=f'Train {y_axis_label}')
                        
                        # Add value annotations for train data
                        for idx, val in zip(train_regular_indices, train_regular_values):
                            ax2.annotate(f'{val:.3f}', 
                                       xy=(idx, val), 
                                       xytext=(0, -12),  # Position below point
                                       textcoords='offset points',
                                       ha='center', 
                                       fontsize=7)
            
            # Plot Special bin point (if exists) without connecting for OOT data
            if not special_bin.empty:
                special_indices = oot_df[oot_df[bin_col_name].str.contains('Special', regex=False, na=False)].index
                for idx in special_indices:
                    special_val = oot_df.loc[idx, metric_column]
                    ax2.plot(idx, special_val,
                            marker='s', color='red', markersize=8, linestyle='None', 
                            label='Special (OOT)' if idx == special_indices[0] else "")
                    # Annotate special bin
                    ax2.annotate(f'{special_val:.3f}', 
                                xy=(idx, special_val), 
                                xytext=(0, 5),
                                textcoords='offset points',
                                ha='center', 
                                fontsize=7)
            
            # Plot Missing bin point (if exists) without connecting for OOT data
            if not missing_bin.empty:
                missing_indices = oot_df[oot_df[bin_col_name].str.contains('Missing', regex=False, na=False)].index
                for idx in missing_indices:
                    missing_val = oot_df.loc[idx, metric_column]
                    ax2.plot(idx, missing_val,
                            marker='D', color='purple', markersize=8, linestyle='None', 
                            label='Missing (OOT)' if idx == missing_indices[0] else "")
                    # Annotate missing bin
                    ax2.annotate(f'{missing_val:.3f}', 
                                xy=(idx, missing_val), 
                                xytext=(0, 5),
                                textcoords='offset points',
                                ha='center', 
                                fontsize=7)
            
            # Add horizontal lines for Totals
            if metric == 'event_rate':
                # OOT data total event rate
                ax2.axhline(y=total_event_rate, color=line_color, linestyle='--', 
                          alpha=0.7, label=f'OOT Total: {total_event_rate:.4f}')
                
                # Training data total event rate
                if compare_with_train and 'Totals' in train_df.index:
                    train_total_val = train_df.loc['Totals'][metric_column]
                    
                    # Handle case where value might be a string
                    if isinstance(train_total_val, str) and train_total_val.strip() == '':
                        train_total_val = 0.0
                    else:
                        try:
                            train_total_val = float(train_total_val)
                        except (ValueError, TypeError):
                            train_total_val = 0.0
                    
                    ax2.axhline(y=train_total_val, color='green', linestyle='--', 
                              alpha=0.7, label=f'Train Total: {train_total_val:.4f}')
            else:  # WoE metric
                # For WoE, add a horizontal line at 0 for reference
                ax2.axhline(y=0, color=line_color, linestyle='--', 
                          alpha=0.7, label='WoE = 0 (Reference)')
            
            # Set y-axis label and title
            ax2.set_ylabel(y_axis_label, color=line_color, fontsize=11)
            ax2.tick_params(axis='y', labelcolor=line_color)
            
            # Set title with variable name
            plt.title(f'{var}: {y_axis_label} Comparison', fontsize=12)
            
            # Add legend with small font size and optimize position
            handles, labels = [], []
            for handle, label in zip(*ax1.get_legend_handles_labels()):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
            for handle, label in zip(*ax2.get_legend_handles_labels()):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
            ax1.legend(handles, labels, loc='best', fontsize=8)
            
            # Adjust layout to prevent cropping
            plt.tight_layout()
            
            # Save the figure with high quality
            plt.savefig(plot_image_path, dpi=dpi, bbox_inches='tight')
            plt.close()
            
            # Store the path for later cleanup
            plot_paths[var] = plot_image_path
            
            # Insert plot image into the Excel sheet
            try:
                img = Image(plot_image_path)
                
                # Calculate position based on data size (including Totals row)
                # Start plot after the data table with some margin
                row_position = len(oot_df_with_totals) + 4  # +4 for margin
                img.anchor = f'A{row_position}'
                
                writer.sheets[sheet_name].add_image(img)
                
                # Adjust column widths for better readability
                for idx, col in enumerate(oot_df_with_totals.columns):
                    column_width = max(len(str(col)), oot_df_with_totals[col].astype(str).map(len).max())
                    writer.sheets[sheet_name].column_dimensions[chr(65 + idx)].width = column_width + 2
            except Exception as e:
                print(f"Error adding image to Excel for variable {var}: {str(e)}")
    
    # Clean up temporary image files
    for path in plot_paths.values():
        if os.path.exists(path):
            os.remove(path)
    
    # Remove temp directory if empty
    if os.path.exists('temp_plots') and not os.listdir('temp_plots'):
        os.rmdir('temp_plots')
    
    print(f"OOT data bivariate analysis completed! Results saved to {filename}.xlsx")
    
    return binning_tables