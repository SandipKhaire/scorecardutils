import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from openpyxl.drawing.image import Image
from typing import List, Optional, Union


def enhanced_bivariate_plot(
    binning_process,
    filename: str = 'bivariate',
    metric: str = 'event_rate',
    variables: Optional[List[str]] = None,
    figsize: tuple = (12, 6),  # Reduced figure size for better viewing
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
    figsize : tuple, default=(14, 8)
        Figure size for plots in inches (width, height)
    dpi : int, default=300
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
    #sns.set_style(style)
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
            # Get the binned variable
            optb = binning_process.get_binned_variable(var)
            
            # Get DataFrame from binning table and clean it up
            df = optb.binning_table.build()
            #df["Event rate"] = (df["Event rate"] * 100).round(1)
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
            df_for_excel.to_excel(writer, sheet_name=var, index=False)
            
            # Create custom plot using Seaborn and Matplotlib
            plot_image_path = f'temp_plots/variable_{var}_{metric}.png'
            
            # Create figure with proper size to avoid cropping
            fig, ax1 = plt.subplots(figsize=figsize)
            
            # Identify regular, special, and missing bins
            # Explicitly check for strings 'Special' and 'Missing' in the bins column
            regular_bins = df[(df[bin_col_name] != 'Special') & (df[bin_col_name] != 'Missing')]
            special_bin = df[df[bin_col_name] == 'Special'] if 'Special' in df[bin_col_name].values else pd.DataFrame()
            missing_bin = df[df[bin_col_name] == 'Missing'] if 'Missing' in df[bin_col_name].values else pd.DataFrame()
            
            # Create indices for x-axis
            x_indices = np.arange(len(df))
            
            # Format bin labels for better display
            bin_labels = []
            for _, row in df.iterrows():
                bin_name = row[bin_col_name]
                if bin_name in ['Special', 'Missing']:
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
                regular_mask = df[bin_col_name].apply(lambda x: x not in ['Special', 'Missing'])
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
                special_idx = df[df[bin_col_name] == 'Special'].index[0]
                special_val = special_bin[metric_column].values[0]
                ax2.plot(special_idx, special_val,
                         marker='s', color='red', markersize=8, linestyle='None', label='Special')
                # Reduce decimal places and font size for special bin annotation
                ax2.annotate(f'{special_val:.3f}', 
                             xy=(special_idx, special_val), 
                             xytext=(0, 5),
                             textcoords='offset points',
                             ha='center', 
                             fontsize=7)
            
            # Plot Missing bin point (if exists) without connecting
            if not missing_bin.empty:
                missing_idx = df[df[bin_col_name] == 'Missing'].index[0]
                missing_val = missing_bin[metric_column].values[0]
                ax2.plot(missing_idx, missing_val,
                         marker='D', color='purple', markersize=8, linestyle='None', label='Missing')
                # Reduce decimal places and font size for missing bin annotation
                ax2.annotate(f'{missing_val:.3f}', 
                             xy=(missing_idx, missing_val), 
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
            lines, labels = ax2.get_legend_handles_labels()
            ax2.legend(lines, labels, loc='best', fontsize=9)
            
            # Add grid for better visualization
            #ax1.grid(axis='y', linestyle='--', alpha=0.5)
            
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
            
            writer.sheets[var].add_image(img)
            
            # Adjust column widths for better readability
            for idx, col in enumerate(df_for_excel.columns):
                column_width = max(len(str(col)), df_for_excel[col].astype(str).map(len).max())
                writer.sheets[var].column_dimensions[chr(65 + idx)].width = column_width + 2
    
    # Clean up temporary image files
    for path in plot_paths.values():
        if os.path.exists(path):
            os.remove(path)
    
    # Remove temp directory if empty
    if os.path.exists('temp_plots') and not os.listdir('temp_plots'):
        os.rmdir('temp_plots')
    
    print(f"Enhanced bivariate analysis completed! Results saved to {filename}.xlsx")


# Example usage with different metrics and variable selections
# 1. Process all variables with event rate
# enhanced_bivariate_plot(binning_process, filename='bivariate_event_rate', metric='event_rate')

# 2. Process all variables with WoE
# enhanced_bivariate_plot(binning_process, filename='bivariate_woe', metric='woe')

# 3. Process only specific variables
# enhanced_bivariate_plot(
#     binning_process, 
#     filename='selected_variables', 
#     metric='event_rate',
#     variables=['age', 'income', 'credit_score']
# )