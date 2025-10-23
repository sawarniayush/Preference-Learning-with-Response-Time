import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse


def _detect_x_column(df):
    """Return the preferred x-axis column present in the DataFrame."""
    for candidate in ['train_size', 'threshold']:
        if candidate in df.columns:
            return candidate
    return None


def _load_summary_dataframe(input_file):
    """
    Load summary statistics while normalising header rows produced by pandas.MultiIndex.to_csv.
    Returns a DataFrame with flat column names such as 'algorithm', 'train_size', 'accuracy_mean'.
    """
    try:
        summary_df = pd.read_csv(input_file, header=[0, 1])
        if isinstance(summary_df.columns, pd.MultiIndex):
            first_row = summary_df.iloc[0]
            flat_columns = []
            for top, bottom in summary_df.columns:
                if 'Unnamed: 0' in top:
                    name = str(first_row[(top, bottom)]).strip() or 'algorithm'
                    flat_columns.append(name)
                elif 'Unnamed: 1' in top:
                    name = str(first_row[(top, bottom)]).strip() or bottom
                    flat_columns.append(name or 'train_size')
                else:
                    suffix = bottom if bottom and bottom == bottom else ''
                    flat_columns.append(f"{top}_{suffix}".strip('_'))
            summary_df.columns = flat_columns
            summary_df = summary_df.iloc[1:].reset_index(drop=True)
    except ValueError:
        summary_df = pd.read_csv(input_file)

    # Normalise column names
    rename_map = {}
    for col in summary_df.columns:
        if col.lower() == 'algorithm':
            rename_map[col] = 'algorithm'
        elif col.lower() in {'train_size', 'threshold'}:
            rename_map[col] = col.lower()
        elif '_' in col:
            rename_map[col] = col.lower()
    if rename_map:
        summary_df = summary_df.rename(columns=rename_map)

    # Convert numeric columns where possible
    for col in summary_df.columns:
        if col != 'algorithm':
            summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce')
    return summary_df

def plot_from_raw_results(input_file, output_dir, selected_algorithms=None, min_size=None, max_size=None, font_size1=20, font_size2=15):
    """Generate plots from raw results CSV file"""
    # Load the raw results
    results_df = pd.read_csv(input_file)
    
    # Filter algorithms if specified
    if selected_algorithms:
        results_df = results_df[results_df['algorithm'].isin(selected_algorithms)]
        if results_df.empty:
            print(f"Warning: No data found for the selected algorithms in {input_file}")
            return
    
    # Filter by training size if specified
    if min_size is not None and 'train_size' in results_df.columns:
        results_df = results_df[results_df['train_size'] >= min_size]
    if max_size is not None and 'train_size' in results_df.columns:
        results_df = results_df[results_df['train_size'] <= max_size]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate each type of plot
    for metric in ['accuracy', 'mse', 'regret']:
        if metric in results_df.columns:
            output_file = os.path.join(output_dir, f"{metric}_plot.png")
            x_column = _detect_x_column(results_df)
            if not x_column:
                print(f"Skipping {metric} plot: no x-axis column ('train_size' or 'threshold') found.")
                continue
            plot_metric(results_df, metric, output_file, x_column, font_size1, font_size2)
            print(f"Generated {output_file}")

def plot_from_summary(input_file, output_dir, selected_algorithms=None, min_size=None, max_size=None):
    """Generate plots from summary statistics CSV file"""
    # Load summary statistics
    summary_df = _load_summary_dataframe(input_file)
    
    # Display column information for debugging
    print(f"Available columns in summary file: {summary_df.columns.tolist()}")
    
    # First row as sample
    if not summary_df.empty:
        print(f"Sample data from summary file (first row): {summary_df.iloc[0].to_dict()}")
    
    # Check if 'algorithm' column exists directly
    if 'algorithm' not in summary_df.columns:
        # Try to find the correct column containing algorithm names
        algorithm_col = None
        for col in summary_df.columns:
            # Check if any column contains algorithm file names
            if summary_df[col].dtype == 'object' and any('.py' in str(val) for val in summary_df[col].dropna()):
                algorithm_col = col
                print(f"Using column '{algorithm_col}' for algorithm filtering")
                break
        
        if not algorithm_col:
            print("Cannot find a column containing algorithm names in the summary file.")
            print("Plotting all data without filtering by algorithm.")
        else:
            # Rename the column for consistency
            summary_df = summary_df.rename(columns={algorithm_col: 'algorithm'})
    
    # Filter algorithms if specified and if we have an algorithm column
    if selected_algorithms and 'algorithm' in summary_df.columns:
        summary_df = summary_df[summary_df['algorithm'].isin(selected_algorithms)]
        if summary_df.empty:
            print(f"Warning: No data found for the selected algorithms in {input_file}")
            return
    
    # Filter by training size if specified
    if 'train_size' in summary_df.columns:
        if min_size is not None:
            summary_df = summary_df[summary_df['train_size'] >= min_size]
        if max_size is not None:
            summary_df = summary_df[summary_df['train_size'] <= max_size]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse the multi-level index columns
    summary_df.columns = [col if not isinstance(col, tuple) else '_'.join(col) for col in summary_df.columns]
    
    # Generate each type of plot
    for metric in ['accuracy', 'mse', 'regret']:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        
        if mean_col in summary_df.columns and std_col in summary_df.columns:
            output_file = os.path.join(output_dir, f"{metric}_plot.png")
            x_column = _detect_x_column(summary_df)
            if not x_column:
                print(f"Skipping {metric} plot: no x-axis column ('train_size' or 'threshold') found.")
                continue
            plot_metric_from_summary(summary_df, mean_col, std_col, metric, output_file, x_column)
            print(f"Generated {output_file}")

def plot_metric(results_df, metric, output_file, x_column, font_size1=20, font_size2=15):
    """Create a plot for the given metric with confidence bands"""
    plt.figure(figsize=(10, 7))

    # Group by algorithm and train_size
    grouped = results_df.groupby(['algorithm', x_column])[metric].agg(['mean', 'std'])

    # Colors for different algorithms
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    markers = ['o', 's', '^', 'D', 'v', '>', '<']
    
    # Print std values for debugging
    print(f"\nStandard deviations for {metric}:")
    for algorithm in sorted(results_df['algorithm'].unique()):
        data = grouped.loc[algorithm]
        print(f"  {algorithm}: {data['std'].to_dict()}")
    
    algorithm_mapping = {
        'log_loss_learner.py': 'Log Loss',
        'orthogonal_loss_learner.py': 'Ortho Loss', 
        'orthogonal_loss_learner_splitting.py': 'Ortho Loss with Splitting',
        'nonorthogonal_loss_learner.py': 'Non-ortho Loss'
    }
    algorithm_order = ['Log Loss', 'Ortho Loss',  'Non-ortho Loss', 'Ortho Loss with Splitting']
    
    # Create reverse mapping to find original algorithm names
    reverse_mapping = {v: k for k, v in algorithm_mapping.items()}
    
    # Iterate through algorithm_order to ensure plotting happens in the specified order
    for i, alg_display in enumerate(algorithm_order):
        if alg_display not in reverse_mapping:
            continue
            
        algorithm = reverse_mapping[alg_display]
        if algorithm not in results_df['algorithm'].unique():
            continue
            
        color_idx = i % len(colors)
        marker_idx = i % len(markers)
        
        data = grouped.loc[algorithm]
        x = data.index.to_numpy()
        y = data['mean'].to_numpy()
        std = data['std'].fillna(0).to_numpy()
        if metric == 'regret':
            std = 3000 * std
            y = 3000 * y
        # Use line plot with shaded confidence band
        plt.plot(x, y, 
                 color=colors[color_idx], 
                 marker=markers[marker_idx], 
                 markersize=6, 
                 linewidth=2, 
                 label=alg_display)
                 
        # Add shaded confidence band (use full std for consistency)
        plt.fill_between(x, 
                        y - 0.2 * std, 
                        y + 0.2 * std, 
                        color=colors[color_idx], 
                        alpha=0.09)  # Increased alpha for better visibility

    axis_label = 'Training Size' if x_column == 'train_size' else x_column.replace('_', ' ').title()
    plt.xlabel(axis_label, fontsize=font_size1)
    if metric == 'mse':
        plt.ylabel('Mean Squared Error', fontsize=font_size1)
        plt.title(f"Mean Squared Error vs {axis_label}", fontsize=font_size1)
    else:
        plt.ylabel(metric.capitalize(), fontsize=font_size1)
        plt.title(f'{metric.capitalize()} vs {axis_label}', fontsize=font_size1)
    plt.xticks(fontsize=font_size2)
    plt.yticks(fontsize=font_size2)
    plt.legend(loc='upper right', fontsize=font_size2, framealpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_metric_from_summary(summary_df, mean_col, std_col, metric_name, output_file, x_column):
    """Create a plot for the given metric with confidence bands from summary stats"""
    plt.figure(figsize=(12, 6))

    # Colors for different algorithms
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    markers = ['o', 's', '^', 'D', 'v', '>', '<']
    
    # Get unique algorithms and train sizes
    if 'algorithm' in summary_df.columns:
        # Use algorithm column if it exists
        algorithm_col = 'algorithm'
        algorithms = summary_df[algorithm_col].unique()
    else:
        # Otherwise, we'll use row index or another identifier
        print("Warning: No 'algorithm' column found. Using row index as algorithm identifier.")
        summary_df['algorithm_id'] = [f"Algorithm {i}" for i in range(len(summary_df))]
        algorithm_col = 'algorithm_id'
        algorithms = summary_df[algorithm_col].unique()
    
    # Print std values for debugging
    print(f"\nStandard deviations for {metric_name}:")
    for algorithm in sorted(algorithms):
        alg_data = summary_df[summary_df[algorithm_col] == algorithm]
        x_vals = alg_data[x_column].to_numpy()
        std_vals = alg_data[std_col].to_numpy()
        std_values = {x_vals[i]: std_vals[i] for i in range(len(x_vals))}
        print(f"  {algorithm}: {std_values}")
    
    for i, algorithm in enumerate(sorted(algorithms)):
        color_idx = i % len(colors)
        marker_idx = i % len(markers)
        
        # Filter data for this algorithm
        alg_data = summary_df[summary_df[algorithm_col] == algorithm]
        
        # Get x, y, and std values
        x = alg_data[x_column].to_numpy()
        y = alg_data[mean_col].to_numpy()
        std = alg_data[std_col].fillna(0).to_numpy()

        # Use line plot with shaded confidence band
        plt.plot(x, y, 
                 color=colors[color_idx], 
                 marker=markers[marker_idx], 
                 markersize=6, 
                 linewidth=2, 
                 label=str(algorithm).replace('.py', ''))
                 
        # Add shaded confidence band
        plt.fill_between(x, 
                        y - std, 
                        y + std, 
                        color=colors[color_idx], 
                        alpha=0.3)  # Increased alpha for better visibility
    
    # Make figure more square-shaped
    plt.gcf().set_size_inches(10, 10)  # Set equal width and height
    
    axis_label = 'Training Size' if x_column == 'train_size' else x_column.replace('_', ' ').title()
    plt.xlabel(axis_label, fontsize=20)
    plt.ylabel(metric_name.capitalize(), fontsize=20)
    plt.title(f'{metric_name.capitalize()} vs {axis_label}', fontsize=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots from existing benchmark results")
    parser.add_argument('--raw', help='Path to raw_results.csv file', default='results/raw_results.csv')
    parser.add_argument('--summary', help='Path to summary_statistics.csv file', default='results/summary_statistics.csv')
    parser.add_argument('--output', help='Output directory for plots', default='results')
    parser.add_argument('--use', choices=['raw', 'summary', 'both'], default='raw',
                        help='Use raw results, summary statistics, or both (default: raw)')
    parser.add_argument('--algorithms', nargs='+', help='List of algorithms to include in the plots')
    parser.add_argument('--min-size', type=int, help='Minimum training size to include')
    parser.add_argument('--max-size', type=int, help='Maximum training size to include')
    parser.add_argument('--font-size1', type=int, default=20, help='Font size for titles and labels')
    parser.add_argument('--font-size2', type=int, default=15, help='Font size for ticks and legend')
    
    args = parser.parse_args()
    
    if args.use == 'raw' or args.use == 'both':
        if os.path.exists(args.raw):
            print(f"Generating plots from raw results: {args.raw}")
            print(f"Selected algorithms: {args.algorithms or 'All'}")
            print(f"Training size range: {args.min_size or 'min'} to {args.max_size or 'max'}")
            plot_from_raw_results(args.raw, args.output, args.algorithms, args.min_size, args.max_size, 
                                args.font_size1, args.font_size2)
        else:
            print(f"Raw results file not found: {args.raw}")
    
    if args.use == 'summary' or args.use == 'both':
        if os.path.exists(args.summary):
            print(f"Generating plots from summary statistics: {args.summary}")
            print(f"Selected algorithms: {args.algorithms or 'All'}")
            print(f"Training size range: {args.min_size or 'min'} to {args.max_size or 'max'}")
            plot_from_summary(args.summary, args.output, args.algorithms, args.min_size, args.max_size)
        else:
            print(f"Summary statistics file not found: {args.summary}") 
