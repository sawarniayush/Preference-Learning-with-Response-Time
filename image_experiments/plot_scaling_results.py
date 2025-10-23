import os
import sys
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from datetime import datetime
import argparse

def calculate_stats(results):
    """Calculate mean and std for policy value, MSE, and regret."""
    if not results:
        return None
    
    policy_values = [r["policy_value"] for r in results if r["policy_value"] is not None]
    mses = [r["mse"] for r in results if r["mse"] is not None]
    regrets = [r["regret"] for r in results if r["regret"] is not None]
    
    if not policy_values or not mses:
        return None
    
    return {
        "policy_value_mean": np.mean(policy_values),
        "policy_value_std": np.std(policy_values),
        "mse_mean": np.mean(mses),
        "mse_std": np.std(mses),
        "regret_mean": np.mean(regrets),
        "regret_std": np.std(regrets)
    }

def plot_results(all_results, output_dir, include_algorithms=None, font_size1=20, font_size2=15):
    """Generate plots comparing performance across different sample sizes."""
    # Set default algorithms to include if not specified
    if include_algorithms is None:
        include_algorithms = ["log_loss", "ortho", "nonortho"]
    
    # Organize results by algorithm and sample size
    log_loss_results = {}
    ortho_results = {}
    nonortho_results = {}
    crossfit_results = {}
    
    for n, results in all_results.items():
        n_val = int(n)  # Convert string key to integer
        
        # Get results for log_loss
        log_loss = [r for r in results if "log_loss" in r["script"] and r["success"] and r["policy_value"] is not None]
        log_loss_stats = calculate_stats(log_loss)
        if log_loss_stats:
            log_loss_results[n_val] = log_loss_stats
        
        # Get results for orthogonal - must contain "orthogonal" but NOT "nonorthogonal"
        ortho = [r for r in results if "orthogonal" in r["script"] and "nonorthogonal" not in r["script"] 
                and "crossfit" not in r["script"] and r["success"] and r["policy_value"] is not None]
        ortho_stats = calculate_stats(ortho)
        if ortho_stats:
            ortho_results[n_val] = ortho_stats
            
        # Get results for nonorthogonal
        nonortho = [r for r in results if "nonorthogonal" in r["script"] and r["success"] and r["policy_value"] is not None]
        nonortho_stats = calculate_stats(nonortho)
        if nonortho_stats:
            nonortho_results[n_val] = nonortho_stats
            
        # Get results for crossfit
        crossfit = [r for r in results if "crossfit" in r["script"] and r["success"] and r["policy_value"] is not None]
        crossfit_stats = calculate_stats(crossfit)
        if crossfit_stats:
            crossfit_results[n_val] = crossfit_stats
    
    # Sort by number of samples and skip the first two data points
    sample_sizes = sorted(log_loss_results.keys())
    if len(sample_sizes) > 2:
        sample_sizes = sample_sizes[2:]  # Skip the first two data points
        print(f"Skipping the first two data points. Using sample sizes: {sample_sizes}")
    else:
        print("Warning: Not enough data points to skip the first two. Using all available data.")
    
    # Get the common sample sizes across all required algorithms
    common_sizes = set(sample_sizes)
    
    # Only include common sizes for the algorithms we need
    if "log_loss" in include_algorithms:
        common_sizes &= set(log_loss_results.keys())
    if "ortho" in include_algorithms:
        common_sizes &= set(ortho_results.keys())
    if "nonortho" in include_algorithms:
        common_sizes &= set(nonortho_results.keys())
    if "crossfit" in include_algorithms:
        common_sizes &= set(crossfit_results.keys())
    
    sample_sizes = sorted(list(common_sizes))
    
    print(f"Using sample sizes present in all algorithm results: {sample_sizes}")
    
    # Check if we need to include each algorithm
    has_log_loss = "log_loss" in include_algorithms and log_loss_results
    has_ortho = "ortho" in include_algorithms and ortho_results
    has_nonortho = "nonortho" in include_algorithms and nonortho_results
    has_crossfit = "crossfit" in include_algorithms and crossfit_results
    
    # Convert to arrays for plotting for the included algorithms
    if has_log_loss:
        log_policy_means = np.array([log_loss_results[n]["policy_value_mean"] for n in sample_sizes])
        log_policy_stds = np.array([log_loss_results[n]["policy_value_std"] for n in sample_sizes])
        log_mse_means = np.array([log_loss_results[n]["mse_mean"] for n in sample_sizes])
        log_mse_stds = np.array([log_loss_results[n]["mse_std"] for n in sample_sizes])
        log_regret_means = np.array([log_loss_results[n]["regret_mean"] for n in sample_sizes])
        log_regret_stds = np.array([log_loss_results[n]["regret_std"] for n in sample_sizes])
        scaled_log_regret_means = log_regret_means * 10000
        scaled_log_regret_stds = log_regret_stds * 10000
    
    if has_ortho:
        ortho_policy_means = np.array([ortho_results[n]["policy_value_mean"] for n in sample_sizes])
        ortho_policy_stds = np.array([ortho_results[n]["policy_value_std"] for n in sample_sizes])
        ortho_mse_means = np.array([ortho_results[n]["mse_mean"] for n in sample_sizes])
        ortho_mse_stds = np.array([ortho_results[n]["mse_std"] for n in sample_sizes])
        ortho_regret_means = np.array([ortho_results[n]["regret_mean"] for n in sample_sizes])
        ortho_regret_stds = np.array([ortho_results[n]["regret_std"] for n in sample_sizes])
        scaled_ortho_regret_means = ortho_regret_means * 10000
        scaled_ortho_regret_stds = ortho_regret_stds * 10000
    
    # Prepare nonorthogonal data if requested and available
    if has_nonortho:
        nonortho_policy_means = np.array([nonortho_results[n]["policy_value_mean"] for n in sample_sizes])
        nonortho_policy_stds = np.array([nonortho_results[n]["policy_value_std"] for n in sample_sizes])
        nonortho_mse_means = np.array([nonortho_results[n]["mse_mean"] for n in sample_sizes])
        nonortho_mse_stds = np.array([nonortho_results[n]["mse_std"] for n in sample_sizes])
        nonortho_regret_means = np.array([nonortho_results[n]["regret_mean"] for n in sample_sizes])
        nonortho_regret_stds = np.array([nonortho_results[n]["regret_std"] for n in sample_sizes])
        scaled_nonortho_regret_means = nonortho_regret_means * 10000
        scaled_nonortho_regret_stds = nonortho_regret_stds * 10000
    
    # Prepare crossfit data if requested and available
    if has_crossfit:
        crossfit_policy_means = np.array([crossfit_results[n]["policy_value_mean"] for n in sample_sizes])
        crossfit_policy_stds = np.array([crossfit_results[n]["policy_value_std"] for n in sample_sizes])
        crossfit_mse_means = np.array([crossfit_results[n]["mse_mean"] for n in sample_sizes])
        crossfit_mse_stds = np.array([crossfit_results[n]["mse_std"] for n in sample_sizes])
        crossfit_regret_means = np.array([crossfit_results[n]["regret_mean"] for n in sample_sizes])
        crossfit_regret_stds = np.array([crossfit_results[n]["regret_std"] for n in sample_sizes])
        scaled_crossfit_regret_means = crossfit_regret_means * 10000
        scaled_crossfit_regret_stds = crossfit_regret_stds * 10000
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot 1: Policy Value
    plt.figure(figsize=(10, 7))
    
    # Plot mean lines for included algorithms
    if has_log_loss:
        plt.plot(sample_sizes, log_policy_means, 'b-', marker='o', markersize=6, linewidth=2, label="Log Loss")
    if has_ortho:
        plt.plot(sample_sizes, ortho_policy_means, 'r-', marker='s', markersize=6, linewidth=2, label="Ortho Loss")
    if has_nonortho:
        plt.plot(sample_sizes, nonortho_policy_means, 'g-', marker='^', markersize=6, linewidth=2, label="Non-ortho Loss")
    if has_crossfit:
        plt.plot(sample_sizes, crossfit_policy_means, 'm-', marker='d', markersize=6, linewidth=2, label="Crossfit")
    
    # Add confidence bands for included algorithms
    if has_log_loss:
        plt.fill_between(sample_sizes, 
                        log_policy_means - log_policy_stds, 
                        log_policy_means + log_policy_stds, 
                        color='blue', alpha=0.2)
    if has_ortho:
        plt.fill_between(sample_sizes, 
                        ortho_policy_means - ortho_policy_stds, 
                        ortho_policy_means + ortho_policy_stds, 
                        color='red', alpha=0.2)
    if has_nonortho:
        plt.fill_between(sample_sizes, 
                        nonortho_policy_means - nonortho_policy_stds, 
                        nonortho_policy_means + nonortho_policy_stds, 
                        color='green', alpha=0.2)
    if has_crossfit:
        plt.fill_between(sample_sizes,
                        crossfit_policy_means - crossfit_policy_stds,
                        crossfit_policy_means + crossfit_policy_stds,
                        color='magenta', alpha=0.2)
    
    # Format the plot
    plt.xlabel("Number of training samples", fontsize=font_size1)
    plt.ylabel("Policy Value", fontsize=font_size1)
    plt.title("Policy Value vs. Training Size", fontsize=font_size1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=font_size2)
    plt.xticks(fontsize=font_size2)
    plt.yticks(fontsize=font_size2)
    
    # Set reasonable y-axis limits with some padding
    y_values_max = []
    y_values_min = []
    
    if has_log_loss:
        y_values_max.append(np.max(log_policy_means + log_policy_stds))
        y_values_min.append(np.min(log_policy_means - log_policy_stds))
    if has_ortho:
        y_values_max.append(np.max(ortho_policy_means + ortho_policy_stds))
        y_values_min.append(np.min(ortho_policy_means - ortho_policy_stds))
    if has_nonortho:
        y_values_max.append(np.max(nonortho_policy_means + nonortho_policy_stds))
        y_values_min.append(np.min(nonortho_policy_means - nonortho_policy_stds))
    if has_crossfit:
        y_values_max.append(np.max(crossfit_policy_means + crossfit_policy_stds))
        y_values_min.append(np.min(crossfit_policy_means - crossfit_policy_stds))
    
    y_max = max(y_values_max)
    y_min = min(y_values_min)
    
    padding = (y_max - y_min) * 0.05
    plt.ylim(max(0, y_min - padding), y_max + padding)
    
    plt.tight_layout()
    
    # Save policy value figure
    policy_filename = os.path.join(output_dir, f"policy_value_comparison_{timestamp}.png")
    plt.savefig(policy_filename, dpi=300, bbox_inches='tight')
    print(f"Policy value plot saved to {policy_filename}")
    
    # Plot 2: MSE
    plt.figure(figsize=(10, 7))
    
    # Plot mean lines for included algorithms
    if has_log_loss:
        plt.plot(sample_sizes, log_mse_means, 'b-', marker='o', markersize=6, linewidth=2, label="Log Loss")
    if has_ortho:
        plt.plot(sample_sizes, ortho_mse_means, 'r-', marker='s', markersize=6, linewidth=2, label="Ortho Loss")
    if has_nonortho:
        plt.plot(sample_sizes, nonortho_mse_means, 'g-', marker='^', markersize=6, linewidth=2, label="Non-ortho Loss")
    if has_crossfit:
        plt.plot(sample_sizes, crossfit_mse_means, 'm-', marker='d', markersize=6, linewidth=2, label="Crossfit")
    
    # Add confidence bands for included algorithms
    if has_log_loss:
        plt.fill_between(sample_sizes, 
                        log_mse_means - log_mse_stds, 
                        log_mse_means + log_mse_stds, 
                        color='blue', alpha=0.2)
    if has_ortho:
        plt.fill_between(sample_sizes, 
                        ortho_mse_means - ortho_mse_stds, 
                        ortho_mse_means + ortho_mse_stds, 
                        color='red', alpha=0.2)
    if has_nonortho:
        plt.fill_between(sample_sizes, 
                        nonortho_mse_means - nonortho_mse_stds, 
                        nonortho_mse_means + nonortho_mse_stds, 
                        color='green', alpha=0.2)
    if has_crossfit:
        plt.fill_between(sample_sizes,
                        crossfit_mse_means - crossfit_mse_stds,
                        crossfit_mse_means + crossfit_mse_stds,
                        color='magenta', alpha=0.2)
    
    # Format the plot
    plt.xlabel("Number of training samples", fontsize=font_size1)
    plt.ylabel("Mean Squared Error", fontsize=font_size1)
    plt.title("MSE vs. Training Size", fontsize=font_size1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=font_size2)
    plt.xticks(fontsize=font_size2)
    plt.yticks(fontsize=font_size2)
    
    # Set reasonable y-axis limits with some padding
    y_values_max = []
    y_values_min = []
    
    if has_log_loss:
        y_values_max.append(np.max(log_mse_means + log_mse_stds))
        y_values_min.append(np.min(log_mse_means - log_mse_stds))
    if has_ortho:
        y_values_max.append(np.max(ortho_mse_means + ortho_mse_stds))
        y_values_min.append(np.min(ortho_mse_means - ortho_mse_stds))
    if has_nonortho:
        y_values_max.append(np.max(nonortho_mse_means + nonortho_mse_stds))
        y_values_min.append(np.min(nonortho_mse_means - nonortho_mse_stds))
    if has_crossfit:
        y_values_max.append(np.max(crossfit_mse_means + crossfit_mse_stds))
        y_values_min.append(np.min(crossfit_mse_means - crossfit_mse_stds))
    
    y_max = max(y_values_max)
    y_min = min(y_values_min)
    
    padding = (y_max - y_min) * 0.05
    plt.ylim(max(0, y_min - padding), y_max + padding)
    
    plt.tight_layout()
    
    # Save MSE figure
    mse_filename = os.path.join(output_dir, f"mse_comparison_{timestamp}.png")
    plt.savefig(mse_filename, dpi=300, bbox_inches='tight')
    print(f"MSE plot saved to {mse_filename}")
    
    # Plot 3: Regret
    plt.figure(figsize=(10, 7))
    
    # Plot mean lines for included algorithms
    if has_log_loss:
        plt.plot(sample_sizes, scaled_log_regret_means, 'b-', marker='o', markersize=6, linewidth=2, label="Log Loss")
    if has_ortho:
        plt.plot(sample_sizes, scaled_ortho_regret_means, 'r-', marker='s', markersize=6, linewidth=2, label="Ortho Loss")
    if has_nonortho:
        plt.plot(sample_sizes, scaled_nonortho_regret_means, 'g-', marker='^', markersize=6, linewidth=2, label="Non-ortho Loss")
    if has_crossfit:
        plt.plot(sample_sizes, scaled_crossfit_regret_means, 'm-', marker='d', markersize=6, linewidth=2, label="Crossfit")
    
    # Add confidence bands for included algorithms
    if has_log_loss:
        plt.fill_between(sample_sizes, 
                        scaled_log_regret_means - scaled_log_regret_stds, 
                        scaled_log_regret_means + scaled_log_regret_stds, 
                        color='blue', alpha=0.2)
    if has_ortho:
        plt.fill_between(sample_sizes, 
                        scaled_ortho_regret_means - scaled_ortho_regret_stds, 
                        scaled_ortho_regret_means + scaled_ortho_regret_stds, 
                        color='red', alpha=0.2)
    if has_nonortho:
        plt.fill_between(sample_sizes, 
                        scaled_nonortho_regret_means - scaled_nonortho_regret_stds, 
                        scaled_nonortho_regret_means + scaled_nonortho_regret_stds, 
                        color='green', alpha=0.2)
    if has_crossfit:
        plt.fill_between(sample_sizes,
                        scaled_crossfit_regret_means - scaled_crossfit_regret_stds,
                        scaled_crossfit_regret_means + scaled_crossfit_regret_stds,
                        color='magenta', alpha=0.2)
    
    # Format the plot
    plt.xlabel("Number of training samples", fontsize=font_size1)
    plt.ylabel("Regret", fontsize=font_size1)
    plt.title("Regret vs. Training Size", fontsize=font_size1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=font_size2)
    plt.xticks(fontsize=font_size2)
    plt.yticks(fontsize=font_size2)
    
    # Set reasonable y-axis limits with some padding
    y_values_max = []
    y_values_min = []
    
    if has_log_loss:
        y_values_max.append(np.max(scaled_log_regret_means + scaled_log_regret_stds))
        y_values_min.append(np.min(scaled_log_regret_means - scaled_log_regret_stds))
    if has_ortho:
        y_values_max.append(np.max(scaled_ortho_regret_means + scaled_ortho_regret_stds))
        y_values_min.append(np.min(scaled_ortho_regret_means - scaled_ortho_regret_stds))
    if has_nonortho:
        y_values_max.append(np.max(scaled_nonortho_regret_means + scaled_nonortho_regret_stds))
        y_values_min.append(np.min(scaled_nonortho_regret_means - scaled_nonortho_regret_stds))
    if has_crossfit:
        y_values_max.append(np.max(scaled_crossfit_regret_means + scaled_crossfit_regret_stds))
        y_values_min.append(np.min(scaled_crossfit_regret_means - scaled_crossfit_regret_stds))
    
    y_max = max(y_values_max)
    y_min = min(y_values_min)
    
    padding = (y_max - y_min) * 0.05
    plt.ylim(max(0, y_min - padding), y_max + padding)
    
    plt.tight_layout()
    
    # Save Regret figure
    regret_filename = os.path.join(output_dir, f"regret_comparison_{timestamp}.png")
    plt.savefig(regret_filename, dpi=300, bbox_inches='tight')
    print(f"Regret plot saved to {regret_filename}")

    # Plot 4: Scaling comparison
    plt.figure(figsize=(10, 7))
    
    if has_log_loss:
        plt.plot(sample_sizes, log_policy_means, 'b-', marker='o', linewidth=2, label="Log Loss - Policy Value")
    if has_ortho:
        plt.plot(sample_sizes, ortho_policy_means, 'r-', marker='s', linewidth=2, label="Orthogonal - Policy Value")
    if has_nonortho:
        plt.plot(sample_sizes, nonortho_policy_means, 'g-', marker='^', linewidth=2, label="Nonorthogonal - Policy Value")
    if has_crossfit:
        plt.plot(sample_sizes, crossfit_policy_means, 'm-', marker='d', linewidth=2, label="Crossfit - Policy Value")
    
    plt.xlabel("Number of training samples", fontsize=font_size1)
    plt.ylabel("Policy Value", fontsize=font_size1)
    plt.title("Scaling Comparison", fontsize=font_size1)
    plt.grid(True, alpha=0.2)
    plt.legend(loc='best', fontsize=font_size2)
    plt.xticks(fontsize=font_size2)
    plt.yticks(fontsize=font_size2)
    
    # Save scaling figure
    scaling_filename = os.path.join(output_dir, f"scaling_comparison_{timestamp}.png")
    plt.savefig(scaling_filename, dpi=300, bbox_inches='tight')
    print(f"Scaling plot saved to {scaling_filename}")

    return policy_filename, mse_filename, regret_filename, scaling_filename

def load_results_from_directory(results_dir):
    """Load results from all JSON files in the given directory."""
    results_by_n = {}
    
    # Load all result files
    json_files = glob.glob(os.path.join(results_dir, "results_N*.json"))
    
    if not json_files:
        print(f"No result files found in {results_dir}")
        return None
    
    print(f"Found {len(json_files)} result files")
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                results = json.load(f)
            
            # Extract N value from filename
            file_name = os.path.basename(file_path)
            n_value = None
            if file_name.startswith("results_N"):
                n_start = 9  # Length of "results_N"
                n_end = file_name.find("_", n_start)
                if n_end > n_start:
                    n_value = file_name[n_start:n_end]
            
            if n_value is None:
                # Try to get N from the results themselves
                if results and isinstance(results, list) and len(results) > 0 and "num_samples" in results[0]:
                    n_value = str(results[0]["num_samples"])
                else:
                    print(f"Warning: Could not determine N value for {file_path}, skipping")
                    continue
            
            results_by_n[n_value] = results
            print(f"Loaded results for N={n_value} from {file_path}")
            
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    return results_by_n

def main():
    parser = argparse.ArgumentParser(description="Plot preference learning results")
    parser.add_argument("results_dir", nargs="?", default="scaling_results", 
                        help="Directory containing result JSON files (default: scaling_results)")
    parser.add_argument("--log_loss", action="store_true", help="Include Log Loss algorithm")
    parser.add_argument("--ortho", action="store_true", help="Include Orthogonal algorithm")
    parser.add_argument("--nonortho", action="store_true", help="Include Nonorthogonal algorithm")
    parser.add_argument("--crossfit", action="store_true", help="Include Crossfit algorithm")
    parser.add_argument("--all", action="store_true", help="Include all algorithms")
    parser.add_argument("--font-size1", type=int, default=20, help="Font size for titles and labels")
    parser.add_argument("--font-size2", type=int, default=15, help="Font size for ticks and legend")
    
    args = parser.parse_args()
    
    results_dir = args.results_dir
    
    if not os.path.exists(results_dir):
        print(f"Error: Directory {results_dir} does not exist")
        sys.exit(1)
    
    # Determine which algorithms to include
    include_algorithms = []
    
    # If specific algorithms are selected, use them
    if args.log_loss or args.ortho or args.nonortho or args.crossfit:
        if args.log_loss:
            include_algorithms.append("log_loss")
        if args.ortho:
            include_algorithms.append("ortho")
        if args.nonortho:
            include_algorithms.append("nonortho")
        if args.crossfit:
            include_algorithms.append("crossfit")
    # If --all is specified, include all algorithms
    elif args.all:
        include_algorithms = ["log_loss", "ortho", "nonortho", "crossfit"]
    # Otherwise, use default (log_loss, ortho, nonortho)
    else:
        include_algorithms = ["log_loss", "ortho", "nonortho"]
    
    print(f"Including algorithms: {include_algorithms}")
    
    # Load results
    print(f"Loading results from {results_dir}...")
    all_results = load_results_from_directory(results_dir)
    
    if not all_results:
        print("No results found. Please run the experiment first.")
        sys.exit(1)
    
    # Plot results
    print("Plotting results...")
    policy_filename, mse_filename, regret_filename, scaling_filename = plot_results(
        all_results, results_dir, include_algorithms, args.font_size1, args.font_size2
    )
    
    print(f"Plots saved to: \n- {policy_filename}\n- {mse_filename}\n- {regret_filename}\n- {scaling_filename}")
    print("Done!")

if __name__ == "__main__":
    main() 