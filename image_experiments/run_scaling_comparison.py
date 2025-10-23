import subprocess
import os
import sys
import json
import threading
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.patches as mpatches

def run_experiment(script_name, train_folder, seed, num_samples, gpu_id):
    """Run a single experiment on the specified GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    if script_name in ["log_loss_learner_large.py"]:
        cmd = [
            "python", script_name, 
            "--train_folder", train_folder,
            "--seed", str(seed),
            "--batch", "256",
            "--num_samples", str(num_samples),
            "--test_size", "10000",
            "--epochs", "15"
        ]
    else:  # log_loss_learner_large.py
        cmd = [
            "python", script_name, 
            "--train_folder", train_folder,
            "--seed", str(seed),
            "--batch", "256",
            "--num_samples", str(num_samples),
            "--test_size", "10000",
            "--epochs1", "5",
            "--epochs2", "10"
        ]
    
    print(f"\n--- Starting {script_name} with seed {seed}, N={num_samples} on GPU {gpu_id} ---")
    print(f"Command: {' '.join(cmd)}")
    
    # Run the process and let output flow to console in real time
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Real-time output handling
    output_lines = []
    for line in process.stdout:
        print(f"[GPU {gpu_id}] {line.strip()}")
        output_lines.append(line)
    
    # Wait for the process to complete
    process.wait()
    
    # Extract results
    accuracy = None
    mse = None
    policy_value = None
    regret = None
    for line in output_lines:
        if "accuracy:" in line.lower():
            try:
                accuracy = float(line.split(':')[-1].strip())
            except ValueError:
                pass
        elif "mse:" in line.lower():
            try:
                mse = float(line.split(':')[-1].strip())
            except ValueError:
                pass
        elif "policy value:" in line.lower():
            try:
                policy_value = float(line.split(':')[-1].strip())
            except ValueError:
                pass
        elif "total regret:" in line.lower():
            try:
                regret = float(line.split(':')[-1].strip())
            except ValueError:
                pass
    
    result = {
        "script": script_name,
        "seed": seed,
        "num_samples": num_samples,
        "gpu_id": gpu_id,
        "success": process.returncode == 0,
        "accuracy": accuracy,
        "mse": mse,
        "policy_value": policy_value,
        "regret": regret
    }
    
    if process.returncode != 0:
        print(f"\n--- Error running {script_name} with seed {seed}, N={num_samples} on GPU {gpu_id} ---")
    else:
        print(f"\n--- Completed {script_name} with seed {seed}, N={num_samples} on GPU {gpu_id} ---")
        if accuracy is None or mse is None or policy_value is None or regret is None:
            print("Warning: Could not extract all metrics (accuracy, MSE, policy value, regret) from output")
    
    return result

def run_experiments_for_n(train_folder, num_samples, seeds, available_gpus=4):
    """Run experiments for a specific number of training samples."""
    print(f"\n=== Running experiments for N={num_samples} ===")
    
    all_results = []
    
    # Create a list of all experiments to run
    experiments = []
    for seed in seeds:
        experiments.append(("log_loss_learner_large.py", seed))
        experiments.append(("orthogonal_loss_learner_large.py", seed))
        experiments.append(("nonorthogonal_loss_learner_large.py", seed))
        experiments.append(("orthogonal_loss_learner_crossfit.py", seed))
    
    # Run experiments in batches of available_gpus
    for i in range(0, len(experiments), available_gpus):
        batch = experiments[i:i+available_gpus]
        threads = []
        batch_results = [None] * len(batch)
        
        # Start each experiment in its own thread
        for j, (script, seed) in enumerate(batch):
            gpu_id = j % available_gpus
            thread = threading.Thread(
                target=lambda idx, s, sd, n, gid: batch_results.__setitem__(
                    idx, run_experiment(s, train_folder, sd, n, gid)
                ),
                args=(j, script, seed, num_samples, gpu_id)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Collect results
        all_results.extend([r for r in batch_results if r is not None])
    
    return all_results

def calculate_stats(results):
    """Calculate mean and std for accuracy, MSE, policy value, and regret."""
    if not results:
        return None
    
    accuracies = [r["accuracy"] for r in results if r["accuracy"] is not None]
    mses = [r["mse"] for r in results if r["mse"] is not None]
    policy_values = [r["policy_value"] for r in results if r["policy_value"] is not None]
    regrets = [r["regret"] for r in results if r["regret"] is not None]
    
    if not accuracies or not mses or not policy_values or not regrets:
        return None
    
    return {
        "accuracy_mean": np.mean(accuracies),
        "accuracy_std": np.std(accuracies),
        "mse_mean": np.mean(mses),
        "mse_std": np.std(mses),
        "policy_value_mean": np.mean(policy_values),
        "policy_value_std": np.std(policy_values),
        "regret_mean": np.mean(regrets),
        "regret_std": np.std(regrets)
    }

def plot_results(all_results, output_dir):
    """Generate plots comparing performance across different sample sizes."""
    # Organize results by algorithm and sample size
    log_loss_results = {}
    ortho_results = {}
    nonortho_results = {}
    
    for n, results in all_results.items():
        # Get results for log_loss
        log_loss = [r for r in results if "log_loss" in r["script"] and r["success"] and r["policy_value"] is not None]
        log_loss_stats = calculate_stats(log_loss)
        if log_loss_stats:
            log_loss_results[n] = log_loss_stats
        
        # Get results for orthogonal (must contain "orthogonal" but NOT "nonorthogonal")
        ortho = [r for r in results if "orthogonal" in r["script"] and "nonorthogonal" not in r["script"] 
                and r["success"] and r["policy_value"] is not None]
        ortho_stats = calculate_stats(ortho)
        if ortho_stats:
            ortho_results[n] = ortho_stats
            
        # Get results for nonorthogonal
        nonortho = [r for r in results if "nonorthogonal" in r["script"] and r["success"] and r["policy_value"] is not None]
        nonortho_stats = calculate_stats(nonortho)
        if nonortho_stats:
            nonortho_results[n] = nonortho_stats
    
    # Sort by number of samples
    sample_sizes = sorted(log_loss_results.keys())
    
    # Get the common sample sizes across all algorithms
    common_sizes = set(sample_sizes)
    common_sizes &= set(ortho_results.keys())
    if nonortho_results:
        common_sizes &= set(nonortho_results.keys())
    sample_sizes = sorted(list(common_sizes))
    
    print(f"Using sample sizes present in all algorithm results: {sample_sizes}")
    
    # Convert to arrays for plotting
    log_policy_means = np.array([log_loss_results[n]["policy_value_mean"] for n in sample_sizes])
    log_policy_stds = np.array([log_loss_results[n]["policy_value_std"] for n in sample_sizes])
    ortho_policy_means = np.array([ortho_results[n]["policy_value_mean"] for n in sample_sizes])
    ortho_policy_stds = np.array([ortho_results[n]["policy_value_std"] for n in sample_sizes])
    
    log_mse_means = np.array([log_loss_results[n]["mse_mean"] for n in sample_sizes])
    log_mse_stds = np.array([log_loss_results[n]["mse_std"] for n in sample_sizes])
    ortho_mse_means = np.array([ortho_results[n]["mse_mean"] for n in sample_sizes])
    ortho_mse_stds = np.array([ortho_results[n]["mse_std"] for n in sample_sizes])
    
    # Add regret data
    log_regret_means = np.array([log_loss_results[n]["regret_mean"] for n in sample_sizes])
    log_regret_stds = np.array([log_loss_results[n]["regret_std"] for n in sample_sizes])
    ortho_regret_means = np.array([ortho_results[n]["regret_mean"] for n in sample_sizes])
    ortho_regret_stds = np.array([ortho_results[n]["regret_std"] for n in sample_sizes])
    
    # Prepare nonorthogonal data if available
    has_nonortho = bool(nonortho_results) and any(n in nonortho_results for n in sample_sizes)
    if has_nonortho:
        nonortho_policy_means = np.array([nonortho_results[n]["policy_value_mean"] for n in sample_sizes])
        nonortho_policy_stds = np.array([nonortho_results[n]["policy_value_std"] for n in sample_sizes])
        nonortho_mse_means = np.array([nonortho_results[n]["mse_mean"] for n in sample_sizes])
        nonortho_mse_stds = np.array([nonortho_results[n]["mse_std"] for n in sample_sizes])
        nonortho_regret_means = np.array([nonortho_results[n]["regret_mean"] for n in sample_sizes])
        nonortho_regret_stds = np.array([nonortho_results[n]["regret_std"] for n in sample_sizes])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot 1: Policy Value
    plt.figure(figsize=(10, 6))
    
    # Plot mean lines
    plt.plot(sample_sizes, log_policy_means, 'b-', marker='o', markersize=6, linewidth=2, label="Log Loss")
    plt.plot(sample_sizes, ortho_policy_means, 'r-', marker='s', markersize=6, linewidth=2, label="Orthogonal")
    if has_nonortho:
        plt.plot(sample_sizes, nonortho_policy_means, 'g-', marker='^', markersize=6, linewidth=2, label="Nonorthogonal")
    
    # Add confidence bands
    plt.fill_between(sample_sizes, 
                    log_policy_means - log_policy_stds, 
                    log_policy_means + log_policy_stds, 
                    color='blue', alpha=0.2)
    plt.fill_between(sample_sizes, 
                    ortho_policy_means - ortho_policy_stds, 
                    ortho_policy_means + ortho_policy_stds, 
                    color='red', alpha=0.2)
    if has_nonortho:
        plt.fill_between(sample_sizes, 
                        nonortho_policy_means - nonortho_policy_stds, 
                        nonortho_policy_means + nonortho_policy_stds, 
                        color='green', alpha=0.2)
    
    # Format the plot
    plt.xlabel("Number of training samples", fontsize=12)
    plt.ylabel("Policy Value", fontsize=12)
    plt.title("Policy Value vs. Training Size", fontsize=14)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=12)
    
    # Format x-axis to show actual numbers instead of powers
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ScalarFormatter())
    plt.xticks(rotation=45)
    
    # Set reasonable y-axis limits with some padding
    if has_nonortho:
        y_max = max(np.max(log_policy_means + log_policy_stds), 
                   np.max(ortho_policy_means + ortho_policy_stds),
                   np.max(nonortho_policy_means + nonortho_policy_stds))
        y_min = min(np.min(log_policy_means - log_policy_stds), 
                   np.min(ortho_policy_means - ortho_policy_stds),
                   np.min(nonortho_policy_means - nonortho_policy_stds))
    else:
        y_max = max(np.max(log_policy_means + log_policy_stds), np.max(ortho_policy_means + ortho_policy_stds))
        y_min = min(np.min(log_policy_means - log_policy_stds), np.min(ortho_policy_means - ortho_policy_stds))
    
    plt.ylim(max(0, y_min - 0.02), min(y_max * 1.1, y_max + 0.5))
    
    plt.tight_layout()
    
    # Save policy value figure
    policy_filename = os.path.join(output_dir, f"policy_value_comparison_{timestamp}.png")
    plt.savefig(policy_filename, dpi=300, bbox_inches='tight')
    print(f"Policy value plot saved to {policy_filename}")
    
    # Plot 2: MSE
    plt.figure(figsize=(10, 6))
    
    # Plot mean lines
    plt.plot(sample_sizes, log_mse_means, 'b-', marker='o', markersize=6, linewidth=2, label="Log Loss")
    plt.plot(sample_sizes, ortho_mse_means, 'r-', marker='s', markersize=6, linewidth=2, label="Orthogonal")
    if has_nonortho:
        plt.plot(sample_sizes, nonortho_mse_means, 'g-', marker='^', markersize=6, linewidth=2, label="Nonorthogonal")
    
    # Add confidence bands
    plt.fill_between(sample_sizes, 
                    log_mse_means - log_mse_stds, 
                    log_mse_means + log_mse_stds, 
                    color='blue', alpha=0.2)
    plt.fill_between(sample_sizes, 
                    ortho_mse_means - ortho_mse_stds, 
                    ortho_mse_means + ortho_mse_stds, 
                    color='red', alpha=0.2)
    if has_nonortho:
        plt.fill_between(sample_sizes, 
                        nonortho_mse_means - nonortho_mse_stds, 
                        nonortho_mse_means + nonortho_mse_stds, 
                        color='green', alpha=0.2)
    
    # Format the plot
    plt.xlabel("Number of training samples", fontsize=12)
    plt.ylabel("Mean Squared Error", fontsize=12)
    plt.title("MSE vs. Training Size", fontsize=14)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=12)
    
    # Format x-axis to show actual numbers instead of powers
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ScalarFormatter())
    plt.xticks(rotation=45)
    
    # Set reasonable y-axis limits with some padding
    if has_nonortho:
        y_max = max(np.max(log_mse_means + log_mse_stds), 
                   np.max(ortho_mse_means + ortho_mse_stds),
                   np.max(nonortho_mse_means + nonortho_mse_stds))
        y_min = min(np.min(log_mse_means - log_mse_stds), 
                   np.min(ortho_mse_means - ortho_mse_stds),
                   np.min(nonortho_mse_means - nonortho_mse_stds))
    else:
        y_max = max(np.max(log_mse_means + log_mse_stds), np.max(ortho_mse_means + ortho_mse_stds))
        y_min = min(np.min(log_mse_means - log_mse_stds), np.min(ortho_mse_means - ortho_mse_stds))
    
    plt.ylim(max(0, y_min - 0.02), min(y_max * 1.1, y_max + 0.5))
    
    plt.tight_layout()
    
    # Save MSE figure
    mse_filename = os.path.join(output_dir, f"mse_comparison_{timestamp}.png")
    plt.savefig(mse_filename, dpi=300, bbox_inches='tight')
    print(f"MSE plot saved to {mse_filename}")
    
    # Plot 3: Regret
    plt.figure(figsize=(10, 6))
    
    # Plot mean lines
    plt.plot(sample_sizes, log_regret_means, 'b-', marker='o', markersize=6, linewidth=2, label="Log Loss")
    plt.plot(sample_sizes, ortho_regret_means, 'r-', marker='s', markersize=6, linewidth=2, label="Orthogonal")
    if has_nonortho:
        plt.plot(sample_sizes, nonortho_regret_means, 'g-', marker='^', markersize=6, linewidth=2, label="Nonorthogonal")
    
    # Add confidence bands
    plt.fill_between(sample_sizes, 
                    log_regret_means - log_regret_stds, 
                    log_regret_means + log_regret_stds, 
                    color='blue', alpha=0.2)
    plt.fill_between(sample_sizes, 
                    ortho_regret_means - ortho_regret_stds, 
                    ortho_regret_means + ortho_regret_stds, 
                    color='red', alpha=0.2)
    if has_nonortho:
        plt.fill_between(sample_sizes, 
                        nonortho_regret_means - nonortho_regret_stds, 
                        nonortho_regret_means + nonortho_regret_stds, 
                        color='green', alpha=0.2)
    
    # Format the plot
    plt.xlabel("Number of training samples", fontsize=12)
    plt.ylabel("Total Regret", fontsize=12)
    plt.title("Regret vs. Training Size", fontsize=14)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=12)
    
    # Format x-axis to show actual numbers instead of powers
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ScalarFormatter())
    plt.xticks(rotation=45)
    
    # Set reasonable y-axis limits with some padding
    if has_nonortho:
        y_max = max(np.max(log_regret_means + log_regret_stds), 
                   np.max(ortho_regret_means + ortho_regret_stds),
                   np.max(nonortho_regret_means + nonortho_regret_stds))
        y_min = min(np.min(log_regret_means - log_regret_stds), 
                   np.min(ortho_regret_means - ortho_regret_stds),
                   np.min(nonortho_regret_means - nonortho_regret_stds))
    else:
        y_max = max(np.max(log_regret_means + log_regret_stds), np.max(ortho_regret_means + ortho_regret_stds))
        y_min = min(np.min(log_regret_means - log_regret_stds), np.min(ortho_regret_means - ortho_regret_stds))
    
    plt.ylim(max(0, y_min - 0.02), min(y_max * 1.1, y_max + 0.5))
    
    plt.tight_layout()
    
    # Save regret figure
    regret_filename = os.path.join(output_dir, f"regret_comparison_{timestamp}.png")
    plt.savefig(regret_filename, dpi=300, bbox_inches='tight')
    print(f"Regret plot saved to {regret_filename}")
    
    # Save a combined scaling plot as well
    plt.figure(figsize=(12, 6))
    plt.plot(sample_sizes, log_policy_means, 'b-', marker='o', linewidth=2, label="Log Loss - Policy Value")
    plt.plot(sample_sizes, ortho_policy_means, 'r-', marker='s', linewidth=2, label="Orthogonal - Policy Value")
    if has_nonortho:
        plt.plot(sample_sizes, nonortho_policy_means, 'g-', marker='^', linewidth=2, label="Nonorthogonal - Policy Value")
    plt.xscale('log')
    plt.xlabel("Number of training samples", fontsize=12)
    plt.ylabel("Policy Value", fontsize=12)
    plt.title("Scaling Comparison", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ScalarFormatter())
    plt.xticks(rotation=45)
    
    # Save scaling figure
    scaling_filename = os.path.join(output_dir, f"scaling_comparison_{timestamp}.png")
    plt.savefig(scaling_filename, dpi=300, bbox_inches='tight')
    print(f"Scaling plot saved to {scaling_filename}")
    
    return policy_filename, mse_filename, regret_filename, scaling_filename

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run comparison experiments with different numbers of training samples")
    parser.add_argument("--train_folder", default="data/training_chunks", help="Folder containing training data")
    parser.add_argument("--output_dir", default="scaling_results", help="Directory to save results")
    parser.add_argument("--gpus", type=int, default=4, help="Number of GPUs to use")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789], help="Random seeds to use")
    parser.add_argument("--skip_sizes", type=int, nargs="+", default=[], help="Sample sizes to skip")
    parser.add_argument("--sizes", type=int, nargs="+", help="Sample sizes to use (overrides default list)")
    parser.add_argument("--algorithms", type=str, nargs="+", default=["log_loss", "orthogonal", "orthogonal_crossfit", "nonorthogonal"], 
                        help="Algorithms to run: log_loss, orthogonal, orthogonal_crossfit, nonorthogonal")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define sample sizes to evaluate
    if args.sizes:
        sample_sizes = args.sizes
    else:
        sample_sizes = [ 20000, 30000, 40000, 50000, 100000, 200000, 300000, 400000,  500000]
        # Remove sample sizes that should be skipped
        sample_sizes = [n for n in sample_sizes if n not in args.skip_sizes]
    
    # Map algorithm names to script names
    algo_map = {
        "log_loss": "log_loss_learner_large.py",
        "orthogonal": "orthogonal_loss_learner_large.py",
        "orthogonal_crossfit": "orthogonal_loss_learner_crossfit.py",
        "nonorthogonal": "nonorthogonal_loss_learner_large.py"
    }
    
    print(f"Running experiments with algorithms: {args.algorithms}")
    print(f"Using sample sizes: {sample_sizes}")
    print(f"Seeds: {args.seeds}")
    print(f"Using {args.gpus} GPUs")
    
    all_results = {}
    all_results_flat = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run experiments for each sample size
    for n in sample_sizes:
        # Filter experiments to only include selected algorithms
        original_experiments = run_experiments_for_n(args.train_folder, n, args.seeds, args.gpus)
        filtered_experiments = [exp for exp in original_experiments 
                               if any(algo in exp["script"] for algo in [algo_map[a] for a in args.algorithms])]
        
        all_results[n] = filtered_experiments
        all_results_flat.extend(filtered_experiments)
        
        # Save results for this N
        results_file = os.path.join(args.output_dir, f"results_N{n}_{timestamp}.json")
        with open(results_file, "w") as f:
            json.dump(filtered_experiments, f, indent=2)
        print(f"Results for N={n} saved to {results_file}")
    
    # Save all results to a single file
    all_results_file = os.path.join(args.output_dir, f"all_results_{timestamp}.json")
    with open(all_results_file, "w") as f:
        json.dump(all_results_flat, f, indent=2)
    print(f"All results saved to {all_results_file}")
    
    # Plot results
    policy_file, mse_file, regret_file, scaling_file = plot_results(all_results, args.output_dir)
    print(f"Plots saved to {policy_file}, {mse_file}, {regret_file}, and {scaling_file}")

if __name__ == "__main__":
    main() 
