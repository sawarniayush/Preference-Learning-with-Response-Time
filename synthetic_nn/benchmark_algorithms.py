import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
from pathlib import Path
import seaborn as sns
from tqdm import tqdm
from itertools import product
import multiprocessing

# List of algorithms to benchmark
ALGORITHMS = [
    'log_loss_learner.py',
    'nonorthogonal_loss_learner.py',
    'orthogonal_loss_learner.py',
    'orthogonal_loss_learner_splitting.py',
    'orthogonal_loss_learner_ybyt.py'
]

# Training sizes to test
TRAIN_SIZES = [  1000, 2000, 3000, 5000, 7000, 9000, 10000, 15000, 20000]
TEST_SIZE = 3000
N_REPEATS = 1
N_ALGORITHM_RUNS = 4

# Calculate optimal number of processes
TOTAL_CORES = multiprocessing.cpu_count()
DATA_GEN_CORES = max(1, TOTAL_CORES // 4)  # Use 1/4 of cores for data generation
ALGORITHM_CORES = max(1, TOTAL_CORES - DATA_GEN_CORES)  # Rest for algorithms

def generate_data(train_size, test_size, seed):
    """Generate synthetic data for given sizes"""
    train_file = f'data/train_{train_size}_{seed}.pkl'
    test_file = f'data/test_{train_size}_{seed}.pkl'
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate training data with limited cores
    subprocess.run([
        'python', 'generate_synthetic_data.py',
        '--train_size', str(train_size),
        '--test_size', str(test_size),
        '--train-out', train_file,
        '--test-out', test_file,
        '--seed', str(seed),
        '--n-jobs', str(DATA_GEN_CORES)  # Limit cores for data generation
    ], check=True)
    
    return train_file, test_file

def run_algorithm(args):
    """Run a single algorithm and return its results"""
    algorithm, train_file, test_file, run_id = args
    try:
        result = subprocess.run([
            'python', algorithm,
            '--train', train_file,
            '--test', test_file
        ], capture_output=True, text=True, check=True)
        
        # Parse output to get accuracy, MSE, and regret
        output = result.stdout
        # Different algorithms might have different output formats
        accuracy = None
        mse = None
        regret = None
        
        if "Preference model" in output:
            # Format: "Preference model accuracy: X.XXXX\nPreference model MSE: X.XXXX\nPreference model regret: X.XXXX"
            accuracy = float(output.split('Preference model accuracy: ')[1].split('\n')[0])
            mse = float(output.split('Preference model MSE: ')[1].split('\n')[0])
            if 'Preference model regret:' in output:
                regret = float(output.split('Preference model regret: ')[1].split('\n')[0])
        elif "Two-stage model" in output:
            # Format: "Two-stage model (f) accuracy: X.XXXX\nTwo-stage model (f) mse: X.XXXX\nTwo-stage model (f) regret: X.XXXX"
            accuracy = float(output.split('Two-stage model (f) accuracy: ')[1].split('\n')[0])
            mse = float(output.split('Two-stage model (f) mse: ')[1].split('\n')[0])
            if 'Two-stage model (f) regret:' in output:
                regret = float(output.split('Two-stage model (f) regret: ')[1].split('\n')[0])
        else:
            # Try to find any line containing accuracy, mse, and regret
            for line in output.split('\n'):
                if 'accuracy' in line.lower():
                    accuracy = float(line.split(':')[1].strip())
                elif 'mse' in line.lower():
                    mse = float(line.split(':')[1].strip())
                elif 'regret' in line.lower():
                    regret = float(line.split(':')[1].strip())
            
            if accuracy is None or mse is None:
                raise ValueError(f"Could not parse accuracy or MSE from output: {output}")
        
        return {
            'algorithm': algorithm,
            'train_file': train_file,
            'test_file': test_file,
            'run_id': run_id,
            'accuracy': accuracy,
            'mse': mse,
            'regret': regret
        }
    except Exception as e:
        print(f"Error running {algorithm}: {str(e)}")
        print(f"Output was: {result.stdout if 'result' in locals() else 'No output'}")
        return None

def run_experiment(args):
    """Run a single experiment configuration"""
    train_size, seed = args
    results = []
    
    # Generate data
    train_file, test_file = generate_data(train_size, TEST_SIZE, seed)
    
    # Prepare all algorithm runs
    algorithm_runs = []
    for algorithm in ALGORITHMS:
        for run_id in range(N_ALGORITHM_RUNS):
            algorithm_runs.append((algorithm, train_file, test_file, run_id))
    
    # Run algorithms in parallel with limited cores
    with ProcessPoolExecutor(max_workers=ALGORITHM_CORES) as executor:
        futures = [executor.submit(run_algorithm, run_args) for run_args in algorithm_runs]
        for future in as_completed(futures):
            result = future.result()
            if result:
                result['train_size'] = train_size
                result['seed'] = seed
                results.append(result)
    
    return results

def plot_results(results_df, metric, output_file):
    """Create a plot for the given metric with confidence intervals"""
    plt.figure(figsize=(12, 6))

    grouped = results_df.groupby(['algorithm', 'train_size'])[metric].agg(['mean', 'std'])

    # Colors for different algorithms
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    markers = ['o', 's', '^', 'D', 'v', '>', '<']
    
    for i, algorithm in enumerate(sorted(results_df['algorithm'].unique())):
        color_idx = i % len(colors)
        marker_idx = i % len(markers)
        
        data = grouped.loc[algorithm]
        x = data.index
        y = data['mean']
        std = data['std']

        # Use line plot with shaded confidence band
        plt.plot(x, y, 
                 color=colors[color_idx], 
                 marker=markers[marker_idx], 
                 markersize=6, 
                 linewidth=2, 
                 label=algorithm.replace('.py', ''))
                 
        # Add shaded confidence band
        plt.fill_between(x, 
                        y - std, 
                        y + std, 
                        color=colors[color_idx], 
                        alpha=0.2)

    plt.xlabel('Training Size', fontsize=12)
    plt.ylabel(metric.capitalize(), fontsize=12)
    plt.title(f'{metric.capitalize()} vs Training Size', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def main():
    print(f"Total CPU cores: {TOTAL_CORES}")
    print(f"Cores for data generation: {DATA_GEN_CORES}")
    print(f"Cores for algorithm runs: {ALGORITHM_CORES}")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Prepare experiment configurations
    experiment_configs = [(size, seed) 
                         for size in TRAIN_SIZES 
                         for seed in range(N_REPEATS)]
    
    # Run experiments in parallel with limited cores
    all_results = []
    with ProcessPoolExecutor(max_workers=ALGORITHM_CORES) as executor:
        futures = [executor.submit(run_experiment, config) for config in experiment_configs]
        for future in tqdm(as_completed(futures), total=len(experiment_configs), desc="Running experiments"):
            results = future.result()
            all_results.extend(results)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save raw results
    results_df.to_csv('results/raw_results.csv', index=False)
    
    # Create plots
    plot_results(results_df, 'accuracy', 'results/accuracy_plot.png')
    plot_results(results_df, 'mse', 'results/mse_plot.png')
    plot_results(results_df, 'regret', 'results/regret_plot.png')
    
    # Calculate and save summary statistics
    summary = results_df.groupby(['algorithm', 'train_size']).agg({
        'accuracy': ['mean', 'std', 'min', 'max'],
        'mse': ['mean', 'std', 'min', 'max'],
        'regret': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    summary.to_csv('results/summary_statistics.csv')

if __name__ == '__main__':
    main() 