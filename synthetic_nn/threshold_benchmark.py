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
    'orthogonal_loss_learner_ybyt.py'
]

# Threshold values to test
THRESHOLD_VALUES = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5]
TRAIN_SIZE = 20000
TEST_SIZE = 2000
N_REPEATS = 5  # Number of times to repeat each combination

# Calculate optimal number of processes
TOTAL_CORES = multiprocessing.cpu_count()
DATA_GEN_CORES = max(1, TOTAL_CORES // 4)  # Use 1/4 of cores for data generation
ALGORITHM_CORES = max(1, TOTAL_CORES - DATA_GEN_CORES)  # Rest for algorithms

def generate_data(train_size, test_size, seed, threshold):
    """Generate synthetic data for given sizes and threshold"""
    train_file = f'data_threshold/train_{train_size}_{threshold}_{seed}.pkl'
    test_file = f'data_threshold/test_{train_size}_{threshold}_{seed}.pkl'
    
    # Create data directory if it doesn't exist
    os.makedirs('data_threshold', exist_ok=True)
    
    # Generate training data with limited cores
    subprocess.run([
        'python', 'generate_synthetic_data.py',
        '--train_size', str(train_size),
        '--test_size', str(test_size),
        '--train-out', train_file,
        '--test-out', test_file,
        '--seed', str(seed),
        '--threshold', str(threshold),
        '--n-jobs', str(DATA_GEN_CORES)  # Limit cores for data generation
    ], check=True)
    
    return train_file, test_file

def run_algorithm(args):
    """Run a single algorithm and return its results"""
    algorithm, train_file, test_file, threshold, run_id = args
    try:
        result = subprocess.run([
            'python', algorithm,
            '--train', train_file,
            '--test', test_file,
            '--threshold', str(threshold)
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
            'threshold': threshold,
            'run_id': run_id,
            'accuracy': accuracy,
            'mse': mse,
            'regret': regret
        }
    except Exception as e:
        print(f"Error running {algorithm} with threshold {threshold}: {str(e)}")
        print(f"Output was: {result.stdout if 'result' in locals() else 'No output'}")
        return None

def run_experiment(args):
    """Run a single experiment configuration"""
    threshold, seed = args
    results = []
    
    # Generate data
    train_file, test_file = generate_data(TRAIN_SIZE, TEST_SIZE, seed, threshold)
    
    # Prepare all algorithm runs
    algorithm_runs = []
    for algorithm in ALGORITHMS:
        for run_id in range(N_REPEATS):
            algorithm_runs.append((algorithm, train_file, test_file, threshold, run_id))
    
    # Run algorithms in parallel with limited cores
    with ProcessPoolExecutor(max_workers=ALGORITHM_CORES) as executor:
        futures = [executor.submit(run_algorithm, run_args) for run_args in algorithm_runs]
        for future in as_completed(futures):
            result = future.result()
            if result:
                result['threshold'] = threshold
                result['seed'] = seed
                results.append(result)
    
    return results

def generate_summary_statistics(results_df):
    """Generate summary statistics for each algorithm and threshold value"""
    # Create a directory for results
    os.makedirs('threshold_results', exist_ok=True)
    
    # Save raw results
    results_df.to_csv('threshold_results/raw_results.csv', index=False)
    
    # Calculate and save summary statistics
    summary = results_df.groupby(['algorithm', 'threshold']).agg({
        'accuracy': ['mean', 'std', 'min', 'max'],
        'mse': ['mean', 'std', 'min', 'max'],
        'regret': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    summary.to_csv('threshold_results/summary_statistics.csv')
    
    # Also create separate CSV files for MSE and regret that are easier to read and import into LaTeX
    # Format: rows are threshold values, columns are algorithms
    
    # For MSE
    mse_table = pd.pivot_table(
        results_df, 
        values='mse', 
        index=['threshold'], 
        columns=['algorithm'], 
        aggfunc=np.mean
    ).round(4)
    mse_table.to_csv('threshold_results/mse_table.csv')
    
    # For regret
    regret_table = pd.pivot_table(
        results_df, 
        values='regret', 
        index=['threshold'], 
        columns=['algorithm'], 
        aggfunc=np.mean
    ).round(4)
    regret_table.to_csv('threshold_results/regret_table.csv')
    
    # Also include standard deviations
    mse_std_table = pd.pivot_table(
        results_df, 
        values='mse', 
        index=['threshold'], 
        columns=['algorithm'], 
        aggfunc=np.std
    ).round(4)
    mse_std_table.to_csv('threshold_results/mse_std_table.csv')
    
    regret_std_table = pd.pivot_table(
        results_df, 
        values='regret', 
        index=['threshold'], 
        columns=['algorithm'], 
        aggfunc=np.std
    ).round(4)
    regret_std_table.to_csv('threshold_results/regret_std_table.csv')
    
    return summary

def main():
    print(f"Total CPU cores: {TOTAL_CORES}")
    print(f"Cores for data generation: {DATA_GEN_CORES}")
    print(f"Cores for algorithm runs: {ALGORITHM_CORES}")
    
    # Create results directory
    os.makedirs('threshold_results', exist_ok=True)
    
    # Prepare experiment configurations
    experiment_configs = [(threshold, seed) 
                         for threshold in THRESHOLD_VALUES 
                         for seed in range(N_REPEATS)]
    
    # Run experiments in parallel with limited cores
    all_results = []
    with ProcessPoolExecutor(max_workers=min(len(experiment_configs), ALGORITHM_CORES)) as executor:
        futures = [executor.submit(run_experiment, config) for config in experiment_configs]
        for future in tqdm(as_completed(futures), total=len(experiment_configs), desc="Running experiments"):
            results = future.result()
            all_results.extend(results)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Generate summary statistics
    summary = generate_summary_statistics(results_df)
    
    print("Benchmark complete! Results are available in the 'threshold_results' directory.")
    print("Summary tables for paper:")
    print("- MSE Table: threshold_results/mse_table.csv")
    print("- Regret Table: threshold_results/regret_table.csv")

if __name__ == '__main__':
    main() 