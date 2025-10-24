#!/usr/bin/env python3
import os
import time
import argparse
import numpy as np
import pandas as pd
# from your_module import generate_preference_data_parallel

# Default parameter values if not provided via CLI
default_dims = [5,20]  # to add 10 later
default_theta_bounds = [1,2,4,5,6,8,10,12,15,20] ###[0.1, 0.5, 1, 2, 4, 5, 10, 15, 20, 25, 50, 75, 100, 200]
default_a_vals = [1.0] ##[i/10 for i in range(5, 18)]
default_iterations = 10
default_n_samples = 15000
default_output_dir = "data_files"
default_base_seed = 42  # Base seed for reproducibility

from data_gen_functions import generate_preference_data_parallel

def main(dims, theta_bounds, a_vals, iterations, n_samples, output_dir, base_seed=42, a_dist="nodist", a_dist_k_vals=None):
    os.makedirs(output_dir, exist_ok=True)
    run_no = 1
    last_time = time.time()

    # Create a master RNG from base seed
    master_rng = np.random.default_rng(base_seed)
    print(f"Initialized master RNG with base seed: {base_seed}")

    if a_dist_k_vals is None:
        a_dist_k_vals = [None]

    for itr in range(iterations): ###changed temporarily to generate more data
        for dim in dims:
            for theta_bound in theta_bounds:
                for a_val in a_vals:
                    for a_dist_k in a_dist_k_vals:
                        
                        if a_dist == "gamma":
                            filename = f"Dim_{dim}_theta{theta_bound}_a{a_val}_gamma_k{a_dist_k}_itr{itr}.csv"
                        elif a_dist == "nodist":
                            filename = f"Dim_{dim}_theta{theta_bound}_a{a_val}_itr{itr}.csv"
                        else:
                            raise ValueError(f"Unknown a_dist: {a_dist}")
                        
                        filepath = os.path.join(output_dir, filename)

                        if os.path.exists(filepath):
                            print(f"Skipping {filepath} (already exists)")
                            continue

                        # only generate & save if file does not exist
                        df = generate_preference_data_parallel(
                            n_samples=n_samples,
                            d=dim,
                            theta_bound=theta_bound,
                            a_val=a_val,
                            a_dist=a_dist,
                            a_dist_k=a_dist_k,
                            rng=master_rng
                        )
                        df.to_csv(filepath, index=False)
                        print(f"Saved {filepath} (iter={itr}, dim={dim}, theta={theta_bound}, a={a_val}, a_dist={a_dist}, a_dist_k={a_dist_k})")

                        curr_time = time.time()
                        elapsed = curr_time - last_time
                        print(f"  Took {elapsed:.2f}s (run #{run_no})")
                        last_time = curr_time
                        run_no += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate and save synthetic preference data in parallel."
    )
    parser.add_argument(
        "--dims", "-d",
        nargs="+", type=int,
        default=default_dims,
        help=f"List of feature dimensions (default: {default_dims})"
    )
    parser.add_argument(
        "--theta-bounds", "-t",
        nargs="+", type=float,
        default=default_theta_bounds,
        help=f"List of theta_bound values (default: {default_theta_bounds})"
    )
    parser.add_argument(
        "--a-vals", "-a",
        nargs="+", type=float,
        default=default_a_vals,
        help=f"List of a_val parameters (default: {default_a_vals})"
    )
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=default_iterations,
        help=f"Number of outer iterations to run (default: {default_iterations})"
    )
    parser.add_argument(
        "--n-samples", "-n",
        type=int,
        default=default_n_samples,
        help=f"Number of samples per call (default: {default_n_samples})"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=default_output_dir,
        help=f"Directory to write CSV files into (default: '{default_output_dir}')"
    )
    parser.add_argument(
        "--base-seed", "-s",
        type=int,
        default=default_base_seed,
        help=f"Base random seed for reproducibility (default: {default_base_seed})"
    )
    args = parser.parse_args()

    main(
        dims=args.dims,
        theta_bounds=args.theta_bounds,
        a_vals=args.a_vals,
        iterations=args.iterations,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        base_seed=args.base_seed,
    )
    #  a_dist="gamma",  # or "gamma" based on your requirement
    #     a_dist_k_vals= [1,3]