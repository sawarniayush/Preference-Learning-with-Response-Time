#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd

# Loader functions matching generate_data.py naming convention:
# Filenames: Dim_<dim>_theta<theta_bound>_a<a_val>[_Neigh_X][_gamma_<k>]_itr<csv_index>.csv
# Data directory is fixed to 'data_gen/data_files'.

SHARED_BASE = 'data_gen/data_files'

def numpy_data_str_pref_data(
    theta_bound=1,
    csv_index=0,
    dimension=2,
    a_val=0.5,
    neighbouring_X=False,
    a_dist="nodist",
    a_dist_k=None
):
    """
    Returns: (X_diff_array, pref_array, time_array, true_w)
    Reads the CSV file saved by generate_data.py from SHARED_BASE directory.
    """
    # Build filename parts
    parts = [
        f"Dim_{dimension}",
        f"theta{theta_bound}",
        f"a{a_val}"
    ]
    if neighbouring_X:
        parts.append('Neigh_X')
    if a_dist == 'gamma' and a_dist_k is not None:
        parts.append(f"gamma_k{a_dist_k}")
    parts.append(f"itr{csv_index}")
    filename = '_'.join(parts) + '.csv'
    filepath = os.path.join(SHARED_BASE, filename)

    # Load DataFrame
    df = pd.read_csv(filepath)

    # Infer dimension from X1 columns
    X1_cols = [c for c in df.columns if c.startswith('X1_')]
    dim = len(X1_cols)

    # Extract arrays
    X1 = df[[f"X1_{i}" for i in range(dim)]].to_numpy()
    X2 = df[[f"X2_{i}" for i in range(dim)]].to_numpy()
    X_diff_array = X1 - X2

    true_w = df[[f"true_w_{i}" for i in range(dim)]].to_numpy()[0, :]
    pref_array = df['preference'].to_numpy()
    time_array = df['T'].to_numpy()

    return X_diff_array, pref_array, time_array, true_w


def numpy_data_str_pref_data_true_w(
    theta_bound=1,
    csv_index=0,
    dimension=2,
    a_val=0.5,
    a_dist="nodist",
    a_dist_k=None
):
    """
    Returns only the true_w vector from the corresponding CSV.
    """
    # Build filename parts (same as above)
    parts = [
        f"Dim_{dimension}",
        f"theta{theta_bound}",
        f"a{a_val}"
    ]
    if a_dist == 'gamma' and a_dist_k is not None:
        parts.append(f"gamma_k{a_dist_k}")
    parts.append(f"itr{csv_index}")
    filename = '_'.join(parts) + '.csv'
    filepath = os.path.join(SHARED_BASE, filename)

    # Load DataFrame and extract true_w
    df = pd.read_csv(filepath)
    dim = len([c for c in df.columns if c.startswith('true_w_')])
    true_w = df[[f"true_w_{i}" for i in range(dim)]].to_numpy()[0, :]
    return true_w
