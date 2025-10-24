##from scipy.stats import bernoulli
from joblib import Parallel, delayed
import numpy as np
def non_linear_reward_function(x, A=1, B=1, C=0, D=1, E=1, F=0, G=0):
    """Dummy non-linear function combining sine and a Gaussian bump."""
    sine_part = A * np.sin(B * x + C)
    gaussian_part = D * np.exp(-E * (x - F)**2)
    return sine_part + gaussian_part + G

# Define the reward function.
# For example, a linear reward: r(X) = X · true_w.
def reward_function(X, true_w):
    return np.dot(X, true_w)

def safe_all(x):
    """Check if all elements are True, handling both NumPy arrays and scalars"""
    if isinstance(x, (bool, int, float)):  # If it's a scalar
        return bool(x)
    elif hasattr(x, 'all'):  # If it's a NumPy array or similar
        return x.all()
    elif hasattr(x, '__iter__'):  # If it's another iterable
        return all(x)
    else:
        return bool(x)  # Default case, convert to bool


# Define the logistic (sigmoid) function.
def logistic(x):
    return 1 / (1 + np.exp(-x))

# Simulate a single trial of the drift–diffusion process.
def simulate_response_time(drift, dt=0.0001, threshold=1.0, max_steps=500000, rng=None):
    x = 0.0  # starting at 0
    t = 0.0
    if rng is None:
        rng = np.random.default_rng()
    sqrt_dt = np.sqrt(dt)
    for _ in range(max_steps):
        # Euler-Maruyama update: drift*dt + sqrt(dt)*noise
        x += drift * dt + sqrt_dt * rng.standard_normal()
        t += dt
        if abs(x) >= threshold:
            if x >= threshold:
                return (+1,t)
            else:
              return (-1,t)
    if x >=0:
          return (1,t)
    else:
          return (-1,t)

import pandas as pd
from joblib import Parallel, delayed
# Helper function to generate one sample.
def generate_one_sample(d, dt, true_w, X1_X2_bound, neighbouring_X, a_val = 1, rng=None):
    # Initialize RNG if not provided
    if rng is None:
        rng = np.random.default_rng()
    
    # Create a row dictionary to hold this sample's data.
    row_data = {}

    # Sample two points X1 and X2.
    X1 = rng.uniform(-X1_X2_bound, X1_X2_bound, d)
    X2 = rng.uniform(-X1_X2_bound, X1_X2_bound, d)
    p = rng.uniform(0, 1)
    if neighbouring_X:

        if p >0.5:
              X1 = X1 + true_w
              X2 = X2
        else:
              X1 = X1
              X2 = X2 + true_w
        # Normalize back to the bound.
    X1 = X1 / np.linalg.norm(X1, ord=2) * X1_X2_bound
    X2 = X2 / np.linalg.norm(X2, ord=2) * X1_X2_bound

    # Compute rewards.
    r1 = reward_function(X1, true_w)
    r2 = reward_function(X2, true_w)
    diff = r1 - r2

    # Compute probability using the logistic function.
    p = logistic(2*diff*a_val) ###2 is important as we have it in denominator

    # Sample the preference.
    preference = 1 if rng.random() < p else -1

    # Simulate the response time.
    T = simulate_response_time(diff, dt=dt, threshold = a_val, rng=rng)

    # Build the row.
    row_data['preference'] = preference
    row_data['T'] = T

    # Add X1 components.
    for i in range(d):
        row_data[f'X1_{i}'] = X1[i]

    # Add X2 components.
    for i in range(d):
        row_data[f'X2_{i}'] = X2[i]

    # Add true_w components.
    for i in range(len(true_w)):
        row_data[f'true_w_{i}'] = true_w[i]

    return row_data

def generate_preference_data_parallel(
    n_samples=1000,
    d=2,
    dt=0.0001,
    X1_X2_bound=1,
    theta_bound=1,
    neighbouring_X=False,
    n_jobs=-1,
    a_val = 1,
    a_dist = "nodist",
    a_dist_k = None,###denotes k for gamma
    rng=None,
):
    # Initialize RNG if not provided
    if rng is None:
        rng = np.random.default_rng()
    
    # Initialize true_w if not provided.
    # if true_w is None:
    true_w = theta_bound * rng.standard_normal(d)
    true_w = true_w/ np.linalg.norm(true_w,ord=2)*theta_bound

    X1 = rng.uniform(-X1_X2_bound, X1_X2_bound, (n_samples,d))
    X2 = rng.uniform(-X1_X2_bound, X1_X2_bound, (n_samples,d))
    p = rng.uniform(0, 1)
    if neighbouring_X:

        if p >0.5:
              X1 = X1 + true_w
              X2 = X2
        else:
              X1 = X1
              X2 = X2 + true_w
        # Normalize back to the bound.
    def renorm_rows(mat, target):
      norms = np.linalg.norm(mat, axis=1, keepdims=True)      # shape (n,1)
      norms[norms == 0] = 1                                   # avoid /0
      return mat / norms * target
    X1 = renorm_rows(X1, X1_X2_bound)
    X2 = renorm_rows(X2, X1_X2_bound)

    # Compute rewards.
    r1 = np.dot(X1, true_w)
    r2 = np.dot(X2, true_w)
    drift = r1 - r2

    if a_dist == "nodist":
      a_val = np.full(n_samples,a_val)

    elif a_dist == "gamma":
      a_val = rng.gamma(shape= a_dist_k, scale=a_val/a_dist_k, size=n_samples)
    else:
      raise ValueError("Invalid a_dist type")

    # Create separate RNGs for each parallel task to ensure reproducibility
    # Each task gets a unique seed derived from the main RNG
    seeds = rng.integers(0, 2**31, size=n_samples)

    # Parallelize the sample generation.
    Y_T_array = Parallel(n_jobs=n_jobs)(
        delayed(simulate_response_time)(drift[itr], dt, a_val[itr], rng=np.random.default_rng(seeds[itr]))
        for itr in range(n_samples)
    )

    X1_cols = [f'X1_{i}' for i in range(d)]
    X2_cols = [f'X2_{i}' for i in range(d)]

    true_w_cols = [f'true_w_{i}' for i in range(d)]

    X_combined = np.hstack([X1, X2])  # shape: (N, 2d)

    df = pd.DataFrame(np.hstack([X1, X2]), columns=X1_cols + X2_cols)
    df[['preference', 'T']] = np.array(Y_T_array)

    df[true_w_cols] = true_w

    

    # preference = 2*np.random.binomial(n=1, p=expit(2*a_val*drift), size=n_samples)-1 ####2*(expit(2*a_val*drift) > 0.5) - 1
    # df_separate['preference'] = preference

    ##true_w_col_names  = [f'true_w_{i}' for i in range(d)]

    

    # Convert the list of dictionaries into a DataFrame.

    return df