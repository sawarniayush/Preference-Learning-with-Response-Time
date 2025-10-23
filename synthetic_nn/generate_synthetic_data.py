import numpy as np
import pandas as pd
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial


def build_reward_function(input_dim: int,
                          hidden1: int,
                          hidden2: int,
                          seed: int = None,
                          power: float = 3.0,
                          weight_scale: float = 1):
    """
    Three-layer neural network with standard activations.
    Architecture: input_dim -> hidden1 -> hidden2 -> output
    Uses ReLU activation for hidden layers and tanh for output.
    Larger `weight_scale` → more extreme values.
    """
    import numpy as np
    if seed is not None:
        np.random.seed(seed)

    # random weights for all three layers
    W1 = weight_scale * np.random.randn(input_dim, hidden1)
    b1 = np.zeros(hidden1)
    W2 = weight_scale * np.random.randn(hidden1, hidden2)
    b2 = np.zeros(hidden2)
    W3 = weight_scale * np.random.randn(hidden2, 1)
    b3 = np.zeros(1)

    def r(x: np.ndarray) -> np.ndarray:
        # x: (..., input_dim)
        # First hidden layer with sigmoid activation
        h1 = 1 / (1 + np.exp(-(x @ W1 + b1)))  # (..., hidden1)
        
        # Second hidden layer with sigmoid activation
        h2 = 1 / (1 + np.exp(-(h1 @ W2 + b2)))  # (..., hidden2)
        
        # Output layer with no activation to allow unbounded values
        out = (h2 @ W3 + b3)  # (..., 1)
        return out.squeeze(-1)  # Remove last dimension

    return r


def generate_Y_T(r1: float, r2: float, seed: int = None, a:float = 1) -> tuple:
    """
    Given true rewards r1 and r2, return:
      Y: in {{-1,+1}} indicating preference
      T: positive real-valued term

    TODO: Fill in the logic for Y and T generation.
    Example stub:
        Y = np.sign(r1 - r2)
        T = np.abs(r1 - r2)
    """
    # Generate synthetic response time
    # Simulate decision time based on pickscore difference
    drift = r1 - r2
    x = 0.0  # starting point
    dt = 0.0001
    sqrt_dt = np.sqrt(dt)
    threshold = a
    rng = np.random.RandomState()
    steps = 0
    for steps in range(100000):  # Cap at max steps
        x += drift * dt + sqrt_dt * rng.randn()
        if  abs(x) >= threshold:
            break
    T = steps * dt
    Y = 1 if x >= 0 else -1
    return Y,T


def _process_pair(args, a):
    """Helper function to process a single pair of rewards"""
    v1, v2 = args
    return generate_Y_T(v1, v2, a=a)


def sample_dataset(r_fn,
                   n_samples: int,
                   input_dim: int,
                   seed: int = None,
                   n_jobs: int = None,
                   threshold: float = 1.0) -> pd.DataFrame:
    """
    Samples X1, X2 ~ N(0,I), computes r1 = r_fn(X1), r2 = r_fn(X2),
    and uses generate_Y_T to produce Y, T. Returns a pandas DataFrame.
    Columns: 'X1', 'X2', 'Y', 'T'.
    
    Args:
        n_jobs: Number of parallel jobs. If None, uses all available CPUs.
        threshold: Barrier/threshold a for response time simulation.
    """
    if seed is not None:
        np.random.seed(seed)

    # Sample feature vectors
    X1 = np.random.randn(n_samples, input_dim)
    X2 = np.random.randn(n_samples, input_dim)
    # Compute true rewards
    r1_vals = r_fn(X1)
    r2_vals = r_fn(X2)

    # Generate Y, T for each pair in parallel
    if n_jobs is None:
        n_jobs = cpu_count()
    
    with Pool(n_jobs) as pool:
        func = partial(_process_pair, a=threshold)
        results = pool.map(func, zip(r1_vals, r2_vals))
    
    Y_list, T_list = zip(*results)

    # Build DataFrame
    df = pd.DataFrame({
        'X1': list(X1),
        'X2': list(X2),
        'Y':  Y_list,
        'T':  T_list,
        'true_r1': r1_vals,
        'true_r2': r2_vals
    })
    return df


def build_sobolev_reward(input_dim: int,
                        n_components: int = 2,
                        seed: int = None,
                        weight_scale: float = 1):
    """
    Creates a reward function based on elements from a Sobolev space.
    Uses a combination of polynomials and trigonometric functions as basis elements.
    Larger `weight_scale` → more extreme values.
    """
    import numpy as np
    if seed is not None:
        np.random.seed(seed)

    # Generate random weights for each basis function
    weights = np.random.randn(n_components)
    weights = weights / np.abs(weights).sum()  # normalize weights

    def r(x: np.ndarray) -> np.ndarray:
        # x: (..., input_dim)
        result = np.zeros(x.shape[:-1])
        
        # Add polynomial terms (x^2, x^3)
        for i in range(n_components):
            # Use different polynomial degrees for different components
            degree = i   # Start with quadratic terms
            poly_term = np.sum(x**degree, axis=-1)
            result += weights[i] * poly_term
            
            # Add sine terms with different frequencies
            freq = (i + 1) * np.pi
            sin_term = np.sum(np.sin(freq * x), axis=-1)
            result += weights[i] * sin_term
        
        return weight_scale * result
    
    return r


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic preference dataset with a richer reward function"
    )
    parser.add_argument('--train_size',  type=int,   default=20000,
                        help='Number of training samples')
    parser.add_argument('--test_size',   type=int,   default=2000,
                        help='Number of test samples')
    parser.add_argument('--input-dim',   type=int,   default=10,
                        help='Dimensionality of X1, X2')
    parser.add_argument('--hidden1',     type=int,   default=64,
                        help='First hidden layer size for true r')
    parser.add_argument('--hidden2',     type=int,   default=32,
                        help='Second hidden layer size for true r')
    parser.add_argument('--weight-scale',type=float, default=2.0,
                        help='Scale for random weights (larger → bigger outputs)')
    parser.add_argument('--seed',        type=int,   default=42,
                        help='Random seed')
    parser.add_argument('--train-out',   type=str,   default='train.pkl',
                        help='Output pickle for training data')
    parser.add_argument('--test-out',    type=str,   default='test.pkl',
                        help='Output pickle for test data')
    parser.add_argument('--n-jobs',      type=int,   default=None,
                        help='Number of parallel jobs. If None, uses all available CPUs.')
    parser.add_argument('--threshold',    type=float, default=1.0,
                        help='Barrier/threshold a for decision time simulation')

    args = parser.parse_args()

    # 1) Build the "blow-up" reward function
    r_fn = build_reward_function(
        input_dim=args.input_dim,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        seed=args.seed,
        weight_scale=args.weight_scale
    )
    # r_fn = build_reward_function(
    #     input_dim=args.input_dim,
    #     n_components=3,  # Using 3 basis components
    #     weight_scale=args.weight_scale
    # )

    # 2) Sample train & test DataFrames
    df_train = sample_dataset(
        r_fn,
        n_samples=args.train_size,
        input_dim=args.input_dim,
        seed=args.seed + 1,
        n_jobs=args.n_jobs,
        threshold=args.threshold
    )
    df_test = sample_dataset(
        r_fn,
        n_samples=args.test_size,
        input_dim=args.input_dim,
        seed=args.seed + 2,
        n_jobs=args.n_jobs,
        threshold=args.threshold
    )

    # 3) Persist to disk
    df_train.to_pickle(args.train_out)
    df_test.to_pickle(args.test_out)

    print(f"Saved {len(df_train)} training samples to {args.train_out}")
    print(f"Saved {len(df_test)} test     samples to {args.test_out}")

if __name__ == '__main__':
    main()
