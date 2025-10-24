
import os
import sys
import time
import numpy as np
from scipy.special import expit
from read_file import numpy_data_str_pref_data
# from learning_functions import (
#     joint_pref_time_learning_regression,
#     joint_pref_time_learning_time_separate,
#     logistic_regression_preferences
# )
from dictionary_gen_functions import compare_reward_time_dml_no_time, compute_metrics, compute_metrics_y_nuisance, compare_reward_time_dml_no_time_y_nuisance
import pickle
###now compute for y_nuisance

# Redirect output to log file
sys.stdout = open('run_dict_metrics.log', 'a', buffering=1)

# -----------------------------
# Main experiment loop
# -----------------------------
dict_file = 'data_gen/dictionary_diff_diff_data_y_nuisance.pkl'

# Load existing dictionary if present
if os.path.exists(dict_file):
    with open(dict_file, 'rb') as f:
        dict_diff_comp_theta_data = pickle.load(f)
else:
    dict_diff_comp_theta_data = {}
os.makedirs('data_gen', exist_ok=True)

# Create master RNG for reproducibility
base_seed = 42
master_rng = np.random.default_rng(base_seed)
print(f"Initialized master RNG with base seed: {base_seed}")

count = 0
iterations = 10  # Adjusted for example; originally was 15 trials per setting
theta_bounds = [1,2,5, 8, 10] #[1, 2,4 5, 6,8, 10, 12,15]
a_val_bound = [1.0]###[i/10 for i in range(5, 18)] ##[i/10 for i in range(5, 18)]

for bound in theta_bounds:
    for dimension in [10, 20]:  # Adjusted for example; originally was 2, 5, 10, 20
        for data_points in [1000, 2000,5000, 8000, 10000, 12000]:
            for a_val in a_val_bound:
                all_metrics = []
                for itr in range(iterations): ### however for cases with a_val  = 1.0 we have 10 runs
                    key = (bound, itr, dimension, data_points, a_val)
                    if key in dict_diff_comp_theta_data:
                        m = dict_diff_comp_theta_data[key]
                    else:
                        m = compare_reward_time_dml_no_time_y_nuisance(
                            theta_bound=bound,
                            csv_index=itr,
                            dimension=dimension,
                            data_points=data_points,
                            a_val=a_val,
                            k_fold=2,
                            random_state=master_rng
                        )
                        dict_diff_comp_theta_data[key] = m

                    all_metrics.append(m)

                for method in all_metrics[0].keys():
                    avg_dist = np.mean([m[method]["distance"] for m in all_metrics])
                    avg_acc = np.mean([m[method]["accuracy"] for m in all_metrics])
                    print(
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')} bound={bound:>3} dim={dimension:>2} "
                        f"a_val={a_val:.1f} {method:40s} avg_dist={avg_dist:.4f} avg_acc={avg_acc:.4f}"
                    )

                print(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} Count {count} over "
                    f"bound={bound}, dim={dimension}, data_points={data_points}, a_val={a_val:.1f}"
                )
                count += 1

# Save updated dictionary

            with open(dict_file, 'wb') as f:
                pickle.dump(dict_diff_comp_theta_data, f)
            print(f"Saved dictionary to {dict_file}")