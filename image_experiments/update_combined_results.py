import json
import glob
import os

# Find all individual results files
result_files = glob.glob('scaling_results/results_N*_*.json')

# Collect all results
all_results = []
for file in result_files:
    with open(file, 'r') as f:
        results = json.load(f)
        all_results.extend(results)

# Save to all_results file
output_file = 'scaling_results/all_results_' + os.path.basename(result_files[0]).split('_')[2]
with open(output_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"Combined results saved to {output_file}") 