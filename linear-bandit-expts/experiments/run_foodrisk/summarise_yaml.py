import yaml
import numpy as np
import pandas as pd

# === CONFIG ===
yaml_file = "run_Clithero/processed_result_parallel.yaml"
method_map = {
    "GLM_trans": "Preference Only",
    "LM_trans": "LZR+24",
    "LMOrtho_trans": "Ours",
}
budgets = [500, 1000, 1500]

# === LOAD YAML ===
with open(yaml_file, "r") as f:
    data = yaml.safe_load(f)

# === COLLECT RESULTS ===
results = []

for method_key, method_name in method_map.items():
    if method_key not in data:
        print(f"⚠️ Skipping {method_key} (not found in YAML)")
        continue
    method_data = data[method_key]

    # iterate over outer index (e.g., 2)
    for outer_idx, outer_data in method_data.items():
        for budget in budgets:
            if budget not in outer_data:
                continue

            budget_data = outer_data[budget]
            vals = []
            for item_id, metrics in budget_data.items():
                if "mistake_at_budget_mean" in metrics:
                    vals.append(metrics["mistake_at_budget_mean"])

            if vals:
                q1, median, q3 = np.percentile(vals, [25, 50, 75])
                results.append({
                    "Budget": budget,
                    "Method": method_name,
                    "Q1": round(q1, 2),
                    "Median": round(median, 2),
                    "Q3": round(q3, 2),
                })
            else:
                print(f"⚠️ No valid mistake values for {method_name} at budget {budget}")

# === PRINT FORMATTED OUTPUT ===
df = pd.DataFrame(results)
if df.empty:
    raise ValueError("No results collected — check structure or keys.")

for budget in budgets:
    print(f"\nBudget = {budget}\n")
    subdf = df[df["Budget"] == budget][["Method", "Q1", "Median", "Q3"]]
    if subdf.empty:
        print("  (No data available)")
    else:
        print(subdf.to_string(index=False))
