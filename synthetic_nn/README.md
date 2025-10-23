# Synthetic Preference Learning Benchmarks

End-to-end instructions for recreating the synthetic preference-learning experiments that accompany the paper. Follow the steps below to generate datasets, run every algorithmic baseline, and recreate the plots used in the analysis.

## Repository Layout
- `generate_synthetic_data.py` – builds the neural reward function and simulates preference/time pairs.
- `benchmark_algorithms.py` – sweeps over training set sizes for all learners and writes aggregated metrics to `results/`.
- `threshold_benchmark.py` – sweeps over diffusion-threshold values and writes outputs to `threshold_results/`.
- `generate_plots.py` – turns CSV results into accuracy/MSE/regret figures. `plot_gen_command.sh` shows an example invocation.
- `*_learner.py` files – individual learning algorithms (PyTorch models) invoked by the benchmark scripts.
- `results/`, `threshold_results/` – default output directories (may already contain sample artifacts).

## Environment Setup
1. **Python**: Use Python 3.9–3.11 (scripts are CPU-only and tested with PyTorch 2.x).
2. **Virtual environment **:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   Installing PyTorch via `requirements.txt` grabs the CPU wheels. If you prefer GPU acceleration, install the CUDA-enabled build from [pytorch.org](https://pytorch.org/get-started/locally/) after activating the environment.

## Reproducing the Training-Size Sweep
This recreates the main benchmark where training size varies and each learner is run multiple times.

```bash
python benchmark_algorithms.py
```

- Expected runtime: ~1–2 hours on a modern 8–16 core CPU (the script parallelises across processes and will fully utilise available cores).
- Intermediate data appear in `data/` as pickle files; final metrics are stored in `results/raw_results.csv` and `results/summary_statistics.csv`.
- Plots (`accuracy_plot.png`, `mse_plot.png`, `regret_plot.png`) are generated automatically in `results/`.

Restarting the script will reuse cached datasets if the pickle files already exist.

## Reproducing the Threshold Sweep
This experiment varies the diffusion threshold (boundaries) used when simulating decision times.

```bash
python threshold_benchmark.py
```

- Creates datasets in `data_threshold/` and writes metrics to `threshold_results/raw_results.csv`.
- Aggregated tables (`threshold_results/mse_table.csv`, `threshold_results/regret_table.csv`, and associated standard deviation tables) are ready for LaTeX import.
- Runtime is similar to the training-size sweep because each threshold value triggers fresh data generation and learner training.

## Plot Generation
Use `generate_plots.py` to rebuild figures from any results CSV. The script can operate on either raw runs or summary statistics.

### From Raw Results
```bash
python generate_plots.py \
  --raw threshold_results/raw_results.csv \
  --output threshold_results/plots \
  --use raw \
  --algorithms log_loss_learner.py orthogonal_loss_learner.py nonorthogonal_loss_learner.py \
  --font-size1 24 \
  --font-size2 18
```

### From Summary Statistics
```bash
python generate_plots.py \
  --summary results/summary_statistics.csv \
  --output results/summary_plots \
  --use summary
```

The helper script `plot_gen_command.sh` contains another ready-to-run command. All generated figures are saved as PNG files in the requested output directory.

## Verification Checklist
- [ ] Environment created and `pip install -r requirements.txt` succeeds.
- [ ] `python benchmark_algorithms.py` finishes and populates `results/`.
- [ ] `python threshold_benchmark.py` finishes and populates `threshold_results/`.
- [ ] `python generate_plots.py ...` reproduces accuracy/MSE/regret figures from the chosen CSV.

If any step fails, capture the console output (including stack traces) for debugging. Long runtimes can be shortened for smoke tests by editing the constants at the top of `benchmark_algorithms.py` or `threshold_benchmark.py` (e.g., reduce `TRAIN_SIZES`, `THRESHOLD_VALUES`, or `N_REPEATS`), though doing so will change the final statistics.

