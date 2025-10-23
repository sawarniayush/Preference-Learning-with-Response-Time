# Preference Learning on PickScore

This repository contains an end-to-end workflow for running preference-learning experiments on the PickScore / Pick-a-Pic image–text dataset. The code covers: downloading the raw data, scoring image pairs with the PickScore model, generating synthetic response times, extracting CLIP embeddings, training several learning algorithms, and visualising policy value, MSE, and regret.

## Prerequisites

- Python 3.9 or newer.
- CUDA-capable GPUs strongly recommended for embedding generation and training.
- Install dependencies into a virtual environment with:

  ```bash
  python -m venv .venv
  source .venv/bin/activate  # or .venv\Scripts\activate on Windows
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

  The `requirements.txt` file captures everything used by the scripts here plus the auxiliary libraries that the [PickScore repository](https://github.com/yuvalkirstain/PickScore) relies on for inference workflows.

- Select the appropriate PyTorch build for your hardware (CUDA, ROCm, or CPU only). If you need a GPU-enabled wheel, install it following the [official PyTorch instructions](https://pytorch.org/get-started/locally/) before or after running `pip install -r requirements.txt`.
- The optional dependencies listed under "Optional utilities commonly used with PickScore" in `requirements.txt` cover the CLI/logging stack used by the upstream PickScore project. They are not required for the scripts in this repo; feel free to omit them if you only need the embedding and training workflow here. (If you do plan to launch PickScore's original training runs, you may also need heavier extras such as `deepspeed` as documented upstream.)

## Workflow Overview

1. **Download raw Pick-a-Pic data** with `download_dataset.py`.
2. **Augment the dataset** with PickScore predictions, synthetic pairwise labels, and drift–diffusion response times using `enhance_dataset.py`.
3. **Generate CLIP embeddings** for text and images via `generate_embeddings.py`.
4. **Convert embeddings to Parquet shards** with `prepare_training_data.py` for fast streaming during training.
5. **Train preference learners** (`log_loss`, `orthogonal`, `nonorthogonal`, `orthogonal_crossfit`) directly or orchestrate scaling experiments using `run_scaling_comparison.py`.
6. **Aggregate & plot results** with `update_combined_results.py` and `plot_scaling_results.py`.

Each stage writes its outputs to `data/` or `scaling_results/`, allowing you to resume without re-running previous steps.

## Step-by-Step Instructions

### 1. Download the dataset

```bash
python download_dataset.py \
  --dataset yuvalkirstain/pickapic_v1 \
  --output_dir data/pickscore_raw
```

The script caches the Hugging Face dataset in `data/hf_cache/` and stores a disk snapshot at `data/pickscore_raw/` for offline use. Pass `--revision` if you need a specific commit, or change `--dataset` to `yuvalkirstain/pickapic_v2` for the newer release.

### 2. Enhance with PickScore & synthetic feedback

```bash
python enhance_dataset.py \
  --dataset_path data/pickscore_raw \
  --output_dir data/pickscore_enhanced
```

What it does:
- loads the PickScore model (`yuvalkirstain/PickScore_v1`) to score each image in the pair,
- normalises scores and stores `pickscore_0`, `pickscore_1`, `pickscore_diff`, and softmax preference probabilities,
- simulates binary decisions and response times via a drift–diffusion process (Brownian motion with drift determined by the PickScore difference),
- writes the enriched dataset back to disk (one folder per split plus a consolidated `DatasetDict`).

Use `--sample_size` to restrict processing during dry runs, and `--num_proc` to parallelise Hugging Face `map()` calls. The script auto-detects whether to call `datasets.load_dataset` (remote) or `datasets.load_from_disk` (local path).

### 3. Generate CLIP embeddings

```bash
python generate_embeddings.py \
  --dataset_path data/pickscore_enhanced \
  --output_dir data/embeddings \
  --batch_size 64 \
  --gpus 2
```

Key behaviour:
- splits each dataset split across the requested GPUs (falls back to CPU if none available),
- runs the `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` model to obtain L2-normalised image and text features,
- concatenates the text feature with each image feature to form 2048-dim inputs (`X1`, `X2`),
- propagates PickScore targets, synthetic labels (`Y`), response times (`T`), and ground-truth rewards (`true_r1`, `true_r2`),
- writes `data/embeddings/<split>_embeddings.pkl`.

Each row retains a `pair_index` to preserve ordering. Adjust `--model_name` / `--processor_name` to experiment with other CLIP checkpoints.

### 4. Convert embeddings to Parquet shards

```bash
python prepare_training_data.py \
  --input_dir data/embeddings \
  --output_dir data/training_chunks \
  --rows_per_chunk 50000
```

This converts every `*_embeddings.pkl` into fixed-size Parquet shards (one subdirectory per split) suitable for streaming with `pandas.read_parquet(engine="pyarrow")`. A `pair_index` column is preserved for traceability while vectors are stored as fixed-length float lists.

### 5. Train preference learners

Each learner expects a directory of Parquet shards. Example for the log-loss learner:

```bash
python log_loss_learner_large.py \
  --train_folder data/training_chunks/train \
  --num_samples 200000 \
  --test_size 10000 \
  --epochs 15 \
  --seed 42
```

Two-stage learners (`orthogonal_loss_learner_large.py`, `nonorthogonal_loss_learner_large.py`, `orthogonal_loss_learner_crossfit.py`) accept analogous arguments plus `--epochs1` / `--epochs2`.

#### Scaling comparisons

Automate seeds, algorithms, and sample sizes:

```bash
python run_scaling_comparison.py \
  --train_folder data/training_chunks/train \
  --output_dir scaling_results \
  --gpus 2 \
  --seeds 42 123 456 \
  --sizes 20000 50000 100000 \
  --algorithms log_loss orthogonal nonorthogonal orthogonal_crossfit
```

The orchestrator pins each subprocess to the requested GPU, streams progress logs, stores JSON summaries under `scaling_results/`, and renders policy value / MSE / regret plots.

### 6. Aggregate & plot results

- Combine per-size JSON outputs into a single file:

  ```bash
  python update_combined_results.py
  ```

- Generate richer or customised visualisations:

  ```bash
  python plot_scaling_results.py \
    --input scaling_results/all_results_<timestamp>.json
  ```

  (The plotting script already runs inside `run_scaling_comparison.py`; invoke it manually when re-plotting archived results.)

## Script Reference

- `download_dataset.py`: Download/cache PickScore data from Hugging Face.
- `enhance_dataset.py`: Apply PickScore scoring, generate synthetic labels + response times, save enhanced dataset.
- `generate_embeddings.py`: Extract CLIP embeddings with multi-device support and persist pickle shards.
- `prepare_training_data.py`: Convert embedding pickles into Parquet chunks with fixed-size vector columns.
- `log_loss_learner_large.py`: Single-model baseline preference learner.
- `orthogonal_loss_learner_large.py`: Two-stage orthogonal loss learner with shared response/label models.
- `nonorthogonal_loss_learner_large.py`: Variant without the orthogonality correction term.
- `orthogonal_loss_learner_crossfit.py`: Cross-fitted two-stage learner.
- `run_scaling_comparison.py`: Batch runner + plotting utility for scaling sweeps.
- `update_combined_results.py`: Merge individual JSON results.
- `plot_scaling_results.py`: Stand-alone plotting helper (used by the runner).

The `scaling_results/` directory contains historical JSON logs and example plots for reference.

## Tips & Troubleshooting

- **GPU memory**: Large batches of 1024-d CLIP features can consume >10 GB. Lower `--batch_size` in `generate_embeddings.py` or training scripts if you see out-of-memory errors.
- **pyarrow version**: Ensure `pyarrow >= 9` for fixed-size list support when writing Parquet.
- **Synthetic response dynamics**: The Brownian-response simulation caps at 100k time steps; increase `threshold` / `dt` in `enhance_dataset.py` if you need different dynamics.
- **Resuming work**: Every major stage writes to disk. If you re-run a step, delete or redirect the corresponding output directory to avoid mixing stale and fresh artefacts.

## Repository Layout

- `data/` – created at runtime; holds cached datasets, enhanced datasets, embeddings, and Parquet shards.
- `models/` – learner checkpoints saved by training scripts.
- `scaling_results/` – experiment JSONs and generated plots.

Feel free to adapt the scripts for alternative datasets or embedding models—the documentation above should help you understand where to plug in new components. Have fun experimenting!
