#!/usr/bin/env python3
"""
Generate CLIP embeddings for the PickScore preference dataset.

The script supports multi-GPU execution and writes one pickle file per split
that can later be converted to Parquet shards via ``prepare_training_data.py``.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from io import BytesIO
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from datasets import load_from_disk
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoProcessor

# Configure logging once, even across multiprocessing
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


class ImageTextDataset(Dataset):
    """Thin wrapper to expose HuggingFace rows as PyTorch tensors."""

    def __init__(self, dataset, start: int = 0, end: int | None = None):
        self.dataset = dataset
        self.start = start
        self.end = len(dataset) if end is None else end

    def __len__(self) -> int:
        return max(0, self.end - self.start)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        item = self.dataset[self.start + idx]
        return {
            "idx": self.start + idx,
            "jpg_0": item["jpg_0"],
            "jpg_1": item["jpg_1"],
            "caption": item["caption"],
            "pickscore_0": item["pickscore_0"],
            "pickscore_1": item["pickscore_1"],
            "synth_label_0": item["synth_label_0"],
            "synth_label_1": item["synth_label_1"],
            "synth_time": item["synth_response_time"],
        }


def open_image(image_bytes):
    """Convert image bytes to PIL Image object."""
    if isinstance(image_bytes, bytes):
        image = Image.open(BytesIO(image_bytes))
    else:
        image = image_bytes
    return image.convert("RGB")


def process_batch(batch, processor, model, device: torch.device) -> pd.DataFrame:
    """Process a batch of data to generate embeddings."""
    indices = batch.pop("idx")
    images_0 = [open_image(img) for img in batch["jpg_0"]]
    images_1 = [open_image(img) for img in batch["jpg_1"]]
    captions = batch["caption"]

    image_inputs_0 = processor(
        images=images_0, padding=True, truncation=True, max_length=77, return_tensors="pt"
    ).to(device)
    image_inputs_1 = processor(
        images=images_1, padding=True, truncation=True, max_length=77, return_tensors="pt"
    ).to(device)
    text_inputs = processor(
        text=captions, padding=True, truncation=True, max_length=77, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        image_embs_0 = model.get_image_features(**image_inputs_0)
        image_embs_1 = model.get_image_features(**image_inputs_1)
        text_embs = model.get_text_features(**text_inputs)

        image_embs_0 = image_embs_0 / torch.norm(image_embs_0, dim=-1, keepdim=True)
        image_embs_1 = image_embs_1 / torch.norm(image_embs_1, dim=-1, keepdim=True)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

    batch_size = len(captions)
    pair_index = np.array(indices, dtype=np.int64)

    labels0 = torch.as_tensor(batch["synth_label_0"])
    labels1 = torch.as_tensor(batch["synth_label_1"])
    response_times = torch.as_tensor(batch["synth_time"], dtype=torch.float32)
    pickscore_0 = torch.as_tensor(batch["pickscore_0"], dtype=torch.float32)
    pickscore_1 = torch.as_tensor(batch["pickscore_1"], dtype=torch.float32)

    df_data = {
        "pair_index": pair_index,
        "X1": [
            torch.cat([image_embs_0[i], text_embs[i]], dim=0).cpu().numpy()
            for i in range(batch_size)
        ],
        "X2": [
            torch.cat([image_embs_1[i], text_embs[i]], dim=0).cpu().numpy()
            for i in range(batch_size)
        ],
        "Y": (labels0 - labels1).cpu().numpy(),
        "T": response_times.cpu().numpy(),
        "true_r1": pickscore_0.cpu().numpy(),
        "true_r2": pickscore_1.cpu().numpy(),
    }
    return pd.DataFrame(df_data)


def worker(
    rank: int,
    device_id: str,
    split_name: str,
    dataset,
    start_idx: int,
    end_idx: int,
    output_file: str,
    model_name: str,
    processor_name: str,
    batch_size: int,
) -> None:
    """Worker function for each device."""
    try:
        device = torch.device(device_id)
        LOGGER.info(
            "Worker %d processing %s split on %s with %d examples",
            rank,
            split_name,
            device,
            end_idx - start_idx,
        )

        processor = AutoProcessor.from_pretrained(processor_name)
        model = AutoModel.from_pretrained(model_name).eval().to(device)

        data = ImageTextDataset(dataset, start=start_idx, end=end_idx)
        pin_memory = device.type == "cuda"
        dataloader = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )

        dfs: List[pd.DataFrame] = []
        for batch in tqdm(dataloader, desc=f"{split_name} [{device_id}]"):
            batch_df = process_batch(batch, processor, model, device)
            dfs.append(batch_df)

        combined_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        shard_path = f"{output_file}_shard{rank:02d}.pkl"
        combined_df.to_pickle(shard_path)
        LOGGER.info(
            "Worker %d finished %s split on %s; wrote %s",
            rank,
            split_name,
            device,
            shard_path,
        )
    except Exception:  # pragma: no cover - propagated back to parent process
        LOGGER.exception("Worker %d failed on split %s", rank, split_name)
        raise


def combine_shards(output_file: str, shard_paths: Sequence[str]) -> None:
    dataframes = [pd.read_pickle(path) for path in shard_paths if os.path.exists(path)]
    if not dataframes:
        raise RuntimeError(f"No shard files were produced for {output_file}")

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.sort_values("pair_index", inplace=True)
    combined_df.reset_index(drop=True, inplace=True)
    combined_df.to_pickle(f"{output_file}.pkl")

    for path in shard_paths:
        os.remove(path)
    LOGGER.info("Wrote combined embeddings to %s.pkl", output_file)


def compute_shards(dataset, num_workers: int) -> List[tuple[int, int, int]]:
    total = len(dataset)
    if total == 0:
        return []

    chunk_size = math.ceil(total / num_workers)
    shards: List[tuple[int, int, int]] = []
    start = 0
    rank = 0
    while start < total and rank < num_workers:
        end = min(start + chunk_size, total)
        shards.append((rank, start, end))
        start = end
        rank += 1
    return shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_path",
        default="data/pickscore_brownian",
        help="Path to the enhanced dataset saved with datasets.save_to_disk",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test_unique"],
        help="Dataset splits to process (default: %(default)s)",
    )
    parser.add_argument(
        "--output_dir",
        default="data/embeddings",
        help="Directory to store the pickle outputs (default: %(default)s)",
    )
    parser.add_argument(
        "--model_name",
        default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        help="CLIP model checkpoint for embeddings",
    )
    parser.add_argument(
        "--processor_name",
        default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        help="Corresponding processor/tokenizer name",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size per device (default: %(default)s)",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: use all available, CPU if none)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    LOGGER.info("Loading dataset from %s", args.dataset_path)
    dataset = load_from_disk(args.dataset_path)

    os.makedirs(args.output_dir, exist_ok=True)

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if args.gpus is None:
        requested_gpus = available_gpus
    else:
        requested_gpus = min(args.gpus, available_gpus)

    if requested_gpus > 0:
        device_ids = [f"cuda:{i}" for i in range(requested_gpus)]
    else:
        device_ids = ["cpu"]

    LOGGER.info("Using devices: %s", device_ids)

    for split_name in args.splits:
        if split_name not in dataset:
            LOGGER.warning("Split %s not found in dataset; skipping", split_name)
            continue

        split_data = dataset[split_name]
        total_examples = len(split_data)
        LOGGER.info("Processing split %s with %d examples", split_name, total_examples)

        output_file = os.path.join(args.output_dir, f"{split_name}_embeddings")

        shards = compute_shards(split_data, len(device_ids))
        if not shards:
            LOGGER.warning("Split %s has no data; skipping", split_name)
            continue

        processes: List[mp.Process] = []
        shard_paths: List[str] = []

        for (rank, start, end), device_id in zip(shards, device_ids):
            mp_args = (
                rank,
                device_id,
                split_name,
                split_data,
                start,
                end,
                output_file,
                args.model_name,
                args.processor_name,
                args.batch_size,
            )
            process = mp.Process(target=worker, args=mp_args)
            process.start()
            processes.append(process)
            shard_paths.append(f"{output_file}_shard{rank:02d}.pkl")

        for p in processes:
            p.join()

        if any(p.exitcode != 0 for p in processes):
            raise RuntimeError(f"At least one worker failed for split {split_name}")

        combine_shards(output_file, shard_paths)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
