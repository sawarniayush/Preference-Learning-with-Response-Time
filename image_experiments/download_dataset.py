#!/usr/bin/env python3
"""
Download the PickScore preference dataset from Hugging Face and persist it
locally so downstream scripts can load it without hitting the network again.
"""

import argparse
import logging
import os
from typing import Optional

from datasets import DatasetDict, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="yuvalkirstain/pickapic_v1",
        help="Hugging Face dataset identifier (default: %(default)s)",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional dataset revision / commit hash",
    )
    parser.add_argument(
        "--cache_dir",
        default="data/hf_cache",
        help="Directory used by Hugging Face to cache the download (default: %(default)s)",
    )
    parser.add_argument(
        "--output_dir",
        default="data/pickscore_raw",
        help="Where to store the downloaded dataset via save_to_disk (default: %(default)s)",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=8,
        help="Number of CPU processes used by Hugging Face (default: %(default)s)",
    )
    return parser.parse_args()


def download_dataset(
    dataset_name: str,
    cache_dir: str,
    output_dir: str,
    num_proc: int,
    revision: Optional[str] = None,
) -> DatasetDict:
    logging.info(
        "Downloading dataset %s (revision=%s) -> cache=%s",
        dataset_name,
        revision,
        cache_dir,
    )
    ds = load_dataset(
        dataset_name,
        revision=revision,
        num_proc=num_proc,
        cache_dir=cache_dir,
    )
    os.makedirs(output_dir, exist_ok=True)
    logging.info("Saving dataset to %s", output_dir)
    ds.save_to_disk(output_dir)
    logging.info("Download complete; splits: %s", {k: len(v) for k, v in ds.items()})
    return ds


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    download_dataset(
        dataset_name=args.dataset,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        num_proc=args.num_proc,
        revision=args.revision,
    )


if __name__ == "__main__":
    main()
