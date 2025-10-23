#!/usr/bin/env python3
"""
Utility to convert embedding pickles produced by ``generate_embeddings.py`` into
PyArrow/Parquet shards that the learning scripts can stream efficiently.

Example
-------
python prepare_training_data.py \\
    --input_dir data/embeddings \\
    --output_dir data/training_chunks \\
    --rows_per_chunk 50000
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Iterable, List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm


def discover_pickles(input_dir: str, pattern: str) -> List[str]:
    """Return sorted list of pickle files to convert."""
    search_pattern = os.path.join(input_dir, pattern)
    files = sorted(glob.glob(search_pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern {search_pattern!r}")
    return files


def infer_split_name(path: str) -> str:
    """Infer split name from the pickle file name."""
    basename = os.path.basename(path)
    if basename.endswith("_embeddings.pkl"):
        return basename.replace("_embeddings.pkl", "")
    if basename.endswith(".pkl"):
        return basename[:-4]
    return os.path.splitext(basename)[0]


def dataframe_to_parquet_chunks(
    df: pd.DataFrame,
    output_dir: str,
    split_name: str,
    rows_per_chunk: int,
) -> Iterable[str]:
    """
    Yield Parquet file paths generated from the dataframe.
    """
    os.makedirs(output_dir, exist_ok=True)

    total_rows = len(df)
    if total_rows == 0:
        return []

    # Ensure consistent dtypes
    df = df.copy()
    if "pair_index" in df.columns:
        df["pair_index"] = df["pair_index"].astype(np.int64)
        df.sort_values("pair_index", inplace=True)
        df.reset_index(drop=True, inplace=True)
    else:
        df.reset_index(drop=True, inplace=True)

    df["Y"] = df["Y"].astype(np.int8)
    df["T"] = df["T"].astype(np.float32)
    df["true_r1"] = df["true_r1"].astype(np.float32)
    df["true_r2"] = df["true_r2"].astype(np.float32)

    produced_files: List[str] = []
    for chunk_idx, start in enumerate(range(0, total_rows, rows_per_chunk)):
        stop = min(start + rows_per_chunk, total_rows)
        chunk = df.iloc[start:stop]

        x1 = np.stack(chunk["X1"].to_numpy()).astype(np.float32, copy=False)
        x2 = np.stack(chunk["X2"].to_numpy()).astype(np.float32, copy=False)

        list_size = x1.shape[1]

        table = pa.table(
            {
                "pair_index": pa.array(
                    chunk["pair_index"].to_numpy(dtype=np.int64, copy=False)
                    if "pair_index" in chunk.columns
                    else np.arange(start, stop, dtype=np.int64)
                ),
                "X1": pa.FixedSizeListArray.from_arrays(
                    pa.array(x1.reshape(-1), type=pa.float32()), list_size
                ),
                "X2": pa.FixedSizeListArray.from_arrays(
                    pa.array(x2.reshape(-1), type=pa.float32()), list_size
                ),
                "Y": pa.array(chunk["Y"].to_numpy(), type=pa.int8()),
                "T": pa.array(chunk["T"].to_numpy(), type=pa.float32()),
                "true_r1": pa.array(chunk["true_r1"].to_numpy(), type=pa.float32()),
                "true_r2": pa.array(chunk["true_r2"].to_numpy(), type=pa.float32()),
            }
        )

        output_path = os.path.join(
            output_dir, f"{split_name}_chunk{chunk_idx:03d}.parquet"
        )
        pq.write_table(table, output_path, compression="snappy")
        produced_files.append(output_path)

    return produced_files


def convert_file(path: str, output_dir: str, rows_per_chunk: int) -> List[str]:
    split_name = infer_split_name(path)
    df = pd.read_pickle(path)
    return list(
        dataframe_to_parquet_chunks(
            df=df,
            output_dir=os.path.join(output_dir, split_name),
            split_name=split_name,
            rows_per_chunk=rows_per_chunk,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir",
        default="data/embeddings",
        help="Directory that holds *_embeddings.pkl files (default: %(default)s)",
    )
    parser.add_argument(
        "--pattern",
        default="*_embeddings.pkl",
        help="Glob pattern relative to --input_dir (default: %(default)s)",
    )
    parser.add_argument(
        "--output_dir",
        default="data/training_chunks",
        help="Where to write the Parquet shards (default: %(default)s)",
    )
    parser.add_argument(
        "--rows_per_chunk",
        type=int,
        default=50_000,
        help="Number of examples per Parquet shard (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pickle_paths = discover_pickles(args.input_dir, args.pattern)
    all_outputs: List[str] = []

    for path in tqdm(pickle_paths, desc="Converting pickles"):
        outputs = convert_file(path, args.output_dir, args.rows_per_chunk)
        all_outputs.extend(outputs)

    print("Generated the following Parquet shards:")
    for out_path in all_outputs:
        print(f"  - {out_path}")


if __name__ == "__main__":
    main()
