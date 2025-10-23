#!/usr/bin/env python3
"""
Simple Dataset Enhancement Script with PickScore
Maps each row of the original dataset to create an enhanced version with additional columns.
"""

import os
import sys
import argparse
import torch
import numpy as np
from transformers import AutoProcessor, AutoModel
from datasets import load_dataset, load_from_disk, DatasetDict
from PIL import Image
from io import BytesIO
import logging
from datasets import Dataset
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhance_dataset.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Global model variables
model = None
processor = None
device = None

@torch.no_grad()
def process_row(row):    
    # Generate synthetic response time
    # Simulate decision time based on pickscore difference
    drift = row['pickscore_diff']
    x = 0.0  # starting point
    dt = 0.0001
    sqrt_dt = np.sqrt(dt)
    threshold = 1.0
    # Use a deterministic seed for reproducibility
    seed_str = f"{row['pickscore_diff']}_{row['pickscore_0']}_{row['pickscore_1']}_{row['__index_level_0__']}"
    seed = hash(seed_str) % (2**32)
    # Set the random seed for this example
    rng = np.random.RandomState(seed)
    steps = 0
    for steps in range(100000):  # Cap at max steps
        x += drift * dt + sqrt_dt * rng.randn()
        if  abs(x) >= threshold:
            break
    row['synth_response_time'] = steps * dt
    row['synth_label_0'] = 1 if x >= 0 else 0
    row['synth_label_1'] = 1 - row['synth_label_0']
    return row

def main():
    parser = argparse.ArgumentParser(description="Enhance dataset with PickScore")
    parser.add_argument("--dataset_path", type=str, default="data/pickscore_enhanced_fixed",
                        help="Path to dataset or HuggingFace dataset name")
    parser.add_argument("--output_dir", type=str, default="data/pickscore_brownian",
                        help="Directory to save enhanced dataset")
    parser.add_argument("--processor_path", type=str, default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                        help="Path to CLIP processor model")
    parser.add_argument("--model_path", type=str, default="yuvalkirstain/PickScore_v1",
                        help="Path to PickScore model")
    parser.add_argument("--num_proc", type=int, default=4,
                        help="Number of processes for dataset.map()")
    parser.add_argument("--load_from_disk", action="store_true",
                        help="Whether to load dataset from disk using load_from_disk")
    parser.add_argument("--sample_size", type=int, default=0,
                        help="Process only a subset of each split (0 for all)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading dataset from {args.dataset_path}")
    args.load_from_disk = True
    if args.load_from_disk:
        dataset = load_from_disk(args.dataset_path)
    else:
        dataset = load_dataset(args.dataset_path)
    
    # Load model and processor
    logger.info(f"Loading model from {args.model_path}")
    global model, processor, device

    
    # Create a dictionary to hold all enhanced splits
    enhanced_dict = {}
    
    # Process each split
    for split_name, split_data in dataset.items():
        logger.info(f"Processing {split_name} split with {len(split_data)} rows")
        
        # Optionally process only a subset
        if args.sample_size > 0:
            logger.info(f"Using sample of {args.sample_size} rows")
            split_data = split_data.select(range(min(args.sample_size, len(split_data))))
        
        # Process the dataset using map
        enhanced_split = split_data.map(
            process_row,
            batched=False,
            desc=f"Processing {split_name}",
            num_proc= 8
        )
        
        # Add to the dictionary of enhanced splits
        enhanced_dict[split_name] = enhanced_split
        
        # Also save individual split for direct access (optional)
        split_dir = os.path.join(args.output_dir, split_name)
        enhanced_split.save_to_disk(split_dir)
        logger.info(f"Saved {split_name} to {args.output_dir}/{split_name}")
    
    # Create a DatasetDict from all the enhanced splits
    enhanced_dataset = DatasetDict(enhanced_dict)
    
    # Save the full DatasetDict to the output directory
    enhanced_dataset.save_to_disk(args.output_dir)
    logger.info(f"Enhancement complete. Complete DatasetDict saved to {args.output_dir}")
    logger.info(f"You can now load the full dataset with: dataset = load_from_disk('{args.output_dir}')")

if __name__ == "__main__":
    main()