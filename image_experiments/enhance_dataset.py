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

def open_image(image_bytes):
    """Convert image bytes to PIL Image object."""
    if isinstance(image_bytes, bytes):
        image = Image.open(BytesIO(image_bytes))
    else:
        image = image_bytes
    return image.convert('RGB')

@torch.no_grad()
def process_row(row):
    """Process a single row of the dataset with PickScore."""
    global model, processor, device
    # Open images and get caption
    images = [open_image(row['jpg_0']), open_image(row['jpg_1'])]
    prompt = row['caption']
    
    # Preprocess images and text
    image_inputs = processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt"
    ).to(device)
    
    text_inputs = processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt"
    ).to(device)
    
    # Compute embeddings
    image_embs = model.get_image_features(**image_inputs)
    image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
    text_embs = model.get_text_features(**text_inputs)
    text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
    
    # Calculate scores
    scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
    
    # Extract individual scores
    pickscore_0 = scores[0].item()
    pickscore_1 = scores[1].item()
    
    # Calculate difference and softmax
    diff = pickscore_0 - pickscore_1
    softmax_probs = torch.softmax(2*scores, dim=0)
    softmax_prob_0 = softmax_probs[0].item()
    
    # Add results to row
    row['pickscore_0'] = pickscore_0
    row['pickscore_1'] = pickscore_1
    row['pickscore_diff'] = diff
    row['pickscore_softmax_0'] = softmax_prob_0
    
    # Generate synthetic labels based on softmax probability
    choice = 0 if np.random.random() < softmax_prob_0 else 1
    row['synth_label_0'] = 1 if choice == 0 else 0
    row['synth_label_1'] = 0 if choice == 0 else 1
    
    # Generate synthetic response time via a simple drift-diffusion process
    drift = diff
    dt = 0.001
    sqrt_dt = np.sqrt(dt)
    threshold = 1.0
    max_steps = 100_000
    x = 0.0
    t = 0.0

    rng = np.random.default_rng()

    for _ in range(max_steps):
        x += drift * dt + sqrt_dt * rng.standard_normal()
        t += dt
        if x >= threshold:
            row['synth_label_0'] = 1
            row['synth_label_1'] = 0
            row['synth_response_time'] = t
            break
        if x <= -threshold:
            row['synth_label_0'] = 0
            row['synth_label_1'] = 1
            row['synth_response_time'] = t
            break
    else:
        # If the threshold was never hit, fall back to a max-time observation
        row['synth_response_time'] = t
    
    return row

def main():
    parser = argparse.ArgumentParser(description="Enhance dataset with PickScore")
    parser.add_argument("--dataset_path", type=str, default="data/pickscore_raw",
                        help="Path to dataset or HuggingFace dataset name")
    parser.add_argument("--output_dir", type=str, default="data/pickscore_enhanced",
                        help="Directory to save enhanced dataset")
    parser.add_argument("--processor_path", type=str, default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                        help="Path to CLIP processor model")
    parser.add_argument("--model_path", type=str, default="yuvalkirstain/PickScore_v1",
                        help="Path to PickScore model")
    parser.add_argument("--num_proc", type=int, default=1,
                        help="Number of processes for dataset.map()")
    parser.add_argument("--load_from_disk", action="store_true",
                        help="Whether to load dataset from disk using load_from_disk (auto-detected if the path exists)")
    parser.add_argument("--sample_size", type=int, default=0,
                        help="Process only a subset of each split (0 for all)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Auto-detect loading mode if not explicitly provided
    if not args.load_from_disk and os.path.isdir(args.dataset_path):
        logger.info("Detected local directory %s; using load_from_disk", args.dataset_path)
        args.load_from_disk = True

    # Load dataset
    logger.info(f"Loading dataset from {args.dataset_path}")
    if args.load_from_disk:
        dataset = load_from_disk(args.dataset_path)
    else:
        dataset = load_dataset(args.dataset_path)
    
    # Load model and processor
    logger.info(f"Loading model from {args.model_path}")
    global model, processor, device
    processor = AutoProcessor.from_pretrained(args.processor_path)
    model = AutoModel.from_pretrained(args.model_path).eval()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    logger.info(f"Using device: {device}")
    
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
            desc=f"Processing {split_name}",
            num_proc=args.num_proc
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
