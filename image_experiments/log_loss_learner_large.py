import argparse
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.nn.parallel import DataParallel
from tqdm.auto import tqdm
import json
from datetime import datetime
import os
import random

# ─── Dataset ─────────────────────────────────────────────────────────────
class PrefDataset(Dataset):
    def __init__(self, df):
        # df must have columns 'X1', 'X2', 'Y', 'true_r1', 'true_r2'
        self.X1 = torch.tensor(np.stack(df['X1'].values), dtype=torch.float32)
        self.X2 = torch.tensor(np.stack(df['X2'].values), dtype=torch.float32)
        # Y in {-1, +1}
        self.Y  = torch.tensor(df['Y'].values, dtype=torch.float32)
        self.true_r1 = torch.tensor(df['true_r1'].values, dtype=torch.float32)
        self.true_r2 = torch.tensor(df['true_r2'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return (
            self.X1[idx],
            self.X2[idx],
            self.Y[idx],
            self.true_r1[idx],
            self.true_r2[idx],
        )

# ─── Larger MLP for 2048-dim inputs ────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dims=[2048,1024, 512, 256, 128]):
        super().__init__()
        layers = []
        prev_dim = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(0.2),
            ]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ─── Training & Evaluation ────────────────────────────────────────────────
def train_pref(train_loader, model, optimizer, criterion, epochs=5):
    device = next(model.parameters()).device
    model.train()
    for epoch in range(epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for x1, x2, y, *_ in loop:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            y  = y.to(device, non_blocking=True)

            r1 = model(x1)
            r2 = model(x2)

            # map -1/+1 → 0/1
            target = ((y + 1) / 2).to(device)

            loss = criterion(2 * (r1 - r2), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())
    return model

@torch.no_grad()
def evaluate_pref(test_loader, model):
    device = next(model.parameters()).device
    model.eval()

    total, correct, mse_sum = 0, 0, 0.0
    total_policy_value = 0.0
    total_regret = 0.0
    
    for x1, x2, y, true_r1, true_r2 in tqdm(test_loader, desc="Evaluating", leave=False):
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        y  = y.to(device, non_blocking=True)
        true_r1 = true_r1.to(device, non_blocking=True)
        true_r2 = true_r2.to(device, non_blocking=True)

        r1 = model(x1)
        r2 = model(x2)

        preds = torch.sign(r1 - r2)
        correct += (preds == y).sum().item()
        total   += y.size(0)
        mse_sum += ((r1 - r2) - (true_r1 - true_r2)).pow(2).sum().item()

        # Calculate policy value: 
        # If prediction is +1, get reward true_r1, if prediction is -1, get reward true_r2
        policy_rewards = torch.where(preds > 0, true_r1, true_r2)
        total_policy_value += policy_rewards.sum().item()
        
        # Calculate regret: max reward - chosen reward
        max_rewards = torch.maximum(true_r1, true_r2)
        total_regret += (max_rewards - policy_rewards).sum().item()

    accuracy = correct / total
    mse = mse_sum / total
    policy_value = total_policy_value / total
    regret = total_regret / total
    
    return accuracy, mse, policy_value, regret

# ─── Utils: faster, incremental parquet loading ────────────────────────────
def load_train_data_parquet(folder):
    files = glob.glob(f"{folder}/*.parquet")
    dfs = []
    for path in tqdm(files, desc="Reading parquet files"):
        # only load the columns we need
        dfs.append(pd.read_parquet(
            path,
            columns=['X1','X2','Y','true_r1','true_r2'],
            engine='pyarrow'
        ))
    df = pd.concat(dfs, ignore_index=True)
    df['Y'] = -df['Y']  # negate labels if required
    return df

def save_model(model, args, metrics):
    """Save the trained model to a file."""
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Create a descriptive filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"models/log_loss_N{args.num_samples}_seed{args.seed}_{timestamp}.pt"
    
    # If the model is wrapped in DataParallel, get the underlying module
    if isinstance(model, DataParallel):
        model_to_save = model.module
    else:
        model_to_save = model
    
    # Save model state dict along with metadata
    model_info = {
        'state_dict': model_to_save.state_dict(),
        'args': vars(args),
        'metrics': metrics,
        'timestamp': timestamp
    }
    
    torch.save(model_info, filename)
    print(f"Model saved to {filename}")
    
    return filename

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_folder', required=True, help='Folder with parquet chunks')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--test_split', type=float, default=0.1, help='Fraction of data to use for testing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of training samples to use (None for all)')
    parser.add_argument('--test_size', type=int, default=10000, help='Number of test samples to use')
    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Using random seed: {args.seed}")

    # Load data
    df = load_train_data_parquet(args.train_folder)
    print(f"Total dataset size: {len(df)}")
    
    # Create full dataset
    full_dataset = PrefDataset(df)
    
    # Create indices for sampling
    all_indices = list(range(len(full_dataset)))
    
    # If num_samples specified, use it to sample the training set
    if args.num_samples is not None:
        # Ensure we have enough data
        if args.num_samples + args.test_size > len(full_dataset):
            print(f"Warning: Requested {args.num_samples} training samples and {args.test_size} test samples, but dataset only has {len(full_dataset)} samples")
            print(f"Using all available samples for training and testing")
            num_train = len(full_dataset) - args.test_size
        else:
            num_train = args.num_samples
        
        # Shuffle indices based on seed
        rng = np.random.RandomState(args.seed)
        rng.shuffle(all_indices)
        
        # Split into train and test
        train_indices = all_indices[:num_train]
        test_indices = all_indices[num_train:num_train+args.test_size]
        
        # Create subsets
        train_dataset = Subset(full_dataset, train_indices)
        test_dataset = Subset(full_dataset, test_indices)
        
        print(f"Using {len(train_dataset)} samples for training")
        print(f"Using {len(test_dataset)} samples for testing")
    else:
        # Use fraction-based split
        test_size = int(len(full_dataset) * args.test_split)
        train_size = len(full_dataset) - test_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], 
                                                generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch,
        num_workers=4, pin_memory=True
    )

    # Create model with best architecture from experiments
    model = MLP(in_dim=2048, hidden_dims=[2048, 1024, 512, 256]).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # Train
    train_pref(train_loader, model, optimizer, criterion, epochs=args.epochs)

    # Evaluate
    acc, mse, policy_value, regret = evaluate_pref(test_loader, model)
    print(f"\nFinal Results:")
    print(f"Preference model accuracy: {acc:.4f}")
    print(f"Preference model MSE:      {mse:.4f}")
    print(f"Policy value:              {policy_value:.4f}")
    print(f"Total regret:              {regret:.4f}")
    
    # Save the trained model
    metrics = {
        'accuracy': acc,
        'mse': mse,
        'policy_value': policy_value,
        'regret': regret
    }
    model_path = save_model(model, args, metrics)
    print(f"Model saved successfully to: {model_path}")

if __name__ == '__main__':
    main()
