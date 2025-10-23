import argparse
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.nn.parallel import DataParallel
from tqdm.auto import tqdm
import random
import os
from datetime import datetime

# ─── Dataset for two-stage ───────────────────────────────────────────────────
class TwoStageDataset(Dataset):
    def __init__(self, df, stage='stage1', precomputed=None):
        # df: pandas DataFrame with X1, X2, Y, T, true_r1, true_r2
        self.X1 = torch.tensor(np.stack(df['X1'].values),      dtype=torch.float32)
        self.X2 = torch.tensor(np.stack(df['X2'].values),      dtype=torch.float32)
        self.Y  = torch.tensor(df['Y'].values,                  dtype=torch.float32)
        self.T  = torch.tensor(df['T'].values,                  dtype=torch.float32)
        self.true_r1 = torch.tensor(df['true_r1'].values,       dtype=torch.float32)
        self.true_r2 = torch.tensor(df['true_r2'].values,       dtype=torch.float32)
        # If stage2: attach CPU tensors for t_hat, r1, r2
        if stage == 'stage2' and precomputed is not None:
            self.t  = precomputed['t']   # already on CPU
            self.r1 = precomputed['r1']
            self.r2 = precomputed['r2']

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        X1, X2, Y, T, r1_true, r2_true = (
            self.X1[idx], self.X2[idx],
            self.Y[idx],  self.T[idx],
            self.true_r1[idx], self.true_r2[idx]
        )
        if hasattr(self, 't'):
            return X1, X2, Y, T, r1_true, r2_true, self.t[idx]
        else:
            return X1, X2, Y, T, r1_true, r2_true

# ─── MLP definition ──────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims=[1024,512,256]):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(0.2)
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ─── Stage1 training ─────────────────────────────────────────────────────────
def train_stage1(loader, model_r, model_t, opt_r, opt_t, bce, mse, epochs):
    device = next(model_r.parameters()).device
    model_r.train(); model_t.train()
    for ep in range(epochs):
        loop = tqdm(loader, desc=f"Stage1 [{ep+1}/{epochs}]", leave=False)
        for x1, x2, y, tval, *_ in loop:
            x1, x2, y, tval = (
                x1.to(device, non_blocking=True),
                x2.to(device, non_blocking=True),
                y.to(device,  non_blocking=True),
                tval.to(device, non_blocking=True),
            )

            # r-loss
            r1 = model_r(x1); r2 = model_r(x2)
            target = ((y+1)/2).to(device)
            loss_r = bce(2*(r1-r2), target)

            # t-prediction
            cat = torch.cat([x1, x2], dim=1)
            t_hat = model_t(cat)
            loss_t = mse(t_hat, tval)

            opt_r.zero_grad(); opt_t.zero_grad()
            loss_r.backward(); loss_t.backward()
            opt_r.step();    opt_t.step()

            loop.set_postfix(r=loss_r.item(), t=loss_t.item())

    return model_r, model_t

# ─── Precompute stage1 outputs (move to CPU) ─────────────────────────────────
@torch.no_grad()
def precompute(loader, model_r, model_t):
    device = next(model_r.parameters()).device
    model_r.eval(); model_t.eval()
    ts, r1s, r2s = [], [], []
    loop = tqdm(loader, desc="Precompute", leave=False)
    for x1, x2, *rest in loop:
        x1, x2 = x1.to(device), x2.to(device)
        cat = torch.cat([x1, x2], dim=1)
        t_hat = model_t(cat)
        r1 = model_r(x1)
        r2 = model_r(x2)
        ts.append(t_hat.cpu())
        r1s.append(r1.cpu())
        r2s.append(r2.cpu())
    return {
        't':  torch.cat(ts),
        'r1': torch.cat(r1s),
        'r2': torch.cat(r2s),
    }

# ─── Stage2 training ─────────────────────────────────────────────────────────
def train_stage2(loader, model_f, opt_f, epochs):
    device = next(model_f.parameters()).device
    model_f.train()
    for ep in range(epochs):
        loop = tqdm(loader, desc=f"Stage2 [{ep+1}/{epochs}]", leave=False)
        for X1, X2, Y, T, r1, r2, t_hat in loop:
            X1, X2, Y, T, r1, r2, t_hat = (
                X1.to(device), X2.to(device), Y.to(device),
                T.to(device), r1.to(device), r2.to(device),
                t_hat.to(device)
            )
            f1 = model_f(X1); f2 = model_f(X2)
            err = Y  - (f1 - f2)*t_hat
            loss = err.pow(2).mean()

            opt_f.zero_grad()
            loss.backward()
            opt_f.step()

            loop.set_postfix(loss=loss.item())
    return model_f

# ─── Stage2 evaluation ───────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_f(loader, model_f):
    device = next(model_f.parameters()).device
    model_f.eval()
    total = correct = 0
    mse_sum = 0.0
    total_policy_value = 0.0
    total_regret = 0.0
    loop = tqdm(loader, desc="Eval f", leave=False)
    for x1, x2, y, t, r1_true, r2_true, *_ in loop:
        x1, x2, y, t = x1.to(device), x2.to(device), y.to(device), t.to(device)
        r1_true, r2_true = r1_true.to(device), r2_true.to(device)

        f1 = model_f(x1); f2 = model_f(x2)
        preds = torch.sign(f1 - f2)
        correct += (preds == y).sum().item()
        total   += y.size(0)
        mse_sum += ((r1_true - r2_true) - (f1 - f2)).pow(2).sum().item()
        
        # Calculate policy value: 
        # If prediction is +1, get reward true_r1, if prediction is -1, get reward true_r2
        policy_rewards = torch.where(preds > 0, r1_true, r2_true)
        total_policy_value += policy_rewards.sum().item()
        # Calculate regret: max reward - chosen reward
        max_rewards = torch.maximum(r1_true, r2_true)
        total_regret += (max_rewards - policy_rewards).sum().item()
    
    accuracy = correct/total
    mse = mse_sum/total
    policy_value = total_policy_value/total
    regret = total_regret/total
    return accuracy, mse, policy_value, regret

# ─── Parquet loading with progress ───────────────────────────────────────────
def load_train_data_parquet(folder):
    files = glob.glob(f"{folder}/*.parquet")
    dfs = []
    for p in tqdm(files, desc="Read parquet"):
        dfs.append(pd.read_parquet(
            p,
            columns=['X1','X2','Y','T','true_r1','true_r2'],
            engine='pyarrow'
        ))
    df = pd.concat(dfs, ignore_index=True)
    df['Y'] = -df['Y']
    return df

# ─── Model saving ─────────────────────────────────────────────────────────────
def save_models(model_r, model_t, model_f, args, metrics):
    """Save the trained models to files."""
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Create a descriptive timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get base filename
    base_filename = f"nonorthogonal_N{args.num_samples}_seed{args.seed}_{timestamp}"
    
    # Helper function to save a single model
    def save_single_model(model, name):
        if isinstance(model, DataParallel):
            model_to_save = model.module
        else:
            model_to_save = model
            
        filename = f"models/{base_filename}_{name}.pt"
        torch.save({
            'state_dict': model_to_save.state_dict(),
            'args': vars(args),
            'metrics': metrics,
            'timestamp': timestamp,
            'model_type': name
        }, filename)
        return filename
    
    # Save all three models
    r_path = save_single_model(model_r, "r_model")
    t_path = save_single_model(model_t, "t_model")
    f_path = save_single_model(model_f, "f_model")
    
    print(f"R model saved to {r_path}")
    print(f"T model saved to {t_path}")
    print(f"F model saved to {f_path}")
    
    return r_path, t_path, f_path

# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_folder', required=True, help='Folder with parquet chunks')
    parser.add_argument('--epochs1', type=int, default=5)
    parser.add_argument('--epochs2', type=int, default=10)
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
    print(f"Device: {device}")
    print(f"Using random seed: {args.seed}")

    # -- load data --
    df = load_train_data_parquet(args.train_folder)
    print(f"Total dataset size: {len(df)}")
    
    # Create full dataset
    full_dataset = TwoStageDataset(df, stage='stage1')
    
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
    
    # -- stage1 DataLoader --
    loader1 = DataLoader(train_dataset, batch_size=args.batch,
                        shuffle=True, num_workers=4, pin_memory=True)

    # -- models & optimizers for stage1 --
    model_r = MLP(2048).to(device)
    model_t = MLP(4096).to(device)
    if torch.cuda.device_count() > 1:
        model_r = DataParallel(model_r)
        model_t = DataParallel(model_t)

    opt_r = torch.optim.Adam(model_r.parameters(), lr=1e-3)
    opt_t = torch.optim.Adam(model_t.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    # -- train stage1 --
    model_r, model_t = train_stage1(
        loader1, model_r, model_t,
        opt_r, opt_t, bce, mse, args.epochs1
    )

    # -- precompute for stage2 --
    # ===== FIX: Only use training data for stage2, not the whole dataset =====
    # Create a new dataset using only training data, not the full df
    if args.num_samples is not None:
        # If we're using train_indices from the subset, extract only those rows
        train_df = df.iloc[train_indices]
    else:
        # If using random_split, we need to extract the indices from train_dataset
        # But random_split doesn't expose indices directly, so we need to recreate the split
        train_size = len(full_dataset) - int(len(full_dataset) * args.test_split)
        # We'll recreate the train indices with the same seed to match the original split
        generator = torch.Generator().manual_seed(args.seed)
        train_indices, _ = torch.utils.data.random_split(
            range(len(full_dataset)), 
            [train_size, len(full_dataset) - train_size],
            generator=generator
        )
        train_indices = list(train_indices.indices)
        train_df = df.iloc[train_indices]
    
    print(f"Using {len(train_df)} samples for stage 2 training (was using {len(df)} incorrectly before)")
    
    ds_pre = TwoStageDataset(train_df, stage='stage1')
    loader_pre = DataLoader(ds_pre, batch_size=args.batch,
                            num_workers=4, pin_memory=True)
    pre = precompute(loader_pre, model_r, model_t)

    # -- stage2 DataLoader --
    ds2 = TwoStageDataset(train_df, stage='stage2', precomputed=pre)
    loader2 = DataLoader(ds2, batch_size=args.batch,
                         shuffle=True, num_workers=4, pin_memory=True)
    # =================================================================

    # -- model & optimizer for stage2 --
    model_f = MLP(2048).to(device)
    if torch.cuda.device_count() > 1:
        model_f = DataParallel(model_f)
    opt_f = torch.optim.Adam(model_f.parameters(), lr=1e-3)

    # -- train stage2 --
    model_f = train_stage2(loader2, model_f, opt_f, args.epochs2)

    # -- evaluate on test set --
    test_loader = DataLoader(test_dataset, batch_size=args.batch,
                            num_workers=4, pin_memory=True)
    acc, mse_val, policy_value, regret = evaluate_f(test_loader, model_f)

    print(f"Two-stage model accuracy: {acc:.4f}")
    print(f"Two-stage model MSE:      {mse_val:.4f}")
    print(f"Two-stage model policy value: {policy_value:.4f}")
    print(f"Two-stage model total regret: {regret:.4f}")
    
    # Save the trained models
    metrics = {
        'accuracy': acc,
        'mse': mse_val,
        'policy_value': policy_value,
        'regret': regret
    }
    r_path, t_path, f_path = save_models(model_r, model_t, model_f, args, metrics)
    print(f"All nonorthogonal models saved successfully!")

if __name__ == '__main__':
    main()
