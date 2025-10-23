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
            err = Y - (T - t_hat)*(r1 - r2) - (f1 - f2)*t_hat
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
def save_models(models, args, metrics):
    """Save the trained models to files."""
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Create a descriptive timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get base filename
    base_filename = f"orthogonal_crossfit_N{args.num_samples}_seed{args.seed}_{timestamp}"
    
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
    
    # Extract models from the dictionary
    model_r1, model_r2 = models['r1'], models['r2']
    model_t1, model_t2 = models['t1'], models['t2']
    model_f = models['f']
    
    # Save all models
    r1_path = save_single_model(model_r1, "r1_model")
    r2_path = save_single_model(model_r2, "r2_model")
    t1_path = save_single_model(model_t1, "t1_model")
    t2_path = save_single_model(model_t2, "t2_model")
    f_path = save_single_model(model_f, "f_model")
    
    print(f"R1 model saved to {r1_path}")
    print(f"R2 model saved to {r2_path}")
    print(f"T1 model saved to {t1_path}")
    print(f"T2 model saved to {t2_path}")
    print(f"F model saved to {f_path}")
    
    return r1_path, r2_path, t1_path, t2_path, f_path

# ─── Precompute with cross-fit ─────────────────────────────────────────────────
@torch.no_grad()
def precompute_crossfit(loader1, loader2, model_r1, model_t1, model_r2, model_t2):
    """
    Cross-fit precomputation: use model1 on data2, and model2 on data1
    Return combined results ordered as the original dataset
    """
    device = next(model_r1.parameters()).device
    model_r1.eval(); model_t1.eval()
    model_r2.eval(); model_t2.eval()
    
    # First fold - using model2 on data1
    ts_fold1, r1s_fold1, r2s_fold1 = [], [], []
    loop = tqdm(loader1, desc="Precompute Fold 1", leave=False)
    for x1, x2, *rest in loop:
        x1, x2 = x1.to(device), x2.to(device)
        cat = torch.cat([x1, x2], dim=1)
        t_hat = model_t2(cat)  # Using model2 on data1
        r1 = model_r2(x1)      # Using model2 on data1
        r2 = model_r2(x2)      # Using model2 on data1
        ts_fold1.append(t_hat.cpu())
        r1s_fold1.append(r1.cpu())
        r2s_fold1.append(r2.cpu())
    
    # Second fold - using model1 on data2
    ts_fold2, r1s_fold2, r2s_fold2 = [], [], []
    loop = tqdm(loader2, desc="Precompute Fold 2", leave=False)
    for x1, x2, *rest in loop:
        x1, x2 = x1.to(device), x2.to(device)
        cat = torch.cat([x1, x2], dim=1)
        t_hat = model_t1(cat)  # Using model1 on data2
        r1 = model_r1(x1)      # Using model1 on data2
        r2 = model_r1(x2)      # Using model1 on data2
        ts_fold2.append(t_hat.cpu())
        r1s_fold2.append(r1.cpu())
        r2s_fold2.append(r2.cpu())
    
    # Concatenate results in the original order (fold1 first, then fold2)
    return {
        't': torch.cat([torch.cat(ts_fold1), torch.cat(ts_fold2)]),
        'r1': torch.cat([torch.cat(r1s_fold1), torch.cat(r1s_fold2)]),
        'r2': torch.cat([torch.cat(r2s_fold1), torch.cat(r2s_fold2)]),
    }

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

    # ────────────────────────────────────────────────────────────────────────────
    # CROSS FITTING: Split train dataset into two equal parts
    # ────────────────────────────────────────────────────────────────────────────
    if isinstance(train_dataset, Subset):
        # For Subset, we need to split the indices
        train_indices = train_dataset.indices
        half_size = len(train_indices) // 2
        train_indices1 = train_indices[:half_size]
        train_indices2 = train_indices[half_size:]
        train_dataset1 = Subset(full_dataset, train_indices1)
        train_dataset2 = Subset(full_dataset, train_indices2)
        
        # Also save the corresponding dataframes for stage 2
        train_df1 = df.iloc[train_indices1]
        train_df2 = df.iloc[train_indices2]
        train_df_combined = pd.concat([train_df1, train_df2], ignore_index=False)
    else:
        # For random_split dataset, recreate with a new split
        half_size = len(train_dataset) // 2
        remainder = len(train_dataset) - 2 * half_size  # In case of odd number
        train_dataset1, train_dataset2 = random_split(
            train_dataset, 
            [half_size, half_size + remainder],
            generator=torch.Generator().manual_seed(args.seed + 1)
        )
        
        # For stage 2, we need to extract the dataframes
        # Recreate the training indices for both parts
        generator = torch.Generator().manual_seed(args.seed)
        train_test_split = random_split(
            range(len(full_dataset)), 
            [train_size, test_size],
            generator=generator
        )
        train_indices = list(train_test_split[0].indices)
        
        generator = torch.Generator().manual_seed(args.seed + 1)
        train_split = random_split(
            range(len(train_indices)),
            [half_size, half_size + remainder],
            generator=generator
        )
        
        indices1 = [train_indices[i] for i in train_split[0].indices]
        indices2 = [train_indices[i] for i in train_split[1].indices]
        
        train_df1 = df.iloc[indices1]
        train_df2 = df.iloc[indices2]
        train_df_combined = pd.concat([train_df1, train_df2], ignore_index=False)
    
    print(f"Cross-fitting: Split training data into two parts: {len(train_dataset1)} and {len(train_dataset2)} samples")
    
    # Create DataLoaders for the two parts
    loader1 = DataLoader(train_dataset1, batch_size=args.batch,
                        shuffle=True, num_workers=4, pin_memory=True)
    loader2 = DataLoader(train_dataset2, batch_size=args.batch,
                        shuffle=True, num_workers=4, pin_memory=True)

    # ────────────────────────────────────────────────────────────────────────────
    # Train two separate stage1 models on each half of the data
    # ────────────────────────────────────────────────────────────────────────────
    print("Training first stage models on fold 1...")
    model_r1 = MLP(2048).to(device)
    model_t1 = MLP(4096).to(device)
    if torch.cuda.device_count() > 1:
        model_r1 = DataParallel(model_r1)
        model_t1 = DataParallel(model_t1)

    opt_r1 = torch.optim.Adam(model_r1.parameters(), lr=1e-3)
    opt_t1 = torch.optim.Adam(model_t1.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    # Train on first half
    model_r1, model_t1 = train_stage1(
        loader1, model_r1, model_t1,
        opt_r1, opt_t1, bce, mse, args.epochs1
    )
    
    print("Training first stage models on fold 2...")
    model_r2 = MLP(2048).to(device)
    model_t2 = MLP(4096).to(device)
    if torch.cuda.device_count() > 1:
        model_r2 = DataParallel(model_r2)
        model_t2 = DataParallel(model_t2)

    opt_r2 = torch.optim.Adam(model_r2.parameters(), lr=1e-3)
    opt_t2 = torch.optim.Adam(model_t2.parameters(), lr=1e-3)

    # Train on second half
    model_r2, model_t2 = train_stage1(
        loader2, model_r2, model_t2,
        opt_r2, opt_t2, bce, mse, args.epochs1
    )

    # ────────────────────────────────────────────────────────────────────────────
    # Precompute stage2 features using cross-fitting
    # (model1 on data2 and model2 on data1)
    # ────────────────────────────────────────────────────────────────────────────
    print("Cross-fitting: Computing nuisances using opposite-fold models...")
    
    # Create DataLoaders for precomputation (no shuffling to maintain order)
    loader_pre1 = DataLoader(train_dataset1, batch_size=args.batch,
                           shuffle=False, num_workers=4, pin_memory=True)
    loader_pre2 = DataLoader(train_dataset2, batch_size=args.batch,
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # Cross-fit: Use model2 for fold1 and model1 for fold2
    pre = precompute_crossfit(loader_pre1, loader_pre2, model_r1, model_t1, model_r2, model_t2)
    
    # ────────────────────────────────────────────────────────────────────────────
    # Train stage2 on the combined dataset with cross-fit nuisances
    # ────────────────────────────────────────────────────────────────────────────
    print("Training second stage model on combined dataset with cross-fit nuisances...")
    ds2 = TwoStageDataset(train_df_combined, stage='stage2', precomputed=pre)
    loader2 = DataLoader(ds2, batch_size=args.batch,
                         shuffle=True, num_workers=4, pin_memory=True)

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

    print(f"Cross-fit two-stage model accuracy: {acc:.4f}")
    print(f"Cross-fit two-stage model MSE:      {mse_val:.4f}")
    print(f"Cross-fit two-stage model policy value: {policy_value:.4f}")
    print(f"Cross-fit two-stage model total regret: {regret:.4f}")
    
    # Save the trained models
    metrics = {
        'accuracy': acc,
        'mse': mse_val,
        'policy_value': policy_value,
        'regret': regret
    }
    
    models = {
        'r1': model_r1,
        'r2': model_r2,
        't1': model_t1,
        't2': model_t2,
        'f': model_f
    }
    
    save_models(models, args, metrics)
    print(f"All cross-fit orthogonal models saved successfully!")

if __name__ == '__main__':
    main() 