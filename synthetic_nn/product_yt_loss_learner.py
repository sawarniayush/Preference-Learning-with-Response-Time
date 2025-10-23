import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── Dataset for two-stage ───────────────────────────────────────────────────
class TwoStageDataset(Dataset):
    def __init__(self, df, stage='stage1', precomputed=None):
        # df: pandas DataFrame with X1, X2, Y, T
        self.X1 = torch.tensor(np.stack(df['X1'].values), dtype=torch.float32)
        self.X2 = torch.tensor(np.stack(df['X2'].values), dtype=torch.float32)
        self.Y  = torch.tensor(df['Y'].values, dtype=torch.float32)
        self.T  = torch.tensor(df['T'].values, dtype=torch.float32)
        self.true_r1 = torch.tensor(df['true_r1'].values, dtype=torch.float32)
        self.true_r2 = torch.tensor(df['true_r2'].values, dtype=torch.float32)
    def __len__(self): return len(self.Y)
    def __getitem__(self, idx):
        X1, X2, Y, T, true_r1, true_r2 = self.X1[idx], self.X2[idx], self.Y[idx], self.T[idx], self.true_r1[idx], self.true_r2[idx]
        return X1, X2, Y, T, true_r1, true_r2

# ─── Simple MLP ─────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim, hidden1=32, hidden2 = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(),             
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),              
            nn.Linear(hidden2, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# ─── Train Stage1: r and t ──────────────────────────────────────────────────
def train_stage1(loader, model_t, model_y, opt_t, opt_y, mse, epochs=5):
    model_t.train(); model_y.train()
    for ep in range(epochs):
        for x1, x2, y, tval, *_ in loader:
            # mse for t and y
            t_hat = model_t(torch.cat([x1, x2], dim=1))
            y_hat = model_y(torch.cat([x1, x2], dim=1))
            loss_y = mse(y_hat, y)
            loss_t = mse(t_hat, tval)
            opt_t.zero_grad()
            opt_y.zero_grad()
            loss_t.backward()
            loss_y.backward()
            opt_t.step()
            opt_y.step()
    return model_t, model_y

# ─── Evaluate r(x) = y(x)/t(x) ─────────────────────────────────────────────
@torch.no_grad()
def evaluate_r(loader, model_t, model_y):
    model_t.eval(); model_y.eval()
    total, correct = 0, 0
    mse = 0
    for x1, x2, y, t, true_r1, true_r2 in loader:
        # Compute y(x) and t(x)
        y = model_y(torch.cat([x1, x2], dim=1))  # Using x1 twice since we need single input
        t = model_t(torch.cat([x1, x1], dim=1))

        
        # Compute r(x) = y(x)/t(x)
        r = y / t
        
        # Evaluate preferences
        preds = torch.sign(r)
        correct += (preds == y).sum().item()
        total += y.size(0)
        
        # Compute MSE with true r values
        mse += (((true_r1 - true_r2) - (r))**2).sum().item()
    
    return correct / total, mse/total

# ─── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--test',  required=True)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--epochs1', type=int, default=5)
    args = parser.parse_args()
    
    df = pd.read_pickle(args.train)
    ds1 = TwoStageDataset(df, stage='stage1')
    loader1 = DataLoader(ds1, batch_size=args.batch, shuffle=True)

    # Models & opts
    model_t = MLP(20)
    model_y = MLP(20)
    opt_t = torch.optim.Adam(model_t.parameters(), lr=1e-3)
    opt_y = torch.optim.Adam(model_y.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    # Stage1 train
    model_t, model_y = train_stage1(loader1, model_t, model_y, opt_t, opt_y, mse, epochs=args.epochs1)

    # Evaluate on test set
    df_test = pd.read_pickle(args.test)
    test_ds = TwoStageDataset(df_test, stage='stage1')
    test_loader = DataLoader(test_ds, batch_size=args.batch)
    acc, mse = evaluate_r(test_loader, model_t, model_y)
    print(f"Model accuracy: {acc:.4f}")
    print(f"Model MSE: {mse:.4f}")

if __name__ == '__main__':
    main()
