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
        # For stage2, precomputed should be dict with 't', 'r1', 'r2'
        if stage=='stage2' and precomputed is not None:
            self.t  = precomputed['t']
            self.r1 = precomputed['r1']
            self.r2 = precomputed['r2']
            self.y = precomputed['y']
    def __len__(self): return len(self.Y)
    def __getitem__(self, idx):
        X1, X2, Y, T,  true_r1, true_r2 = self.X1[idx], self.X2[idx], self.Y[idx], self.T[idx] ,self.true_r1[idx], self.true_r2[idx]
        if hasattr(self, 't'):
            return X1, X2, Y, T,  true_r1, true_r2, self.t[idx], self.y[idx]
        return X1, X2, Y, T , true_r1, true_r2

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

class TMLP(nn.Module):
    def __init__(self, in_dim, hidden1=32, hidden2=16):
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
def train_stage1(loader, model_r, model_t, model_y, opt_r, opt_t,opt_y, bce, mse, epochs=5):
    model_r.train(); model_t.train(); model_y.train()
    for ep in range(epochs):
        for x1, x2, y, tval, *_ in loader:
            # logistic for r
            r1 = model_r(x1); r2 = model_r(x2)
            target = (y + 1)/2
            loss_r = bce(2*(r1 - r2), target)
            # mse for t
            t_hat = model_t(torch.cat([x1, x2], dim=1))
            y_hat = model_y(torch.cat([x1, x2], dim=1))
            loss_y = mse(y_hat, y)
            loss_t = mse(t_hat, tval)
            opt_r.zero_grad(); opt_t.zero_grad()
            (loss_r).backward()
            (loss_t).backward()
            (loss_y).backward()
            opt_r.step(); opt_t.step(); opt_y.step()
    return model_r, model_t, model_y

# ─── Precompute on half ─────────────────────────────────────────────────────
@torch.no_grad()
def precompute(loader, model_r, model_t,model_y):
    model_r.eval(); model_t.eval()
    t_list, y_list, r1_list, r2_list = [], [], [],[]
    for x1, x2, _, _, *_ in loader:
        t_hat = model_t(torch.cat([x1, x2], dim=1))
        y_hat = model_y(torch.cat([x1, x2], dim=1))
        r1 = model_r(x1); r2 = model_r(x2)
        t_list.append(t_hat); r1_list.append(r1); r2_list.append(r2), y_list.append(y_hat)
    t_all  = torch.cat(t_list)
    r1_all = torch.cat(r1_list)
    r2_all = torch.cat(r2_list)
    y_all = torch.cat(y_list)
    return {'t': t_all, 'r1': r1_all, 'r2': r2_all, 'y': y_all}

# ─── Train Stage2: f with custom loss ───────────────────────────────────────
def train_stage2(loader, model_f, optimizer, epochs=5):
    model_f.train()
    for ep in range(epochs):
        for X1, X2, Y, T, r1, r2, t_hat, y_hat in loader:
            f1 = model_f(X1); f2 = model_f(X2)
            err = Y - (T - t_hat)*(torch.tanh(r1 - r2))/t_hat - (f1 - f2)*t_hat
            loss = (err**2).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    return model_f

# ─── Evaluate f on preferences ─────────────────────────────────────────────
@torch.no_grad()
def evaluate_f(loader, model_f, a=1.0):
    model_f.eval()
    total, correct = 0, 0
    mse = 0
    regret = 0
    for x1, x2, y, t, true_r1, true_r2, *_ in loader:
        f1 = model_f(x1); f2 = model_f(x2)
        preds = torch.sign(f1 - f2)
        correct += (preds == y).sum().item()
        total += y.size(0)
        mse += (((true_r1 - true_r2) - (f1-f2)*a)**2).sum().item()
        
        # Calculate regret: select the lower reward when prediction is wrong
        # When preds == 1 but y == -1, we wrongly chose x1 over x2
        # When preds == -1 but y == 1, we wrongly chose x2 over x1
        wrong_choices = (preds != y)
        choose_x1 = (preds == 1)
        choose_x2 = (preds == -1)
        
        # Calculate regret for each prediction
        # For cases where we chose x1 (preds == 1) and were wrong (y == -1)
        # regret = true_r2 - true_r1 (the optimal minus what we chose)
        # For cases where we chose x2 (preds == -1) and were wrong (y == 1)
        # regret = true_r1 - true_r2 (the optimal minus what we chose)
        regret_values = torch.zeros_like(y, dtype=torch.float32)
        regret_values[wrong_choices & choose_x1] = true_r2[wrong_choices & choose_x1] - true_r1[wrong_choices & choose_x1]
        regret_values[wrong_choices & choose_x2] = true_r1[wrong_choices & choose_x2] - true_r2[wrong_choices & choose_x2]
        
        regret += regret_values.sum().item()
    
    return correct / total, mse / total, regret / total

# ─── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--test',  required=True)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--epochs1', type=int, default=5)
    parser.add_argument('--epochs2', type=int, default=10)
    parser.add_argument('--type', type=str, default='tr')
    parser.add_argument('--threshold', type=float, default=1.0,
                        help='Barrier/threshold a for evaluation')
    args = parser.parse_args()
    df = pd.read_pickle(args.train)
    # split first-half and second-half
    N = len(df); half = int(N * 0.5)
    # df1 = df.iloc[:half].reset_index(drop=True)
    df1 = df
    # df2 = df.iloc[half:].reset_index(drop=True)
    df2 = df

    # Stage1 data
    ds1 = TwoStageDataset(df1, stage='stage1')
    loader1 = DataLoader(ds1, batch_size=args.batch, shuffle=True)

    # Models & opts
    model_r = MLP(10); model_t = MLP(20); model_y =MLP(20)
    opt_r = torch.optim.Adam(model_r.parameters(), lr=1e-3)
    opt_t = torch.optim.Adam(model_t.parameters(), lr=1e-3)
    opt_y = torch.optim.Adam(model_t.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss(); mse = nn.MSELoss()

    # Stage1 train
    model_r, model_t, model_y = train_stage1(loader1, model_r, model_t,model_y, opt_r, opt_t,opt_y, bce, mse, epochs=args.epochs1)

    # Precompute on second half
    ds2_base = TwoStageDataset(df2, stage='stage1')
    loader2_base = DataLoader(ds2_base, batch_size=args.batch)
    pre = precompute(loader2_base, model_r, model_t, model_y)

    # Stage2 data with precomputed
    ds2 = TwoStageDataset(df2, stage='stage2', precomputed=pre)
    loader3 = DataLoader(ds2, batch_size=args.batch, shuffle=True)

    # Train f
    model_f = MLP(10)
    opt_f = torch.optim.Adam(model_f.parameters(), lr=1e-3)
    model_f = train_stage2(loader3, model_f, opt_f, epochs=args.epochs2)

    # Evaluate on test set
    df_test = pd.read_pickle(args.test)
    test_ds = TwoStageDataset(df_test, stage='stage1')
    test_loader = DataLoader(test_ds, batch_size=args.batch)
    acc, mse, regret = evaluate_f(test_loader, model_f, args.threshold)
    print(f"Two-stage model (f) accuracy: {acc:.4f}")
    print(f"Two-stage model (f) mse: {mse:.4f}")
    print(f"Two-stage model (f) regret: {regret:.4f}")

if __name__ == '__main__':
    main()
