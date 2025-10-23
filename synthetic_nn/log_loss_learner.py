# File: preference_training.py
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── Dataset ─────────────────────────────────────────────────────────────
class PrefDataset(Dataset):
    def __init__(self, df):
        # df must have columns 'X1', 'X2', 'Y' where X1, X2 are length-10 lists/arrays
        self.X1 = torch.tensor(np.stack(df['X1'].values), dtype=torch.float32)
        self.X2 = torch.tensor(np.stack(df['X2'].values), dtype=torch.float32)
        # Y in {{-1,+1}}
        self.Y  = torch.tensor(df['Y'].values, dtype=torch.float32)
        self.true_r1 = torch.tensor(df['true_r1'].values, dtype=torch.float32)
        self.true_r2 = torch.tensor(df['true_r2'].values, dtype=torch.float32)
    def __len__(self):
        return len(self.Y)
    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.Y[idx] , self.true_r1[idx], self.true_r2[idx]

# ─── Model ────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# ─── Training & Evaluation ─────────────────────────────────────────────────
def train_pref(train_loader, model, optimizer, criterion, a, epochs=5):
    model.train()
    for epoch in range(epochs):
        for x1, x2, y , *_ in train_loader:
            r1 = model(x1)
            r2 = model(x2)
            # BCEWithLogits: targets in {0,1}
            target = (y + 1) / 2  # map -1->0, +1->1
            loss = criterion(2*a*(r1 - r2), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

@torch.no_grad()
def evaluate_pref(test_loader, model):
    model.eval()
    total, correct = 0, 0
    mse = 0 
    regret = 0 
    for x1, x2, y, true_r1, true_r2 in test_loader:
        r1 = model(x1)
        r2 = model(x2)
        preds = torch.sign(r1 - r2)
        correct += (preds == y).sum().item()
        total += y.size(0)
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
        mse += ((r1-r2) - (true_r1 - true_r2)).pow(2).sum().item()
    
    return correct / total, mse / total, regret / total

# ─── Main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True, help='path to train pickle')
    parser.add_argument('--test',  required=True, help='path to test pickle')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch',  type=int, default=64)
    parser.add_argument('--threshold', type=float, default=1.0,
                        help='Barrier/threshold a for train_pref')
    args = parser.parse_args()

    # Load dataframes
    df_train = pd.read_pickle(args.train)
    df_test  = pd.read_pickle(args.test)

    # Datasets & loaders
    train_ds = PrefDataset(df_train)
    test_ds  = PrefDataset(df_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch)

    # Model, optimizer, loss
    model = MLP(in_dim=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # Train & eval
    train_pref(train_loader, model, optimizer, criterion, args.threshold, epochs=args.epochs)
    acc, mse, regret = evaluate_pref(test_loader, model)
    print(f"Preference model accuracy: {acc:.4f}")
    print(f"Preference model MSE: {mse:.4f}")
    print(f"Preference model regret: {regret:.4f}")

if __name__ == '__main__':
    main()

