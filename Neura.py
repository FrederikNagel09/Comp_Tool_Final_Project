import numpy as np
from sklearn.model_selection import train_test_split
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataloader.dataloader import LoadDataset
import polars as pl
from sklearn.model_selection import train_test_split
from tqdm import tqdm

print("Loading data...")
df = pl.read_parquet("data/data.parquet").to_pandas()
# First split → train + temp
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

# Second split temp → val + test
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Convert back to Polars DataFrames
train_df = pl.from_pandas(train_df)
val_df = pl.from_pandas(val_df)
test_df = pl.from_pandas(test_df)
print("Data loaded, and split into train, val, test.")


train_dataset = LoadDataset(train_df)
val_dataset = LoadDataset(val_df)
test_dataset = LoadDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("DataLoaders created.")


def accuracy(loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y, ID in loader:
            y = y.unsqueeze(1)
            preds = (model(X) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(392, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


print("Starting training...")
model = NeuralNetwork()
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())
epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0

    print(f"Epoch {epoch + 1}/{epochs}")

    # tqdm INSIDE the epoch (batch loop)
    for X, y, ID in tqdm(train_loader, desc="Training Batches", leave=False):
        y = y.unsqueeze(1)

        logits = model(X)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # ----- ACCURACIES -----
    train_acc = accuracy(train_loader)
    val_acc = accuracy(val_loader)

    print(
        f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
    )


test_acc = accuracy(test_loader)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")
