import os

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.dataloader import LoadDataset


def get_train_test_val_dataloaders(batch_size=32):
    df = pl.read_parquet("data/data.parquet").to_pandas()
    # First split → train + temp
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

    # Second split temp → val + test
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Convert back to Polars DataFrames
    train_df = pl.from_pandas(train_df)
    val_df = pl.from_pandas(val_df)
    test_df = pl.from_pandas(test_df)

    train_loader, val_loader, test_loader = set_up_dataloaders(
        train_df, val_df, test_df, batch_size=batch_size
    )

    return train_loader, val_loader, test_loader


def set_up_dataloaders(train_df, val_df, test_df, batch_size=32):
    train_dataset = LoadDataset(train_df)
    val_dataset = LoadDataset(val_df)
    test_dataset = LoadDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def run_training_and_testing(
    model, device, optimizer, loss_fn, epochs, train_loader, val_loader, test_loader
):
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        print(f"Epoch {epoch + 1}/{epochs}")

        # Training loop with tqdm
        for X, y, ID in tqdm(train_loader, desc="Training Batches", leave=False):
            X = X.to(device)
            y = y.to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.size(0)
            predicted = (logits > 0.5).float()
            correct += (predicted == y).sum().item()
            total += y.size(0)

        avg_train_loss = total_loss / total
        train_accuracy = correct / total
        train_losses.append(avg_train_loss)
        train_accs.append(train_accuracy)

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y, ID in tqdm(val_loader, desc="Validation Batches", leave=False):
                X = X.to(device)
                y = y.to(device).unsqueeze(1)

                logits = model(X)
                loss = loss_fn(logits, y)
                val_loss += loss.item() * X.size(0)

                predicted = (logits > 0.5).float()
                correct += (predicted == y).sum().item()
                total += y.size(0)

        avg_val_loss = val_loss / total
        val_accuracy = correct / total
        val_losses.append(avg_val_loss)
        val_accs.append(val_accuracy)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}"
        )

    # Final testing
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X, y, ID in tqdm(test_loader, desc="Testing Batches"):
            X = X.to(device)
            y = y.to(device).unsqueeze(1)

            logits = model(X)
            predicted = (logits > 0.5).float()

            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    return train_losses, train_accs, val_losses, val_accs, all_preds, all_labels


def plot_confusion_matrix(all_preds, all_labels):
    """
    Plots a confusion matrix with total accuracy in the title.
    """
    cm = confusion_matrix(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix | Accuracy: {acc:.4f}")
    plt.savefig("graphs/confusion_matrix.png")


def plot_training_history(
    train_losses, val_losses, train_accs, val_accs, save_path="graphs/training_plot.png"
):
    """
    Plots training/validation loss and accuracy, then saves the figure in `graph` folder.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss plot
    axes[0].plot(train_losses, label="Train Loss")
    axes[0].plot(val_losses, label="Val Loss")
    axes[0].set_title("Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy plot
    axes[1].plot(train_accs, label="Train Acc")
    axes[1].plot(val_accs, label="Val Acc")
    axes[1].set_title("Accuracy over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
