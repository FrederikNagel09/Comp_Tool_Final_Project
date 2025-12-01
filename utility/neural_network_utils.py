import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def run_training_and_testing(
    model: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
) -> tuple[list[float], list[float], list[float], list[float], list[float], list[float]]:
    """Train model and evaluate on test set, returning metrics and predictions."""
    train_losses, train_accs, val_losses, val_accs = train_model(
        model, device, optimizer, loss_fn, epochs, train_loader, val_loader
    )

    all_preds, all_labels = evaluate_model(model, device, test_loader)

    return train_losses, train_accs, val_losses, val_accs, all_preds, all_labels


def train_model(
    model: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Train model for specified epochs and track training/validation metrics."""
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        train_loss, train_acc = train_one_epoch(model, device, optimizer, loss_fn, train_loader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        val_loss, val_acc = validate_one_epoch(model, device, loss_fn, val_loader)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    return train_losses, train_accs, val_losses, val_accs


def train_one_epoch(
    model: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    train_loader: DataLoader,
) -> tuple[float, float]:
    """Execute one training epoch and return average loss and accuracy."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y, _ in tqdm(train_loader, desc="Training Batches", leave=False):
        x = x.to(device)
        y = y.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        predicted = (logits > 0.5).float()
        correct += (predicted == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate_one_epoch(
    model: torch.nn.Module,
    device: torch.device,
    loss_fn: torch.nn.Module,
    val_loader: DataLoader,
) -> tuple[float, float]:
    """Execute one validation epoch and return average loss and accuracy."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y, _ in tqdm(val_loader, desc="Validation Batches", leave=False):
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * x.size(0)

            predicted = (logits > 0.5).float()
            correct += (predicted == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate_model(
    model: torch.nn.Module, device: torch.device, test_loader: DataLoader
) -> tuple[list[float], list[float]]:
    """Evaluate model on test set and return predictions and labels."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y, _ in tqdm(test_loader, desc="Testing Batches"):
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            logits = model(x)
            predicted = (logits > 0.5).float()

            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    return all_preds, all_labels


def plot_training_history(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
    save_path: str = "graphs/training_plot_neural_network.png",
) -> None:
    """Generate and save training/validation loss and accuracy plots."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    _, axes = plt.subplots(1, 2, figsize=(12, 5))

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
    plt.close()
