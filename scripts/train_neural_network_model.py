"""
This script trains a neural network model on the processed dataset located at data/data.parquet.
It uses PyTorch for model definition, training, and evaluation.

After training it runs evaluation on the test set and plots 
the confusion matrix and training history.


Run this script with:
    python scripts/train_neural_network_model.py
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch import nn

from models.NeuralNetwork import NeuralNetwork
from utility.neural_network_utils import (
    get_train_test_val_dataloaders,
    plot_confusion_matrix,
    plot_training_history,
    run_training_and_testing,
)

print("Loading data...")
batch_size = 64
train_loader, val_loader, test_loader = get_train_test_val_dataloaders(batch_size=batch_size)
print("Data loaded, and split into train, val, test.")


print("Starting training...")

model = NeuralNetwork()
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())
device = torch.device("cpu")
epochs = 15


train_losses, train_accs, val_losses, val_accs, all_preds, all_labels = run_training_and_testing(
    model,
    device=device,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=epochs,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
)

plot_confusion_matrix(all_preds, all_labels)

plot_training_history(train_losses, val_losses, train_accs, val_accs)
