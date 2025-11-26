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

from utility.neural_network_utils import get_train_test_val_dataloaders, run_training_and_testing, plot_confusion_matrix, plot_training_history
from models.NeuralNetwork import NeuralNetwork

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



