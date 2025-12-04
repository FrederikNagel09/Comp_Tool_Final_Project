"""
This script trains a neural network model on the processed dataset located at data/data.parquet.
It uses PyTorch for model definition, training, and evaluation.

After training it runs evaluation on the test set and plots
the confusion matrix and training history.


Run this script with:
    python scripts/train_neural_network_model.py
"""

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch import nn

from config import BATCH_SIZE
from models.NeuralNetwork import NeuralNetwork
from utility.neural_network_utils import (
    plot_training_history,
    run_training_and_testing,
)
from utility.other_utils import get_train_test_val_dataloaders, plot_confusion_matrix


def main(data_path: str):
    output_dir = "results/neural_network/"

    print("Loading data...")
    train_loader, val_loader, test_loader = get_train_test_val_dataloaders(
        batch_size=BATCH_SIZE, data_path=data_path
    )
    print("Data loaded, and split into train, val, test.")

    print("Starting training...")

    model = NeuralNetwork()
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cpu")
    epochs = 25

    train_losses, train_accs, val_losses, val_accs, all_preds, all_labels = (
        run_training_and_testing(
            model,
            device=device,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )
    )

    plot_confusion_matrix(
        all_preds, all_labels, save_path="results/graphs/confusion_matrix_neural_network.png"
    )

    plot_training_history(train_losses, val_losses, train_accs, val_accs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path to data
    parser.add_argument("--data_path", type=str, default="data/")
    args = parser.parse_args()

    main(args.data_path)
