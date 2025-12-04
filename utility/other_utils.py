import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.dataloader import LoadDataset


def get_train_test_val_dataloaders(
    batch_size: int = 32,
    data_path: str = "data/data_subset.parquet",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load data and create train, validation, and test dataloaders."""
    df = pl.read_parquet(data_path).to_pandas()

    train_df, val_df, test_df = split_train_val_test(df)

    train_df = pl.from_pandas(train_df)
    val_df = pl.from_pandas(val_df)
    test_df = pl.from_pandas(test_df)

    return set_up_dataloaders(train_df, val_df, test_df, batch_size)


def split_train_val_test(
    df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.5, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train (80%), validation (10%), and test (10%) sets."""
    train_df, temp_df = train_test_split(df, test_size=test_size, random_state=random_state)
    val_df, test_df = train_test_split(temp_df, test_size=val_size, random_state=random_state)
    return train_df, val_df, test_df


def set_up_dataloaders(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders from Polars DataFrames."""
    train_dataset = LoadDataset(train_df)
    val_dataset = LoadDataset(val_df)
    test_dataset = LoadDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def plot_confusion_matrix(
    all_preds: list[float], all_labels: list[float], save_path: str = "graphs/confusion_matrix.png"
) -> None:
    """Generate and save confusion matrix with accuracy in title."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cm = confusion_matrix(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix | Accuracy: {acc:.4f}")
    plt.savefig(save_path)
    plt.close()


def get_csv_dataframe(path: str) -> pl.DataFrame:
    """
    Load a CSV file into a pandas DataFrame and returns it.
    Using Polars for better performance on large files.
    """
    return pl.read_csv(path)


def dataloader_to_arrays(dataloader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert PyTorch DataLoader to NumPy arrays for LSH.
    """
    x_list = []
    y_list = []

    for x, y, _ in tqdm(dataloader, desc="Loading batches"):
        x_list.append(x.numpy())
        y_list.append(y.numpy())

    x = np.vstack(x_list)
    y = np.concatenate(y_list)

    return x, y
