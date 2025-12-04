import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from IPython.display import display
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config import NUMERIC_COLS, STANDARD_COLS
from utility.other_utils import get_csv_dataframe


def load_data(path: str, subset: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw and processed datasets."""

    data_path_before = os.path.join(path, "Merged_dataset.csv")
    if subset:
        data_path_after = os.path.join(path, "data_subset.parquet")
        data_path_before = os.path.join(path, "merged_dataset_subset.csv")
    else:
        data_path_after = os.path.join(path, "data.parquet")
        data_path_before = os.path.join(path, "Merged_dataset.csv")

    df_before = get_csv_dataframe(data_path_before)
    df_after = pl.read_parquet(data_path_after).to_pandas()
    return df_before, df_after


def compute_text_lengths(df: pd.DataFrame) -> list[int]:
    """Calculate word count for each text in dataframe."""
    return [len(text.split()) for text in df["text"]]


def plot_text_length_histogram(
    text_lengths: list[int],
    min_len: int = 100,
    max_len: int = 1000,
    save_path: str = "graphs/text_length_histogram.png",
) -> None:
    """Create and save histogram of text lengths with cutoff lines."""
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    sns.histplot(text_lengths, bins=50, kde=True, color="skyblue")
    plt.axvline(min_len, color="black", linestyle="--")
    plt.axvline(max_len, color="black", linestyle="--")

    plt.title("Histogram of Text Lengths")
    plt.xlabel("Text Length")
    plt.ylabel("Frequency")
    plt.savefig(save_path)
    print(f"Saved histogram in {save_path}")
    plt.show()


def display_basic_statistics(df: pd.DataFrame, stage: str) -> None:
    """Display descriptive statistics for numeric columns."""
    print(f"Basic statistics {stage} processing:")
    if stage == "before":
        display(df[STANDARD_COLS].describe())
    else:
        print("Human data (generated = 0):")
        human_data = df[df["generated"] == 0]
        display(human_data[NUMERIC_COLS].describe())

        print("\nAI data (generated = 1):")
        ai_data = df[df["generated"] == 1]
        display(ai_data[NUMERIC_COLS].describe())


def display_label_distribution(df: pd.DataFrame, stage: str) -> None:
    """Display percentage distribution of generated labels."""
    if isinstance(df, pl.DataFrame):
        gen_pct = df["generated"].value_counts(normalize=True).to_pandas()
        result = gen_pct.loc[:, ["generated", "proportion"]].rename(
            columns={"generated": "label", "proportion": "percentage"}
        )
    else:
        gen_pct = (
            df["generated"]
            .value_counts(normalize=True)
            .rename_axis("label")
            .reset_index(name="percentage")
        )
        result = gen_pct[["label", "percentage"]]

    result["percentage"] = (result["percentage"] * 100).round(2)
    print(f"Label distribution {stage} processing:")
    print(result.to_string(index=False))


def combine_features(df: pd.DataFrame) -> np.ndarray:
    """Stack embeddings and numeric features into combined array."""
    print("Stacking embeddings...")
    emb_list = df["embedding"].values
    emb_array = np.vstack(list(tqdm(emb_list, desc="Embedding stacking")))

    numeric_array = df[NUMERIC_COLS].to_numpy()
    combined = np.hstack([emb_array, numeric_array])

    print(f"Combined feature shape: {combined.shape}")
    return combined


def normalize_features(features: np.ndarray) -> np.ndarray:
    """Standardize features using StandardScaler."""
    scaler = StandardScaler()
    return scaler.fit_transform(features)


def perform_pca(features: np.ndarray, n_components: int = 5) -> np.ndarray:
    """Run PCA dimensionality reduction."""
    print("Running PCA...")
    pca = PCA(n_components=n_components, svd_solver="randomized")
    pca_result = pca.fit_transform(features)
    print("PCA complete.")
    return pca_result, pca


def display_pca_variance(pca: PCA) -> None:
    """Display explained variance table for PCA components."""
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    pca_table = pd.DataFrame(
        {
            "PC": [f"PC{i}" for i in range(1, len(explained_var) + 1)],
            "Explained Variance (%)": (explained_var * 100).round(2),
            "Cumulative Variance (%)": (cumulative_var * 100).round(2),
        }
    )
    display(pca_table)


def plot_pca_scatter(
    pca_result: np.ndarray,
    labels: pd.Series,
    save_path: str = "graphs/embedding_pca_scatterplot.png",
) -> None:
    """Create and save PCA scatter plot colored by labels."""
    cmap = ListedColormap(["blue", "red"])
    plt.figure(figsize=(8, 6))

    plt.scatter(
        pca_result[:, 0],
        pca_result[:, 1],
        c=labels,
        cmap=cmap,
        alpha=0.6,
    )

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="label = 0",
            markerfacecolor="blue",
            markersize=8,
            alpha=0.6,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="label = 1",
            markerfacecolor="red",
            markersize=8,
            alpha=0.6,
        ),
    ]
    plt.legend(handles=handles)

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("PCA of Text Embeddings and metadata")
    plt.savefig(save_path)
    print(f"Saved scatter plot in {save_path}")
    plt.show()
