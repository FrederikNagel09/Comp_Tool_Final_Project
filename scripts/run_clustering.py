"""
Main script for running K-means clustering on text generation detection task.

Usage:
    python scripts/run_clustering.py --data_path subset_data/data_subset.parquet
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import N_CLUSTERS_GRID, RANDOM_STATE, TEST_SIZE, TRAIN_SIZE, VAL_SIZE
from utility.clustering_utils import (
    assign_cluster_labels_by_majority_vote,
    evaluate_and_save_results,
    fit_kmeans_and_assign_clusters,
    grid_search_kmeans,
    load_and_prepare_data,
    split_data,
)


def main(data_path: str) -> None:
    """Main execution function."""

    # Create output directory
    output_dir = Path("results/clustering/")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading in data...")
    # Load and prepare data
    x, y = load_and_prepare_data(
        data_path=data_path,
    )

    # Normalize features
    print("\nNormalizing features with StandardScaler...")
    scaler = StandardScaler()
    x_normalized = scaler.fit_transform(x)

    print("Splitting up data into train, test and val...")
    # Split data
    x_train, x_val, x_test, _, y_val, y_test = split_data(
        x=x_normalized,
        y=y,
        train_size=TRAIN_SIZE,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # Grid search for best hyperparameters
    print(f"Testing n_clusters: {N_CLUSTERS_GRID}")

    best_n_clusters, best_validation_accuracy = grid_search_kmeans(
        x_train=x_train,
        x_val=x_val,
        y_val=y_val,
        n_clusters_grid=N_CLUSTERS_GRID,
        random_state=RANDOM_STATE,
    )

    print("BEST CONFIGURATION:")
    print(f"  n_clusters = {best_n_clusters}")
    print(f"  Validation Accuracy = {best_validation_accuracy:.4f}")

    print(f"Training final model with n_clusters={best_n_clusters}...")
    test_cluster_assignments, final_centroids = fit_kmeans_and_assign_clusters(
        x_train=x_train,
        x_eval=x_test,
        n_clusters=best_n_clusters,
        random_state=RANDOM_STATE,
    )

    # Assign labels via majority voting
    cluster_to_class_mapping, y_test_predicted = assign_cluster_labels_by_majority_vote(
        cluster_assignments=test_cluster_assignments,
        true_labels=y_test,
    )

    # Save predictions and labels for confusion matrix plotting
    predictions_path = output_dir / "test_predictions.npz"
    np.savez(
        predictions_path,
        y_true=y_test,
        y_pred=y_test_predicted,
        cluster_assignments=test_cluster_assignments,
    )
    print(f"Test predictions saved to: {predictions_path}")

    # Save cluster mappings
    mappings_path = output_dir / "cluster_mappings.pkl"
    with open(mappings_path, "wb") as f:
        pickle.dump(
            {
                "cluster_to_class": cluster_to_class_mapping,
                "centroids": final_centroids,
            },
            f,
        )
    print(f"Cluster mappings saved to: {mappings_path}")

    # Evaluate and save results
    results_csv_path = output_dir / "clustering_results.csv"
    evaluate_and_save_results(
        y_test=y_test,
        y_predicted=y_test_predicted,
        best_n_clusters=best_n_clusters,
        best_accuracy=best_validation_accuracy,
        output_filepath=str(results_csv_path),
    )

    print("\nClustering complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path to data
    parser.add_argument("--data_path", type=str, default="data/")
    args = parser.parse_args()

    main(args.data_path)
