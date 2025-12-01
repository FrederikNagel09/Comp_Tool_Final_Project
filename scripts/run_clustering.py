"""
Main script for running K-means clustering on text generation detection task.

Usage:
    python scripts/run_clustering.py --data_path data/data.parquet --output_dir results/
"""

import argparse
import os
from pathlib import Path
import numpy as np
import sys
import polars as pl
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import List
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utility.clustering_utils import (
    fit_kmeans_and_assign_clusters,
    assign_cluster_labels_by_majority_vote,
    grid_search_kmeans,
    evaluate_and_save_results,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="K-means clustering for text generation detection"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to input parquet file containing features and labels"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/",
        help="Directory to save results (default: results/)"
    )
    
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Number of samples to use (default: all samples)"
    )
    
    parser.add_argument(
        "--train_size",
        type=float,
        default=0.8,
        help="Proportion of data for training (default: 0.8)"
    )
    
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.1,
        help="Proportion of data for validation (default: 0.1)"
    )
    
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Proportion of data for testing (default: 0.1)"
    )
    
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )   
    
    parser.add_argument(
        "--n_clusters_grid",
        type=int,
        nargs="+",
        default=[2, 30, 100],
        help="List of n_clusters values for grid search (default: 10 20 50 100)"
    )
    
    return parser.parse_args()


def load_and_prepare_data(
    data_path: str,
    n_samples: int = None,
    feature_columns: List[str] = None,
) -> tuple:
    """
    Load data from parquet file and extract features and labels.
    
    Args:
        data_path: Path to parquet file
        n_samples: Number of samples to load (None = all samples)
        feature_columns: List of feature column names
        
    Returns:
        X: Feature matrix, shape (n_samples, n_features)
        y: Label vector, shape (n_samples,)
    """
    print(f"Loading data from: {data_path}")
    
    # Default feature columns
    if feature_columns is None:
        feature_columns = [
            "word_count",
            "character_count",
            "lexical_diversity",
            "avg_sentence_length",
            "avg_word_length",
            "flesch_reading_ease",
            "gunning_fog_index",
            "punctuation_ratio"
        ]
    
    # Load parquet file
    df = pl.read_parquet(data_path)
    
    # Sample if requested
    if n_samples is not None:
        df = df.head(n_samples)
        print(f"Using first {n_samples} samples")
    else:
        print(f"Using all {len(df)} samples")
    
    # Extract features and labels
    X = np.hstack((df.select(feature_columns).to_numpy(), np.array(df["embedding"].to_list())))
    y = df["generated"].to_numpy()
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y)}")
    
    return X, y


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    train_size: float,
    val_size: float,
    test_size: float,
    random_state: int,
) -> tuple:
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Feature matrix
        y: Label vector
        train_size: Proportion for training
        val_size: Proportion for validation
        test_size: Proportion for testing
        random_state: Random seed
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Verify sizes sum to 1
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
        "train_size, val_size, and test_size must sum to 1.0"
    
    # First split: separate training set
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(1 - train_size),
        random_state=random_state,
        stratify=y,
    )
    
    # Second split: separate validation and test sets
    relative_test_size = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=y_temp,
    )
    
    print("\nData split sizes:")
    print(f"  Training:   {X_train.shape[0]:6d} samples ({train_size:.1%})")
    print(f"  Validation: {X_val.shape[0]:6d} samples ({val_size:.1%})")
    print(f"  Test:       {X_test.shape[0]:6d} samples ({test_size:.1%})")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("K-MEANS CLUSTERING FOR TEXT GENERATION DETECTION")
    print("="*70)
    
    # Load and prepare data
    X, y = load_and_prepare_data(
        data_path=args.data_path,
        n_samples=args.n_samples,
    )
    
    # Normalize features
    print("\nNormalizing features with StandardScaler...")
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    # Save scaler for future use
    scaler_path = output_dir / "scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {scaler_path}")
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X=X_normalized,
        y=y,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    
    # Grid search for best hyperparameters
    print("\n" + "="*70)
    print("GRID SEARCH")
    print("="*70)
    print(f"Testing n_clusters: {args.n_clusters_grid}")
    
    best_n_clusters, best_validation_accuracy = grid_search_kmeans(
        X_train=X_train,
        X_val=X_val,
        y_val=y_val,
        n_clusters_grid=args.n_clusters_grid,
        random_state=args.random_state,
    )
    
    print("\n" + "-"*70)
    print("BEST CONFIGURATION:")
    print(f"  n_clusters = {best_n_clusters}")
    print(f"  Validation Accuracy = {best_validation_accuracy:.4f}")
    print("-"*70)
    
    # Final evaluation on test set
    print("\n" + "="*70)
    print("FINAL EVALUATION ON TEST SET")
    print("="*70)
    
    print(f"Training final model with n_clusters={best_n_clusters}...")
    test_cluster_assignments, final_centroids = fit_kmeans_and_assign_clusters(
        X_train=X_train,
        X_eval=X_test,
        n_clusters=best_n_clusters,
        random_state=args.random_state,
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
    with open(mappings_path, 'wb') as f:
        pickle.dump({
            'cluster_to_class': cluster_to_class_mapping,
            'centroids': final_centroids,
        }, f)
    print(f"Cluster mappings saved to: {mappings_path}")
    
    # Evaluate and save results
    results_csv_path = output_dir / "clustering_results.csv"
    results_df = evaluate_and_save_results(
        y_test=y_test,
        y_predicted=y_test_predicted,
        best_n_clusters=best_n_clusters,
        best_accuracy=best_validation_accuracy,
        output_filepath=str(results_csv_path),
    )
    
    # Print final results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(results_df.to_string(index=False))
    print("="*70)
    
    print("\n✓ Clustering complete!")
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nTo plot confusion matrix, use:")
    print(f"  data = np.load('{predictions_path}')")
    print(f"  plot_confusion_matrix(data['y_pred'], data['y_true'])")


if __name__ == "__main__":
    main()