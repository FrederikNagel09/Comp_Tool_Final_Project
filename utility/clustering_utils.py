"""
Utility functions for K-means clustering on text generation detection.

This module provides functions for:
- K-means clustering with centroid-based classification
- Grid search for hyperparameter optimization
- Model evaluation and result saving
"""

import numpy as np
import pandas as pd
import sys
import os
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, confusion_matrix
from typing import Tuple, Dict, List
from itertools import product
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utility.other_utils import plot_confusion_matrix


def fit_kmeans_and_assign_clusters(
    X_train: np.ndarray,
    X_eval: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit K-means on training data and assign evaluation data to nearest centroids.
    
    Args:
        X_train: Training feature matrix of shape (n_train_samples, n_features)
        X_eval: Evaluation feature matrix of shape (n_eval_samples, n_features)
        n_clusters: Number of clusters for K-means
        random_state: Random seed for reproducibility
        
    Returns:
        eval_cluster_assignments: Cluster assignments for eval data, shape (n_eval_samples,)
        cluster_centroids: Centroid coordinates, shape (n_clusters, n_features)
    """
    # Fit K-means on training data
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300,
    )
    kmeans.fit(X_train)
    
    # Get centroids from fitted model
    cluster_centroids = kmeans.cluster_centers_
    
    # Assign evaluation samples to nearest centroid
    eval_cluster_assignments = kmeans.predict(X_eval)
    
    return eval_cluster_assignments, cluster_centroids


def assign_cluster_labels_by_majority_vote(
    cluster_assignments: np.ndarray,
    true_labels: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[Dict[int, int], np.ndarray]:
    """
    Assign class labels to clusters using majority voting.
    
    For each cluster, compute the fraction of AI-generated samples.
    If >= threshold, assign cluster to class 1 (AI), otherwise class 0 (human).
    
    Args:
        cluster_assignments: Cluster IDs for each sample, shape (n_samples,)
        true_labels: True binary labels (0=human, 1=AI), shape (n_samples,)
        threshold: Threshold for majority vote (default 0.5)
        
    Returns:
        cluster_to_class_mapping: Dictionary mapping cluster_id -> class_label
        predicted_labels: Predicted class labels, shape (n_samples,)
    """
    unique_cluster_ids = np.unique(cluster_assignments)
    cluster_to_class_mapping = {}
    
    for cluster_id in unique_cluster_ids:
        # Get all samples in this cluster
        cluster_mask = (cluster_assignments == cluster_id)
        cluster_true_labels = true_labels[cluster_mask]
        
        # Compute fraction of AI-generated samples
        fraction_ai_generated = np.mean(cluster_true_labels)
        
        # Assign cluster to class based on majority
        cluster_class = 1 if fraction_ai_generated >= threshold else 0
        cluster_to_class_mapping[cluster_id] = cluster_class
    
    # Generate predictions for all samples
    predicted_labels = np.array([
        cluster_to_class_mapping[cluster_id] 
        for cluster_id in cluster_assignments
    ])
    
    return cluster_to_class_mapping, predicted_labels


def evaluate_single_kmeans_config(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> float:
    """
    Evaluate a single K-means configuration on validation data.
    
    Args:
        X_train: Training features, shape (n_train_samples, n_features)
        X_val: Validation features, shape (n_val_samples, n_features)
        y_val: Validation labels, shape (n_val_samples,)
        n_clusters: Number of clusters to use
        random_state: Random seed for reproducibility
        
    Returns:
        accuracy: Accuracy score on validation data
    """
    # Fit K-means and assign validation samples
    val_cluster_assignments, _ = fit_kmeans_and_assign_clusters(
        X_train=X_train,
        X_eval=X_val,
        n_clusters=n_clusters,
        random_state=random_state,
    )
    
    # Assign class labels to clusters via majority vote
    _, y_val_predicted = assign_cluster_labels_by_majority_vote(
        cluster_assignments=val_cluster_assignments,
        true_labels=y_val,
    )
    
    # Compute accuracy
    accuracy = np.mean(y_val_predicted == y_val)
    
    return accuracy


def grid_search_kmeans(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_clusters_grid: List[int],
    random_state: int = 42,
) -> Tuple[int, float]:
    """
    Perform grid search over K-means hyperparameters.
    
    Tests different values of n_clusters and selects the configuration
    with the highest F1 score on the validation set.
    
    Args:
        X_train: Training features, shape (n_train_samples, n_features)
        X_val: Validation features, shape (n_val_samples, n_features)
        y_val: Validation labels, shape (n_val_samples,)
        n_clusters_grid: List of n_clusters values to test
        random_state: Random seed for reproducibility
        
    Returns:
        best_n_clusters: Best number of clusters found
        best_f1_score: Best F1 score achieved
    """
    best_f1_score = -1.0
    best_n_clusters = None
    
    print(f"\nStarting grid search over {len(n_clusters_grid)} configurations...")
    
    for n_clusters in tqdm(n_clusters_grid, desc="Grid search"):
        print(f"  Testing n_clusters={n_clusters}")
        
        # Evaluate this configuration
        f1_score_ai = evaluate_single_kmeans_config(
            X_train=X_train,
            X_val=X_val,
            y_val=y_val,
            n_clusters=n_clusters,
            random_state=random_state,
        )
        
        print(f"    F1 score (AI): {f1_score_ai:.4f}")
        
        # Update best configuration if improved
        if f1_score_ai > best_f1_score:
            best_f1_score = f1_score_ai
            best_n_clusters = n_clusters
            print(f"    ✓ New best configuration!")
    
    return best_n_clusters, best_f1_score


def evaluate_and_save_results(
    y_test: np.ndarray,
    y_predicted: np.ndarray,
    best_n_clusters: int,
    best_accuracy: float,
    output_filepath: str = "clustering_results.csv",
) -> pd.DataFrame:
    """
    Evaluate final model performance and save results to CSV.
    
    Args:
        y_test: True test labels, shape (n_test_samples,)
        y_predicted: Predicted test labels, shape (n_test_samples,)
        best_n_clusters: Best n_clusters from validation
        best_f1_score: Best F1 score from validation
        output_filepath: Path to save results CSV
        
    Returns:
        results_dataframe: DataFrame containing all results
    """
    # Compute confusion matrix
    plot_confusion_matrix(y_test, y_predicted, save_path="graphs/confusion_matrix_kmeans.png")
    
    # Compute test accuracy
    test_accuracy = np.mean(y_predicted == y_test)
    
    # Compute test F1 score
    test_f1_score = f1_score(y_test, y_predicted, pos_label=1, zero_division=0.0)
    
    # Create results dataframe
    results_dataframe = pd.DataFrame({
        'best_n_clusters': [best_n_clusters],
        'validation_accuracy': [best_accuracy],
        'test_accuracy': [test_accuracy],
        'test_f1_score': [test_f1_score],
    })
    
    # Save to CSV
    results_dataframe.to_csv(output_filepath, index=False)
    print(f"\nResults saved to: {output_filepath}")
    
    return results_dataframe