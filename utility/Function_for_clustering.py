# Function_for_clustering.py
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    pairwise_distances_argmin_min,
)


def dbscan_with_centroids(
    X_train,
    X_eval,
    eps,
    min_samples,
):
    # Fit DBSCAN on the training data
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    train_labels_raw = dbscan.fit_predict(X_train)  # -1 = noise

    # Get valid (non-noise) clusters
    unique_raw = np.unique(train_labels_raw)
    non_noise_labels = [lbl for lbl in unique_raw if lbl != -1]

    # Map DBSCAN labels like {3,7,10} -> {0,1,2,...}
    label_to_idx = {lbl: i for i, lbl in enumerate(non_noise_labels)}
    K = len(non_noise_labels)

    # Prepare centroid matrix
    d = X_train.shape[1]
    centroids = np.zeros((K, d), dtype=np.float32)

    # Compute centroids for each DBSCAN cluster
    for raw_lbl in non_noise_labels:
        mask = train_labels_raw == raw_lbl
        if np.sum(mask) == 0:
            continue
        cluster_idx = label_to_idx[raw_lbl]
        centroids[cluster_idx] = X_train[mask].mean(axis=0)
    if K == 0:
        # return empty arrays; caller must handle this
        return np.array([], dtype=int), centroids  # centroids has shape (0, d)

    # Assign eval set to nearest centroid
    eval_cluster_ids, _ = pairwise_distances_argmin_min(X_eval, centroids)

    return eval_cluster_ids, centroids


def majority_vote_predict(cluster_ids, Y_true):
    uniq_clusters = np.unique(cluster_ids)

    cluster_to_class = {}
    for c in uniq_clusters:
        mask = cluster_ids == c
        frac_ai = np.mean(Y_true[mask])
        majority = int(frac_ai >= 0.5)
        cluster_to_class[c] = majority

    Y_pred = np.array([cluster_to_class[c] for c in cluster_ids])

    return cluster_to_class, Y_pred


def gridsearch_dbscan(X_train, X_val, Y_val, eps_grid, min_samples_grid):
    """
    Grid search over eps and min_samples for DBSCAN+centroid classifier.

    Returns:
        best_eps : float
        best_ms  : int
        best_f1  : float
    """
    best_f1 = -1.0
    best_eps = None
    best_ms = None

    for min_samples in min_samples_grid:
        for eps in eps_grid:
            print("min_samlple,eps:", (min_samples, eps))
            # Run DBSCAN + centroid assignment on validation set
            val_cluster_ids, centroids = dbscan_with_centroids(
                X_train,
                X_val,
                eps=eps,
                min_samples=min_samples,
            )

            if centroids.shape[0] == 0 or val_cluster_ids.size == 0:
                continue

            # Majority vote: cluster -> class (0/1)
            cluster_to_class, _ = majority_vote_predict(val_cluster_ids, Y_val)

            # Predict labels for validation set
            Y_val_pred = np.array([cluster_to_class[c] for c in val_cluster_ids])

            # Metrics
            f1_ai = f1_score(Y_val, Y_val_pred, pos_label=1, zero_division=0)

            # Track best F1
            if f1_ai > best_f1:
                best_f1 = f1_ai
                best_eps = float(eps)
                best_ms = int(min_samples)

    return best_eps, best_ms, best_f1


def evaluate_and_save_results(
    Y_test,
    Y_pred,
    best_eps,
    best_ms,
    best_f1,
    filename="clustering_results.csv",
):
    """
    Returns:
        results_df : pd.DataFrame
        cm : np.ndarray
        test_acc : float
    """
    cm = confusion_matrix(Y_test, Y_pred)

    test_acc = np.mean(Y_pred == Y_test)

    results_df = pd.DataFrame(
        {
            "best_eps": [best_eps],
            "best_min_samples": [best_ms],
            "best_f1_ai": [best_f1],
            "test_accuracy": [test_acc],
            "confusion_matrix": [cm.tolist()],
        }
    )

    results_df.to_csv(filename, index=False)

    return results_df
