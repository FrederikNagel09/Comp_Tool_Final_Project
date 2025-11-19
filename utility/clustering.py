import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment


###############################
## different accuract mesures ##
###############################
def _hungarian_accuracy(y_true, y_pred, labels_true, labels_pred):
    """
    Compute best-match accuracy using Hungarian algorithm.
    labels_true / labels_pred are label lists to include in confusion matrix (order matters).
    """
    cm = confusion_matrix(
        y_true, y_pred, labels=labels_true + labels_pred
    )  # safe: we'll slice below
    # Build confusion matrix with rows = true labels, cols = pred labels
    # But confusion_matrix above could include extra rows/cols when labels overlap; we'll recompute properly:
    cm = confusion_matrix(y_true, y_pred, labels=labels_true, sample_weight=None)
    # Solve assignment (maximize matches) -> we minimize -cm
    row_ind, col_ind = linear_sum_assignment(-cm)
    matched = cm[row_ind, col_ind].sum()
    return matched / cm.sum()


def cluster_accuracy_ignore_noise(y_true, y_pred):
    """
    Remove DBSCAN noise points (pred == -1) from evaluation.
    If after removal there are no points, returns np.nan.
    """
    mask = y_pred != -1
    if mask.sum() == 0:
        return np.nan  # nothing to evaluate
    return _hungarian_accuracy(
        y_true[mask],
        y_pred[mask],
        labels_true=list(np.unique(y_true[mask])),
        labels_pred=list(np.unique(y_pred[mask])),
    )


def cluster_accuracy_count_noise_as_wrong(y_true, y_pred):
    """
    Keep noise points and count them as mismatches.
    Implementation: treat noise as its own predicted label but do NOT allow it to be matched to any true label.
    We implement this by computing best assignment for non-noise predicted clusters, then
    subtracting noise count from matched total.
    """
    # points that are assigned to a real cluster (not -1)
    mask_assigned = y_pred != -1
    # If no assigned points, accuracy = 0 (all noise -> all considered wrong)
    if mask_assigned.sum() == 0:
        return 0.0
    # Compute best-match accuracy on assigned points only
    matched_assigned = _hungarian_accuracy(
        y_true[mask_assigned],
        y_pred[mask_assigned],
        labels_true=list(np.unique(y_true[mask_assigned])),
        labels_pred=list(np.unique(y_pred[mask_assigned])),
    )
    # matched_assigned is fraction over assigned points; convert to counts
    matched_count_assigned = matched_assigned * mask_assigned.sum()
    # noise points count as wrong
    total_points = len(y_true)
    matched_total = matched_count_assigned  # noise not matched
    return matched_total / total_points


def cluster_accuracy_treat_noise_as_cluster(y_true, y_pred):
    """
    Treat -1 as a regular cluster label and include it in the Hungarian matching.
    If all predictions are the same single label, function still works.
    """
    labels_true = list(np.unique(y_true))
    labels_pred = list(np.unique(y_pred))
    # Ensure confusion_matrix rows=labels_true, cols=labels_pred
    cm = confusion_matrix(y_true, y_pred, labels=labels_true)
    # If number of pred clusters differs from number of true classes, Hungarian still works:
    row_ind, col_ind = linear_sum_assignment(-cm)
    matched = cm[row_ind, col_ind].sum()
    return matched / cm.sum()


###############################
## clustering                ##
###############################
def optimize_dbscan(X, min_samples_range=(1, 10)):
    """
    Automatically estimate optimal DBSCAN parameters for given embeddings.

    Returns:
        best_eps: estimated eps
        best_min_samples: chosen min_samples
    """

    # Compute nearest neighbors distances
    neigh = NearestNeighbors(n_neighbors=max(min_samples_range))
    neigh.fit(X)
    distances, _ = neigh.kneighbors(X)

    # Sort the distance to the k-th nearest neighbor for each min_samples
    # Typically the "elbow" in the curve is used to pick eps
    k_distances = np.sort(distances[:, -1])  # using max min_samples for elbow

    # Automatic eps estimation: use the point with maximum curvature (elbow)
    # Simple heuristic: 95th percentile distance
    best_eps = np.percentile(k_distances, 95)

    # i have found that the best is the smallest possible (1)
    best_min_samples = 1

    return best_eps, best_min_samples


def clustering_dbscan(data: np.ndarray, eps: int = 0.48628131225528587, min_samples: int = 1):
    # split data into x and y and normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(data[:, :-1])
    y = data[:, -1]

    # cluster
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    print("clustering data...")
    dbscan.fit(X)

    return dbscan


if __name__ == "__main__":
    # exmaple usage
    data = np.load("data/A.npy")
    y = data[:, -1]

    cluster = clustering_dbscan(data)

    y_pred = cluster.labels_

    print(f"Accuracy: {cluster_accuracy_count_noise_as_wrong(y, y_pred)}")
