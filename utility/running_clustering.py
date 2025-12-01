from Function_for_clustering import (
    dbscan_with_centroids,
    gridsearch_dbscan,
    majority_vote_predict,
    evaluate_and_save_results,
)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    pairwise_distances_argmin_min,
    confusion_matrix,
    f1_score,
)

# -----------------------------
# Parameters
# -----------------------------
random_state = 42

train_size = 0.8
val_size   = 0.1
test_size  = 0.1


#If too slow, make these list smaller
eps_grid = [
     6.0,  7.0, 8.0,  8.5,  9.0,  9.5,
    10.0, 10.5, 11.0, 11.5,
    12.0, 12.5, 13.0, 13.5,
    14.0, 14.5, 15.0, 15.5,
    16.0, 16.5, 17.0, 17.5,
    18.0, 18.5, 19.0, 19.5,
    20.0, 22.0, 24.0, 26.0,
    28.0, 30.0, 32.0, 35.0
]
min_samples_grid = [
     5,  6,  7,  8,  9, 10, 12, 15, 18, 20, 25, 30,
    40, 50, 60, 75, 100, 150, 200, 300, 500,
    750,   1000,  1500,  2000,  3000,
    5000,  7500, 10000, 15000, 20000, 30000
]
##################################################################################################
#
### Frederik, her kan du loade data...... Ved ikke hvordan du vil have

#Load and split data

# N = number of samples 
# d = number of features
X = np.load("X.npy")     # shape (N, d)
Y = np.load("Y_ids.npy") # shape (N,), binary: 0 = human, 1 = AI

#test case
X = X[:100]
Y= Y[:100]

print("X shape:", X.shape)
print("Y shape:", Y.shape)

norms = np.linalg.norm(X, axis=1)
print("Mean vector norm:", norms.mean(), "Std of norms:", norms.std())

X_train, X_temp, Y_train, Y_temp = train_test_split(
    X,
    Y,
    test_size=(1 - train_size),
    random_state=random_state,
    stratify=Y,
)

X_val, X_test, Y_val, Y_test = train_test_split(
    X_temp,
    Y_temp,
    test_size=test_size / (test_size + val_size),
    random_state=random_state,
    stratify=Y_temp,
)

#######################################################################################
print('grid search')
best_eps, best_ms, best_f1 = gridsearch_dbscan(
    X_train=X_train,
    X_val=X_val,
    Y_val=Y_val,
    eps_grid=eps_grid,
    min_samples_grid=min_samples_grid,)

print("\nchosen parameters:")
print(f"  best_eps     = {best_eps}")
print(f"  best_ms      = {best_ms}")
print(f"  best F1_AI   = {best_f1:.4f}")


# Run DBSCAN with best parameters and assign clusters
test_cluster_ids, final_centroids = dbscan_with_centroids(
    X_train=X_train,
    X_eval=X_test,
    eps=best_eps,
    min_samples=best_ms,
)

# Majority vote
cluster_to_class, Y_pred = majority_vote_predict(
    cluster_ids=test_cluster_ids,
    Y_true=Y_test,
)

# Evaluate and save results 
results_df = evaluate_and_save_results(
    Y_test=Y_test,
    Y_pred=Y_pred,
    best_eps=best_eps,
    best_ms=best_ms,
    best_f1=best_f1,
    filename="clustering_results.csv",
)

print('done')