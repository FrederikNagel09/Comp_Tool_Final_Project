import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BATCH_SIZE
from utility.clustering_utils import (
    dbscan_with_centroids,
    evaluate_and_save_results,
    gridsearch_dbscan,
    majority_vote_predict,
)
from utility.other_utils import get_train_test_val_dataloaders, dataloader_to_arrays
from sklearn.preprocessing import StandardScaler

print("Loading data...")
data_path = "data/data_subset.parquet"
train_loader, val_loader, test_loader = get_train_test_val_dataloaders(
    batch_size=BATCH_SIZE,
    data_path=data_path,
)

print("Converting dataloaders to arrays...")
X_train, Y_train = dataloader_to_arrays(train_loader)
X_val, Y_val = dataloader_to_arrays(val_loader)
X_test, Y_test = dataloader_to_arrays(test_loader)

print("Normalizing (StandardScaler)...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print("\nDataset sizes:")
print(f"  Train: {X_train.shape}")
print(f"  Val:   {X_val.shape}")
print(f"  Test:  {X_test.shape}")

#######################################################################################
# define parameter grid
# Define parameter grid
# If too slow, make these list smaller
eps_grid = [5.0]
min_samples_grid = [5, 20, 50, 100]

print("grid search")
best_eps, best_ms, best_f1 = gridsearch_dbscan(
    X_train=X_train,
    X_val=X_val,
    Y_val=Y_val,
    eps_grid=eps_grid,
    min_samples_grid=min_samples_grid,
)

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

print("done")
