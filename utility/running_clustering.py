from Function_for_clustering import (
    dbscan_with_centroids,
    gridsearch_dbscan,
    majority_vote_predict,
    evaluate_and_save_results,
)
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pandas as pd
import polars as pl

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
    8, 16.0, 32.0
]
min_samples_grid = [
    100, 1000, 10000
]


print("loading data...")
df = pl.read_parquet("data/data.parquet")

# sample first 10.000 rows from data
df = df.head(200000)

feature_colums = feature_columns=["word_count","character_count","lexical_diversity","avg_sentence_length","avg_word_length","flesch_reading_ease","gunning_fog_index","punctuation_ratio"]
X = df.select(feature_columns).to_numpy()
y = df["generated"].to_numpy()

# Normalise data and split into X and y
scaler = StandardScaler()
X = scaler.fit_transform(X)
Y = y

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

print("data shapes.")
print(f"  X_train: {X_train.shape}, Y_train: {Y_train.shape}")
print(f"  X_val:   {X_val.shape},   Y_val:   {Y_val.shape}")
print(f"  X_test:  {X_test.shape},  Y_test:  {Y_test.shape}")

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