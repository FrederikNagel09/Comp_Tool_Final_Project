import pickle

# add root path to dir
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
print(ROOT)


class LSH_AI_Detector:
    def __init__(self, num_hash_tables=10, num_hash_bits=16):
        """
        num_hash_tables: number of separate LSH tables
        num_hash_bits: number of random hyperplanes per table
        """
        self.num_hash_tables = num_hash_tables
        self.num_hash_bits = num_hash_bits
        self.hash_tables = []
        self.hyperplanes = []
        self.X = None
        self.y = None

    def _generate_hyperplanes(self, dim):
        """Generate random hyperplanes for LSH."""
        return np.random.randn(self.num_hash_tables, self.num_hash_bits, dim)

    def _hash_vector(self, v):
        """Hash a single vector across all LSH tables."""
        hashes = []
        for planes in self.hyperplanes:
            projection = np.dot(planes, v)
            bits = (projection > 0).astype(int)
            hash_key = "".join(bits.astype(str))
            hashes.append(hash_key)
        return hashes

    def fit(self, X, y):
        """Build LSH index."""
        self.X = X
        self.y = y
        n_samples, dim = X.shape

        # Generate random hyperplanes
        self.hyperplanes = self._generate_hyperplanes(dim)

        # Create hash tables
        self.hash_tables = [defaultdict(list) for _ in range(self.num_hash_tables)]

        # Insert all points into hash tables
        for idx, v in enumerate(X):
            hash_keys = self._hash_vector(v)
            for table_idx, h in enumerate(hash_keys):
                self.hash_tables[table_idx][h].append(idx)

    def query(self, x_new, top_k=10):
        """Return nearest neighbors from candidate LSH buckets."""
        hash_keys = self._hash_vector(x_new)

        # Gather candidates from all hash tables
        candidates = set()
        for table_idx, h in enumerate(hash_keys):
            bucket = self.hash_tables[table_idx].get(h, [])
            candidates.update(bucket)

        if not candidates:
            return []  # No similar items found

        # Compute cosine similarity to candidates
        candidate_vectors = self.X[list(candidates)]
        sims = euclidean_distances([x_new], candidate_vectors)[0]

        # Pick top k most similar
        best_idx = np.argsort(sims)[::-1][:top_k]
        neighbor_ids = np.array(list(candidates))[best_idx]

        return neighbor_ids

    def predict(self, x_new, top_k=10):
        """Predict 1 (AI) or 0 (human)."""
        neighbors = self.query(x_new, top_k=top_k)

        if len(neighbors) == 0:
            # If no similar texts, treat as anomaly → likely AI
            return 1

        votes = self.y[neighbors]
        return int(np.mean(votes) >= 0.5)

    def predict_proba(self, x_new, top_k=10):
        """Return probability the new text is AI."""
        neighbors = self.query(x_new, top_k=10)

        if len(neighbors) == 0:
            return 0.75  # default suspicion score

        votes = self.y[neighbors]
        return np.mean(votes)

    def save_hash_tables(self, path="data/hash_tables.pkl"):
        "Saves the hash tables to be loaded later"
        with open(path, "wb") as f:
            pickle.dump(self.hash_tables, f)

    def load_hash_tables(self, X, y, path="data/hash_tables.pkl"):
        "Load already made hash tables"
        with open(path, "rb") as f:
            self.hash_tables = pickle.load(f)

        self.X = X
        self.y = y


if __name__ == "__main__":
    # Example usage:
    #data = np.load("data/A.npy")
    print("loading data...")
    df = pl.read_parquet("data/data.parquet",n_rows=10000)

    feature_colums = feature_columns=["word_count","character_count","lexical_diversity","avg_sentence_length","avg_word_length","flesch_reading_ease","gunning_fog_index","punctuation_ratio"]
    X = np.hstack((df.select(feature_columns),np.array(df["embedding"].to_list())))
    y = df["generated"].to_numpy()

    # Normalise data and split into X and y
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = y

    # split into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.1, random_state=0)

    # create ai detector and train on data
    detector = LSH_AI_Detector(num_hash_tables=24, num_hash_bits=18)
    print("Fitting model to data...")
    detector.fit(X, y)

    # test on test data
    print("Testing model...")
    predictions = []
    sum = 0
    for i in range(len(y_test)):
        pred = detector.predict(X_test[i])
        predictions.append(pred)
        if pred == y_test[i]:
            sum += 1

        print(f"\r{i}/{len(y_test)}", end="", flush=True)

    print("")
    print(f"Accuracy: {sum / len(y_test)}")

    # save hash_table
    detector.save_hash_tables("data/hash_tables.pkl")
