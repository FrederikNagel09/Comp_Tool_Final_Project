import pickle
from collections import defaultdict
from itertools import product

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from utility.other_utils import plot_confusion_matrix


class LocalitySensitiveHashing:
    """Locality-Sensitive Hashing based AI text detector."""

    def __init__(self, num_hash_tables: int = 24, num_hash_bits: int = 18):
        """
        Initialize LSH detector.

        Args:
            num_hash_tables: Number of separate LSH tables (more = better recall)
            num_hash_bits: Number of random hyperplanes per table (more = more selective)
        """
        self.num_hash_tables = num_hash_tables
        self.num_hash_bits = num_hash_bits
        self.hash_tables = []
        self.hyperplanes = None
        self.X = None
        self.y = None
        self.scaler = StandardScaler()

    def _generate_hyperplanes(self, dim: int) -> np.ndarray:
        """Generate random hyperplanes for LSH hashing."""
        return np.random.randn(self.num_hash_tables, self.num_hash_bits, dim)

    def _hash_vector(self, v: np.ndarray) -> list[str]:
        """
        Hash a single vector across all LSH tables.
        """
        hashes = []
        for planes in self.hyperplanes:
            # Project vector onto hyperplanes
            projection = np.dot(planes, v)
            # Convert to binary hash
            bits = (projection > 0).astype(int)
            hash_key = "".join(bits.astype(str))
            hashes.append(hash_key)
        return hashes

    def fit(self, feature_matrix: np.ndarray, labels: np.ndarray, normalize: bool = True):
        """
        Build LSH index from training data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (1 = AI, 0 = human)
            normalize: Whether to apply StandardScaler normalization
        """
        # Normalize features
        if normalize:
            feature_matrix = self.scaler.fit_transform(feature_matrix)

        self.X = feature_matrix
        self.y = labels
        n_samples, dim = feature_matrix.shape

        print(f"Fitting LSH on {n_samples} samples with {dim} features...")

        # Generate random hyperplanes
        self.hyperplanes = self._generate_hyperplanes(dim)

        # Initialize hash tables
        self.hash_tables = [defaultdict(list) for _ in range(self.num_hash_tables)]

        # Insert all training points into hash tables
        for idx, v in tqdm(
            enumerate(feature_matrix), total=len(feature_matrix), desc="Building LSH index"
        ):
            hash_keys = self._hash_vector(v)
            for table_idx, hash_key in enumerate(hash_keys):
                self.hash_tables[table_idx][hash_key].append(idx)

        print(f"LSH index built with {self.num_hash_tables} tables")

    def query(
        self, feature_matrix_new: np.ndarray, top_k: int = 10, normalize: bool = True
    ) -> np.ndarray:
        """
        Find nearest neighbors using LSH.
        """
        # Normalize query vector
        if normalize:
            feature_matrix_new = self.scaler.transform([feature_matrix_new])[0]

        # Get hash keys for query
        hash_keys = self._hash_vector(feature_matrix_new)

        # Gather candidates from all hash tables
        candidates = set()
        for table_idx, hash_key in enumerate(hash_keys):
            bucket = self.hash_tables[table_idx].get(hash_key, [])
            candidates.update(bucket)

        if not candidates:
            return np.array([])

        # Compute cosine similarity to candidates
        candidate_list = list(candidates)
        candidate_vectors = self.X[candidate_list]
        similarities = cosine_similarity([feature_matrix_new], candidate_vectors)[0]

        # Select top-k most similar
        top_k = min(top_k, len(candidate_list))
        best_indices = np.argsort(similarities)[::-1][:top_k]
        neighbor_ids = np.array(candidate_list)[best_indices]

        return neighbor_ids

    def predict(
        self, feature_vector_new: np.ndarray, top_k: int = 10, normalize: bool = True
    ) -> int:
        """
        Predict if text is AI-generated (1) or human-written (0).

        Args:
            feature_vector_new: Query feature vector
            top_k: Number of neighbors to consider
            normalize: Whether to normalize input

        Returns:
            Prediction (1 = AI, 0 = human)
        """
        neighbors = self.query(feature_vector_new, top_k=top_k, normalize=normalize)

        if len(neighbors) == 0:
            # No similar texts found - treat as anomaly (likely AI)
            return 1

        # Majority vote from neighbors
        votes = self.y[neighbors]
        return int(np.mean(votes) >= 0.5)

    def predict_probability(
        self, feature_vector_new: np.ndarray, top_k: int = 10, normalize: bool = True
    ) -> float:
        """
        Return probability that text is AI-generated.

        Args:
            feature_vector_new: Query feature vector
            top_k: Number of neighbors to consider
            normalize: Whether to normalize input

        Returns:
            Probability of being AI-generated [0, 1]
        """
        neighbors = self.query(feature_vector_new, top_k=top_k, normalize=normalize)

        if len(neighbors) == 0:
            return 0.75  # Default suspicion score for anomalies

        votes = self.y[neighbors]
        return float(np.mean(votes))

    def evaluate(
        self, feature_matrix_test: np.ndarray, y_test: np.ndarray, top_k: int = 10
    ) -> dict:
        """
        Evaluate model on test data.

        Args:
            X_test: Test features
            y_test: Test labels
            top_k: Number of neighbors to use

        Returns:
            Dictionary with accuracy and predictions
        """
        predictions = []

        for x in tqdm(feature_matrix_test, desc="Evaluating", unit="sample"):
            pred = self.predict(x, top_k=top_k, normalize=True)
            predictions.append(pred)

        predictions = np.array(predictions)
        accuracy = np.mean(predictions == y_test)

        return {
            "accuracy": accuracy,
            "predictions": predictions,
            "correct": np.sum(predictions == y_test),
            "total": len(y_test),
        }

    def save(self, path: str = "data/lsh_model.pkl"):
        """Save complete model (hash tables, hyperplanes, scaler, data)."""
        model_data = {
            "hash_tables": self.hash_tables,
            "hyperplanes": self.hyperplanes,
            "X": self.X,
            "y": self.y,
            "scaler": self.scaler,
            "num_hash_tables": self.num_hash_tables,
            "num_hash_bits": self.num_hash_bits,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {path}")

    def load(self, path: str = "data/lsh_model.pkl"):
        """Load complete model from file."""
        with open(path, "rb") as f:
            model_data = pickle.load(f)

        self.hash_tables = model_data["hash_tables"]
        self.hyperplanes = model_data["hyperplanes"]
        self.X = model_data["X"]
        self.y = model_data["y"]
        self.scaler = model_data["scaler"]
        self.num_hash_tables = model_data["num_hash_tables"]
        self.num_hash_bits = model_data["num_hash_bits"]

        print(f"Model loaded from {path}")


def run_lsh(
    feature_matrix_train: np.ndarray,
    labels_train: np.ndarray,
    feature_matrix_val: np.ndarray,
    labels_val: np.ndarray,
    num_hash_tables: int,
    num_hash_bits: int,
    top_k: int,
) -> tuple[dict, LocalitySensitiveHashing]:
    """Run LSH training and evaluation."""
    # Create and train LSH detector
    detector = LocalitySensitiveHashing(
        num_hash_tables=num_hash_tables, num_hash_bits=num_hash_bits
    )
    detector.fit(feature_matrix_train, labels_train, normalize=True)

    # Evaluate on validation set
    val_results = detector.evaluate(feature_matrix_val, labels_val, top_k=top_k)
    print(f"Validation Accuracy: {val_results['accuracy']:.4f}")

    return val_results, detector


def grid_search_lsh(
    feature_matrix_train: np.ndarray,
    labels_train: np.ndarray,
    feature_matrix_val: np.ndarray,
    labels_val: np.ndarray,
    param_grid: dict,
) -> tuple[dict, object, dict]:
    """
    Perform grid search and return best detector and results.
    """
    # Generate all parameter combinations
    param_combinations = list(
        product(param_grid["num_hash_tables"], param_grid["num_hash_bits"], param_grid["top_k"])
    )
    num_of_iterations = len(param_combinations)
    print(f"Testing {num_of_iterations} parameter combinations...")

    best_accuracy = 0
    best_detector = None
    best_params = None

    all_results = []
    for i, (num_tables, num_bits, k) in enumerate(param_combinations):
        print(
            f"\nTesting configuration ({i + 1}/{num_of_iterations}): "
            f"num_hash_tables={num_tables}, num_hash_bits={num_bits}, top_k={k}"
        )
        val_results, detector = run_lsh(
            feature_matrix_train,
            labels_train,
            feature_matrix_val,
            labels_val,
            num_tables,
            num_bits,
            k,
        )

        # Store current configuration
        config = {
            "num_hash_tables": num_tables,
            "num_hash_bits": num_bits,
            "top_k": k,
            "accuracy": val_results["accuracy"],
        }
        all_results.append(config)

        # Update best if this is better
        if val_results["accuracy"] > best_accuracy:
            best_accuracy = val_results["accuracy"]
            best_detector = detector
            best_params = {"num_hash_tables": num_tables, "num_hash_bits": num_bits, "top_k": k}

    # Print summary
    print("Grid Search Complete. Best Parameters:\n")
    print(f"num_hash_tables:...{best_params['num_hash_tables']}")
    print(f"num_hash_bits:.....{best_params['num_hash_bits']}")
    print(f"top_k:.............{best_params['top_k']}")
    print(f"Validation Acc:....{best_accuracy:.4f}")

    return best_detector, best_params


def evaluate_and_plot(
    detector: object,
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    top_k: int,
    dataset_name: str = "Validation",
    save_path: str = "graphs/confusion_matrix.png",
) -> dict:
    """
    Evaluate detector and create confusion matrix plot.
    """
    print(f"\n=== {dataset_name} Set Evaluation ===")

    # Get predictions
    results = detector.evaluate(feature_matrix, labels, top_k=top_k)

    # Extract predictions and labels as lists
    all_preds = results["predictions"].tolist()
    all_labels = labels.tolist()

    # Print metrics
    print(f"Test Accuracy: {results['accuracy']:.4f}")

    # Create confusion matrix
    plot_confusion_matrix(all_preds, all_labels, save_path)
