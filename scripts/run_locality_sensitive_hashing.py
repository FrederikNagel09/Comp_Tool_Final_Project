import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BATCH_SIZE
from utility.locality_sensitive_hashing_utils import (
    evaluate_and_plot,
    grid_search_lsh,
)
from utility.other_utils import dataloader_to_arrays, get_train_test_val_dataloaders


def main():
    """Complete training pipeline with grid search."""

    print("Loading data...")
    data_path = "data/data.parquet"
    train_loader, val_loader, test_loader = get_train_test_val_dataloaders(
        batch_size=BATCH_SIZE, data_path=data_path
    )

    # Convert DataLoaders to NumPy arrays
    print("Converting dataloaders to arrays...")
    feature_matrix_train, labels_train = dataloader_to_arrays(train_loader)
    feature_matrix_val, labels_val = dataloader_to_arrays(val_loader)
    feature_matrix_test, labels_test = dataloader_to_arrays(test_loader)

    print("\nDataset sizes:")
    print(f"  Train: {feature_matrix_train.shape}")
    print(f"  Val:   {feature_matrix_val.shape}")
    print(f"  Test:  {feature_matrix_test.shape}")

    # Define parameter grid
    param_grid = {"num_hash_tables": [16, 24, 32], "num_hash_bits": [16, 18], "top_k": [10, 15]}

    # Run grid search
    print("\nStarting grid search for LSH hyperparameters...")
    best_detector, best_params = grid_search_lsh(
        feature_matrix_train, labels_train, feature_matrix_val, labels_val, param_grid
    )

    print("\nEvaluating best model on test set...")
    evaluate_and_plot(
        best_detector,
        feature_matrix_test,
        labels_test,
        top_k=best_params["top_k"],
        dataset_name="Test",
        save_path="graphs/lsh_confusion_matrix_test.png",
    )

    # Save best model
    model_path = "data/lsh_best_model.pkl"
    best_detector.save(model_path)
    print(f"\nBest model saved to {model_path}")


if __name__ == "__main__":
    main()
