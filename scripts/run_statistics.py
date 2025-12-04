"""
Script to run statistical analysis on the dataset before and after processing.

We do a simple histogram of text lengths,
Then display basic statistics and label distributions for the data before we process it.

Then a PCA is performed on the processed data, and plotted in 2D space,
further with the explained variance of each component.

Lastly we display basic statistics and label distributions for the processed data.
Here we split the data into the human written and AI written, so the basic statistics for each are
displayed and easilier compared.

run with:
    python scripts/run_statistics.py --path subset_data/ --subset True

"""

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utility.statistics_utils import (
    combine_features,
    compute_text_lengths,
    display_basic_statistics,
    display_label_distribution,
    display_pca_variance,
    load_data,
    normalize_features,
    perform_pca,
    plot_pca_scatter,
    plot_text_length_histogram,
)


def main(path: str, subset: bool) -> None:
    """Execute complete analysis pipeline."""
    # Load data
    df_before, df_after = load_data(path, subset)

    # Text length analysis
    text_lengths = compute_text_lengths(df_before)
    plot_text_length_histogram(text_lengths)

    # Statistics before processing
    display_basic_statistics(df_before, "before")
    display_label_distribution(df_before, "before")

    # PCA analysis
    combined_features = combine_features(df_after)
    scaled_features = normalize_features(combined_features)
    pca_result, pca_model = perform_pca(scaled_features)

    # Display PCA results
    display_pca_variance(pca_model)
    plot_pca_scatter(pca_result, df_after["generated"])

    # Statistics after processing
    display_basic_statistics(df_after, "after")
    display_label_distribution(df_after, "after")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # path to data
    parser.add_argument("--subset", required=True)
    parser.add_argument("--path", type=str, default="data/")
    args = parser.parse_args()

    main(args.path, args.subset)
