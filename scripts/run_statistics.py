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


def main() -> None:
    """Execute complete analysis pipeline."""
    # Load data
    df_before, df_after = load_data()

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
    main()
