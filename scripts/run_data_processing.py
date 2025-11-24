"""
This script takes our 4 datasets and combines them into one big dataset.
It then cleans the data by removing very large and very small text samples.
(100 words < text < 1000 words)

Data cleanup is performed, by removing duplicates and adding a unique ID to each text sample.

Each text sammple is split into chunks of max 250 tokens, chunks from the same text share ID.

Metadata is computed for each text chunk, including and then embeddings are computed and added.
Finally, the updated DataFrame is saved to a Parquet file.

Run this script with (Takes a long time so use DTU server with submit.sh)
    python scripts/run_data_processing.py

Requires these files in the data/ folder:
    data/AI Generated Essays Dataset.csv
    data/AI_Human.csv
    data/ai_human_content_detection_dataset.csv
    data/balanced_ai_human_prompts.csv
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utility.data_processing_utils import (
    add_embeddings,
    add_unique_id,
    chunk_by_tokens,
    drop_duplicate_text,
    extend_with_metadata,
    filter_word_count,
    merge_datasets,
)


def run_data_processing():
    df = merge_datasets()
    df_sample = df.head(100)
    # Remove outliers based on text length
    df = filter_word_count(df_sample, min_words=100, max_words=1000)

    # Remove duplicate texts
    df = drop_duplicate_text(df)
    # Add unique ID
    df = add_unique_id(df, column_name="id")
    # Chunk by tokens and extend dataset
    df = chunk_by_tokens(df, token_limit=250)

    # Calculate and add metadata
    df = extend_with_metadata(df)

    df = add_embeddings(df)

    # Save updated DataFrame to Parquet
    df.write_parquet("data/data_sample.parquet")


if __name__ == "__main__":
    run_data_processing()
