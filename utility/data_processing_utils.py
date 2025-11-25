import sys
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

# set root path one step back
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from sentence_transformers import SentenceTransformer  # noqa: E402

from utility.file_utils import get_csv_dataframe  # noqa: E402
from utility.meta_data_utils import calculate_metadata  # noqa: E402

# Load a pretrained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text: str) -> list[float]:
    """
    Get embedding for a single text using a sentence-transformer model.
    """
    embedding = model.encode(text, show_progress_bar=False)
    return embedding.tolist()


def count_length_outliers(df: pl.DataFrame, min_len: int = 100, max_len: int = 5000) -> dict:
    """
    Count rows where text_content length is < min_len or > max_len.
    Returns counts and percentages.
    """
    df = df.with_columns(pl.col("text").str.len_bytes().alias("len"))

    total = df.height
    too_short = df.filter(pl.col("len") < min_len).height
    too_long = df.filter(pl.col("len") > max_len).height
    removed = too_short + too_long

    return {
        "too_short": too_short,
        "too_long": too_long,
        "removed": removed,
        "pct_removed": removed / total * 100,
    }


def filter_word_count(df: pl.DataFrame, min_words: int, max_words: int) -> pl.DataFrame:
    """
    Manually filter rows by word count using len(line.split()).
    Removes rows where word count is outside [min_words, max_words].
    """
    # Collect rows as list of dictionaries
    rows = df.to_dicts()

    # Keep only rows within the word count limits
    filtered_rows = [row for row in rows if min_words <= len(row["text"].split()) <= max_words]

    # Convert back to a Polars DataFrame
    return pl.DataFrame(filtered_rows)


def drop_duplicate_text(df: pl.DataFrame) -> pl.DataFrame:
    """
    Remove duplicate rows based on text_content.
    Keeps the first occurrence of each unique text.
    """
    return df.unique(subset=["text"])


def add_unique_id(df: pl.DataFrame, column_name: str = "id") -> pl.DataFrame:
    """
    Add a unique ID column starting at 0.
    """
    return df.with_columns(pl.arange(0, df.height).alias(column_name))


# Chunk by tokens and extend dataset
def chunk_by_tokens(
    df: pl.DataFrame, token_limit: int = 250, min_tokens: int = 100
) -> pl.DataFrame:
    """
    Split each text_content into chunks of <= token_limit tokens (words).
    Creates roughly equal-sized chunks, each between min_tokens and token_limit.
    Preserves ALL original columns.
    """
    rows = []
    for row in tqdm(df.iter_rows(named=True), total=df.height):
        text = row["text"]
        if not text or not text.strip():
            continue

        # Split into words
        words = text.split()
        total_words = len(words)

        if total_words == 0:
            continue

        # Calculate number of chunks needed
        n_chunks = max(1, (total_words + token_limit - 1) // token_limit)

        # Calculate words per chunk for equal distribution
        base_words_per_chunk = total_words // n_chunks
        remainder = total_words % n_chunks

        # Create chunks
        start = 0
        for i in range(n_chunks):
            # Distribute remainder words across first chunks
            chunk_size = base_words_per_chunk + (1 if i < remainder else 0)
            end = start + chunk_size

            chunk_words = words[start:end]

            # Only add chunk if it meets minimum requirement (or if it's the only chunk)
            if len(chunk_words) >= min_tokens or n_chunks == 1:
                new_row = dict(row)
                new_row["text"] = " ".join(chunk_words)
                rows.append(new_row)

            start = end

    df = pl.DataFrame(rows)

    df = df.with_columns(
        [
            pl.col("generated").cast(pl.Boolean),  # save memory for binary
            pl.col("id").cast(pl.UInt32),  # smaller integer type
        ]
    )

    return df


def merge_datasets():
    # Load datasets
    df_ai_human = get_csv_dataframe("data/AI_Human.csv")
    df_ai_human_content = get_csv_dataframe("data/ai_human_content_detection_dataset.csv")
    df_ai_generated = get_csv_dataframe("data/AI Generated Essays Dataset.csv")
    df_balanced = get_csv_dataframe("data/balanced_ai_human_prompts.csv")

    # Ensure 'generated' is Int8 for the 3 main datasets
    for i, df in enumerate([df_ai_human, df_ai_generated, df_balanced]):
        df = df.with_columns([pl.col("generated").cast(pl.Int8)])
        if i == 0:
            df_ai_human = df
        elif i == 1:
            df_ai_generated = df
        else:
            df_balanced = df

    df_ai_human_content = df_ai_human_content.with_columns([pl.col("label").cast(pl.Int8)])

    df_ai_human_content = df_ai_human_content.select(["text_content", "label"]).rename(
        {"text_content": "text", "label": "generated"}
    )

    # Merge all datasets
    merged_df = pl.concat(
        [df_ai_human, df_ai_generated, df_balanced, df_ai_human_content], how="vertical"
    )  # df_ai_human

    return merged_df


def extend_with_metadata(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extends a Polars DataFrame with calculated metadata for each row.

    Args:
        df (pl.DataFrame): Input DataFrame with a 'text' column.

    Returns:
        pl.DataFrame: Original DataFrame extended with metadata columns.
    """
    # Compute metadata for each row
    metadata_dicts = [
        calculate_metadata(text) for text in tqdm(df["text"].to_list(), desc="Calculating metadata")
    ]

    # Convert list of dicts to Polars DataFrame
    metadata_df = pl.DataFrame(metadata_dicts)
    metadata_df = metadata_df.drop(["generated"], strict=False)

    # Horizontally stack metadata with original DataFrame
    df = df.hstack(metadata_df)

    df = df.with_columns(
        [
            pl.col("word_count").cast(pl.UInt16),  # save memory for binary
            pl.col("character_count").cast(pl.UInt16),  # smaller integer type
            pl.col("lexical_diversity").cast(pl.Float32),  # smaller integer type
            pl.col("avg_sentence_length").cast(pl.Float32),  # smaller integer type
            pl.col("avg_word_length").cast(pl.Float32),  # smaller integer type
            pl.col("flesch_reading_ease").cast(pl.Float32),  # smaller integer type
            pl.col("gunning_fog_index").cast(pl.Float32),  # smaller integer type
            pl.col("punctuation_ratio").cast(pl.Float32),
        ]
    )
    return df


def add_embeddings(df: pl.DataFrame, batch_size: int = 64) -> pl.DataFrame:
    """
    Compute embeddings in batches and add them to the DataFrame.
    Removes the 'text' column.
    """
    texts: list[str] = df["text"].to_list()
    embeddings: list[list[float]] = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        batch_emb = model.encode(
            batch, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
        )
        # Convert batch embeddings to float32 before adding to list
        embeddings.extend(batch_emb.astype(np.float32).tolist())

    df = df.with_columns(pl.Series("embedding", embeddings))
    df = df.drop("text")
    return df
