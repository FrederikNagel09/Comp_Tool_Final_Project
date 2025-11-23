import sys
from pathlib import Path

import polars as pl
from tqdm import tqdm

# set root path one step back
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from utility.file_utils import get_csv_dataframe  # noqa: E402
from utility.meta_data_utils import calculate_metadata  # noqa: E402
from sentence_transformers import SentenceTransformer
from utility.file_utils import get_csv_dataframe  # noqa: E402
# Load a pretrained embedding model (no API required)
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
    df = df.with_columns(
        pl.col("text").str.len_bytes().alias("len")
    )

    total = df.height
    too_short = df.filter(pl.col("len") < min_len).height
    too_long = df.filter(pl.col("len") > max_len).height
    removed = too_short + too_long

    return {
        "too_short": too_short,
        "too_long": too_long,
        "removed": removed,
        "pct_removed": removed / total * 100
    }

def filter_word_count(df: pl.DataFrame, min_words: int, max_words: int) -> pl.DataFrame:
    """
    Manually filter rows by word count using len(line.split()).
    Removes rows where word count is outside [min_words, max_words].
    """
    # Collect rows as list of dictionaries
    rows = df.to_dicts()
    
    # Keep only rows within the word count limits
    filtered_rows = [
        row for row in rows
        if min_words <= len(row["text"].split()) <= max_words
    ]
    
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
    return df.with_columns(
        pl.arange(0, df.height).alias(column_name)
    )

def chunk_by_tokens(df: pl.DataFrame, token_limit: int = 500) -> pl.DataFrame:
    """
    Split each text_content into chunks of <= token_limit tokens, by lines.
    Preserves ALL original columns.
    """
    rows = []

    for row in tqdm(df.iter_rows(named=True), total=df.height):
        lines = row["text"].split("\n")
        if not lines:
            continue

        # Count total tokens
        line_tokens = [len(line.split()) for line in lines]
        total_tokens = sum(line_tokens)

        # Calculate number of chunks needed
        n_chunks = max(1, (total_tokens + token_limit - 1) // token_limit)

        # Determine lines per chunk
        total_lines = len(lines)
        base = total_lines // n_chunks
        remainder = total_lines % n_chunks

        start = 0
        for i in range(n_chunks):
            end = start + base + (1 if i < remainder else 0)
            chunk_lines = lines[start:end]

            if chunk_lines:
                new_row = dict(row)        # copy all original columns
                new_row["text"] = "\n".join(chunk_lines)  # replace only text
                rows.append(new_row)

            start = end

    return pl.DataFrame(rows)

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

    
    df_ai_human_content = df_ai_human_content.select(
        ["text_content", "label"]
    ).rename({"text_content": "text", "label": "generated"})

    
    # Merge all datasets
    merged_df = pl.concat(
        [df_ai_human, df_ai_generated, df_balanced, df_ai_human_content], how="vertical"
    )   #df_ai_human

    return merged_df


def extend_with_metadata(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extends a Polars DataFrame with calculated metadata for each row.

    Args:
        df (pl.DataFrame): Input DataFrame with a 'text' column.

    Returns:
        pl.DataFrame: Original DataFrame extended with metadata columns.
    """
    print("Columns before adding metadata: ")
    print(df.columns)
    # Compute metadata for each row
    metadata_dicts = [calculate_metadata(text) for text in tqdm(df["text"].to_list(), desc="Calculating metadata")]

    # Convert list of dicts to Polars DataFrame
    metadata_df = pl.DataFrame(metadata_dicts)
    metadata_df = metadata_df.drop(["generated"], strict=False)
    print("Metadata columns to add: ")
    print(metadata_df.columns)

    # Horizontally stack metadata with original DataFrame
    df = df.hstack(metadata_df)
    print("Columns after adding metadata: ")
    print(df.columns)
    return df


def add_embeddings(df: pl.DataFrame) -> pl.DataFrame:
    """
    Replace 'text' column in a Polars DataFrame with embeddings.

    Args:
        df (pl.DataFrame): Input DataFrame with a 'text' column.

    Returns:
        pl.DataFrame: DataFrame with an 'embedding' column instead of 'text'.
    """
    # Compute embeddings for each text
    embeddings = [get_embedding(text) for text in tqdm(df["text"].to_list(), desc="Generating embeddings")]

    # Create a new DataFrame with the embeddings
    embedding_df = pl.DataFrame({"embedding": embeddings})

    return embedding_df


if __name__ == "__main__":

    """
    df = merge_datasets()
    # Remove outliers based on text length
    df = filter_word_count(df, min_words=100, max_words=1000)
    # Remove duplicate texts
    df = drop_duplicate_text(df)
    # Add unique ID
    df = add_unique_id(df, column_name="id")
    # Chunk by tokens and extend dataset
    df = chunk_by_tokens(df, token_limit=250)
    
    df = extend_with_metadata(df)
    
    # Save to CSV in current directory
    df.write_csv("full_dataset.csv")

    print(" number of columns, after merging and processing6: ", df.columns)
    print(f"Final dataset size after cleaning: {df.height} samples.")
    """


    df = get_csv_dataframe("full_dataset.csv")
    embedding_df = add_embeddings(df)

    # Save embeddings to CSV
    embedding_df.write_csv("full_dataset_embeddings.csv")

    print(f"Embeddings dataset shape: {embedding_df.shape}")

    
