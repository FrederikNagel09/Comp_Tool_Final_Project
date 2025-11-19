import sys
from pathlib import Path

import polars as pl
from tqdm import tqdm

# set root path one step back
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from utility.file_utils import get_csv_dataframe  # noqa: E402
from utility.meta_data_utils import calculate_metadata  # noqa: E402


def add_metadata(df: pl.DataFrame) -> pl.DataFrame:
    """Compute metadata for all rows in a dataframe with a progress bar."""
    metadata_dicts = [calculate_metadata(text) for text in tqdm(df["text"].to_list(), desc="Calculating metadata")]
    metadata_df = pl.DataFrame(metadata_dicts)
    return df.hstack(metadata_df)


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

    # Keep only relevant columns from AI Human Content dataset
    metadata_cols = [
        "word_count",
        "character_count",
        "lexical_diversity",
        "avg_sentence_length",
        "avg_word_length",
        "flesch_reading_ease",
        "gunning_fog_index",
        "punctuation_ratio",
    ]
    df_ai_human_content = df_ai_human_content.select(
        ["text_content", "label"] + metadata_cols
    ).rename({"text_content": "text", "label": "generated"})

    
    # Merge all datasets
    merged_df = pl.concat(
        [df_ai_human, df_ai_generated, df_balanced, df_ai_human_content], how="vertical"
    )

    return merged_df

    

if __name__ == "__main__":
    
    merged_df = merge_datasets()

    # Save to CSV in current directory
    merged_df.write_csv("merged_dataset_with_metadata.csv")

    print(f"Merged dataset shape: {merged_df.shape}")