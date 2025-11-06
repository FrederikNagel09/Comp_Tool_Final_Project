from pathlib import Path
import sys
import polars as pl

# set root path one step back
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from utility.file_utils import get_csv_dataframe
from utility.meta_data_utils import calculate_metadata


if __name__ == "__main__":

    df_ai_human = get_csv_dataframe("data/AI_Human.csv")
    df_ai_human_content = get_csv_dataframe("data/ai_human_content_detection_dataset.csv")
    df_ai_generated = get_csv_dataframe("data/AI Generated Essays Dataset.csv")
    df_balanced = get_csv_dataframe("data/balanced_ai_human_prompts.csv")

    print(df_ai_human.columns)
    print(df_ai_human_content.columns)
    print(df_ai_generated.columns)
    print(df_balanced.columns)

    # Columns are "text" and "generated" in all datasets

    # Merge datasets
    merged_df = pl.concat([df_ai_human, df_ai_generated, df_balanced], how="vertical")

    print(f"Merged dataset shape: {merged_df.shape}")

    