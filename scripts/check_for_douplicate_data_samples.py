import polars as pl
from pathlib import Path
import sys

# Set root path one step back
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
from utility.file_utils import get_csv_dataframe  # noqa: E402

def check_unique_text(csv_path: str, text_column: str = "text") -> str:
    """
    Checks if all elements in the text column of a CSV are unique.
    """
    df = get_csv_dataframe(csv_path)
    total_count = df.height
    unique_count = df.select(pl.col(text_column).n_unique()).to_series()[0]

    if total_count == unique_count:
        return "All items are unique"
    else:
        non_unique = total_count - unique_count
        return f"{non_unique} items are NOT unique out of {total_count} total"

# Example usage:
if __name__ == "__main__":
    result = check_unique_text("data/Merged_dataset.csv")
    print(result)