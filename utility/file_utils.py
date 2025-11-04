import pandas as pd


def load_csv_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame and returns it.
    """
    return pd.read_csv(file_path)