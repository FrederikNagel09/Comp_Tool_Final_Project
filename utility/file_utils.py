import polars as pl


def get_csv_dataframe(path: str) -> pl.DataFrame:
    """
    Load a CSV file into a pandas DataFrame and returns it.
    Using Polars for better performance on large files.
    """
    return pl.read_csv(path)
