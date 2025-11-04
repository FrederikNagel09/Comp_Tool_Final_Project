import polars as pl


def load_large_csv_polars(path: str) -> pl.DataFrame:
    """
    Load a CSV file into a pandas DataFrame and returns it.
    Using Polars for better performance on large files.
    """
    return pl.read_csv(path)