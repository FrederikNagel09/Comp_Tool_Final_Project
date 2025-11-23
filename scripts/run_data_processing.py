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
    # Remove outliers based on text length
    df = filter_word_count(df, min_words=100, max_words=1000)
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
    df.to_parquet("your_updated.parquet", index=False)



if __name__ == "__main__":
    
    run_data_processing()