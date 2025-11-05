from utility.file_utils import get_csv_dataframe
from utility.meta_data_utils import calculate_metadata

if __name__ == "__main__":
    # Example usage
    df = get_csv_dataframe("data/AI_Human.csv")

    # Sample 1471 is AI generated
    n = 1471
    sample_text = df["text"][n]
    label = df["generated"][n]
    metadata = calculate_metadata(sample_text)

    print("Text sample:\n", sample_text)

    print("\nCalculated Metadata:")
    for key, value in metadata.items():
        print(f"{key}: {value}")

    print("\nAI generated: ", label)
