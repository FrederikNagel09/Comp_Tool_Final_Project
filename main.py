from utility.file_utils import get_csv_dataframe
from utility.meta_data_utils import calculate_metadata

if __name__ == "__main__":
    # Example usage
    df = get_csv_dataframe("Comp_Tool_Final_Project/data/AI_Human.csv")

    
    #Sample 1471 is AI generated
    n = 1471
    sample_text = df["text"][n]
    label = df['generated'][n]
    metadata = calculate_metadata(sample_text)
    
    print("Text sample:\n", sample_text)

    print("\nCalculated Metadata:")
    for key, value in metadata.items():
        print(f"{key}: {value}")

    print("\nAI generated: ", label)

import pandas as pd

if __name__ == "__main__":
    # Load the dataframe
    df = pd.read_csv("Comp_Tool_Final_Project/data/AI_Human.csv")  # Ensure correct loading
    
    # Debug: Print column names
    print("Columns in the dataframe:", df.columns)
    
    # Strip spaces from column names
    df.columns = df.columns.str.strip()
    
    # Drop rows with missing values
    df = df.dropna(subset=["text", "generated"])  # Ensure column names match
    
    # Initialize accumulators and counters
    ai_metadata_sum = {}
    human_metadata_sum = {}
    ai_count = 0
    human_count = 0
    iter = 0
    
    # Loop through all rows
    for index, row in df.iterrows():
        sample_text = row["text"]
        label = row["generated"]  # Assuming 'generated' is 1 for AI and 0 for human
        metadata = calculate_metadata(sample_text)
        
        # Update accumulators and counters
        if label == 1:  # AI-generated
            ai_count += 1
            for key, value in metadata.items():
                ai_metadata_sum[key] = ai_metadata_sum.get(key, 0) + value
        else:  # Human-generated
            human_count += 1
            for key, value in metadata.items():
                human_metadata_sum[key] = human_metadata_sum.get(key, 0) + value

        iter +=1
        print(iter)

    
    # Calculate averages
    ai_metadata_avg = {key: value / ai_count for key, value in ai_metadata_sum.items()}
    human_metadata_avg = {key: value / human_count for key, value in human_metadata_sum.items()}
    
    # Print results
    print("\nAverage Metadata for AI-generated texts:")
    for key, value in ai_metadata_avg.items():
        print(f"{key}: {value}")
    
    print("\nAverage Metadata for Human-generated texts:")
    for key, value in human_metadata_avg.items():
        print(f"{key}: {value}")