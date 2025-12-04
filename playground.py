import os

import pandas as pd

# Specify your CSV file path
input_path = "data/Merged_dataset.csv"

# Load the CSV file
df = pd.read_csv(input_path)

# Extract a random subset of 100 samples
subset = df.sample(n=min(100, len(df)), random_state=42)

# Create output filename in the same directory
directory = os.path.dirname(input_path)
filename = os.path.basename(input_path)
name, ext = os.path.splitext(filename)
output_path = os.path.join(directory, f"{name}_subset{ext}")

# Save the subset to a new CSV file
subset.to_csv(output_path, index=False)

print(f"Subset saved to: {output_path}")
print(f"Original size: {len(df)} rows")
print(f"Subset size: {len(subset)} rows")
