import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# set root path one step back
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from utility.file_utils import get_csv_dataframe  # noqa: E402

# Load your CSV
df = get_csv_dataframe("data/Merged_dataset.csv")

# Compute text lengths
text_lengths = [len(text.split()) for text in df["text"]]

# ----------------------
# Statistics
# ----------------------
lengths_array = np.array(text_lengths)
print("Basic Statistics:")
print(f"Count: {len(lengths_array)}")
print(f"Mean: {lengths_array.mean():.2f}")
print(f"Median: {np.median(lengths_array):.2f}")
print(f"Min: {lengths_array.min()}")
print(f"Max: {lengths_array.max()}")
print(f"Standard Deviation: {lengths_array.std():.2f}")
print(f"25th Percentile: {np.percentile(lengths_array, 25)}")
print(f"75th Percentile: {np.percentile(lengths_array, 75)}")
print(f"Skewness: {pd.Series(lengths_array).skew():.2f}")
print(f"Kurtosis: {pd.Series(lengths_array).kurtosis():.2f}")

# ----------------------
# Visualizations
# ----------------------
sns.set(style="whitegrid")

# Histogram
min_len = 100
max_len = 1000

plt.figure(figsize=(10, 6))
sns.histplot(text_lengths, bins=50, kde=True, color="skyblue")

plt.axvline(min_len, color="black", linestyle="--")
plt.axvline(max_len, color="black", linestyle="--")

plt.title("Histogram of Text Lengths")
plt.xlabel("Text Length")
plt.ylabel("Frequency")
plt.show()
