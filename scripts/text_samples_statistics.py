import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# set root path one step back
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from utility.file_utils import get_csv_dataframe  # noqa: E402

# Load your CSV
df = get_csv_dataframe("data/AI_Human.csv")

# Compute text lengths
text_lengths = [len(text) for text in df['text']]

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
plt.figure(figsize=(10,6))
sns.histplot(text_lengths, bins=50, kde=True, color='skyblue')
plt.title("Histogram of Text Lengths")
plt.xlabel("Text Length")
plt.ylabel("Frequency")
plt.show()

# Boxplot
plt.figure(figsize=(10,4))
sns.boxplot(x=text_lengths, color='lightgreen')
plt.title("Boxplot of Text Lengths")
plt.xlabel("Text Length")
plt.show()

# Violin plot
plt.figure(figsize=(10,6))
sns.violinplot(x=text_lengths, color='lightcoral')
plt.title("Violin Plot of Text Lengths")
plt.xlabel("Text Length")
plt.show()

# Cumulative Distribution Function (CDF)
sorted_lengths = np.sort(lengths_array)
cdf = np.arange(1, len(sorted_lengths)+1) / len(sorted_lengths)
plt.figure(figsize=(10,6))
plt.plot(sorted_lengths, cdf, marker='.', linestyle='none')
plt.title("CDF of Text Lengths")
plt.xlabel("Text Length")
plt.ylabel("CDF")
plt.show()

# Frequency of common length ranges
bins = [0, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
hist, bin_edges = np.histogram(text_lengths, bins=bins)
plt.figure(figsize=(10,6))
sns.barplot(x=[f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)],
            y=hist, palette="viridis")
plt.xticks(rotation=45)
plt.title("Frequency of Text Length Ranges")
plt.xlabel("Text Length Range")
plt.ylabel("Count")
plt.show()

