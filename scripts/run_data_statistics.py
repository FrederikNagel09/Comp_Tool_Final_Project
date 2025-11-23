"""

Run with:
    python scripts/run_data_statistics.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
# Visualizations
# ----------------------
title_text = "Histogram of Text Lengths"
x_axis_label = "Text Length"
y_axis_label = "Frequency"
number_of_bins = 50
histogram_color = "skyblue"
linestyle = "--"
line_color = "black"

graph_name = "text_length_histogram"

# Minumum and Maximum text lengths we cut off
min_len = 100
max_len = 1000

# Initialize plot
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# Plot histogram
sns.histplot(text_lengths, bins=number_of_bins, kde=True, color=histogram_color)

# Add vertical lines for min and max lengths
plt.axvline(min_len, color=line_color, linestyle=linestyle)
plt.axvline(max_len, color=line_color, linestyle=linestyle)

plt.title(title_text)
plt.xlabel(x_axis_label)
plt.ylabel(y_axis_label)

plt.savefig(f"graphs/{graph_name}.png")
