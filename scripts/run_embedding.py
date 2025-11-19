import sys
from pathlib import Path
import time
# set root path one step back
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from sentence_transformers import SentenceTransformer
from utility.file_utils import get_csv_dataframe  # noqa: E402
# Load a pretrained embedding model (no API required)
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str) -> list[float]:
    """
    Get embedding for a single text using a sentence-transformer model.
    """
    embedding = model.encode(text, show_progress_bar=False)
    return embedding.tolist()


"""
# Example usage
# Load your CSV
df = get_csv_dataframe("data/Merged_dataset.csv")  # assuming this function returns a pd.DataFrame

max_len = 0
min_len = 1e6
k = 0
print("Dataframe loaded with", len(df), "rows.")
for i in range(len(df)):
    text = df['text'][i]
    len_text = len(text)
    if len_text < 30:
        k += 1
       
print("number of very small texts:", k)

start_time = time.time()
# Get the first element in the 'text_content' column
for i in range(100):
    text = df['text'][i]
    emb = get_embedding(text)
    #print("Embedding dimension:", len(emb))
    #print(emb[:10], "…")

end_time = time.time()
print(f"Time taken for 100 embeddings: {end_time - start_time:.2f} seconds")

"""