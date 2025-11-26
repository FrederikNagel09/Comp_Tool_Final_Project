import torch
from torch.utils.data import Dataset


class LoadDataset(Dataset):
    def __init__(self, df):
        """
        df: Polars DataFrame
        """
        # Extract columns as Python lists
        self.ids = df["id"].to_list()
        self.y = [float(v) for v in df["generated"].to_list()]

        feature_cols = [
            "word_count",
            "character_count",
            "lexical_diversity",
            "avg_sentence_length",
            "avg_word_length",
            "flesch_reading_ease",
            "gunning_fog_index",
            "punctuation_ratio",
        ]

        self.features = df.select(feature_cols).to_numpy().astype("float32")
        self.embeddings = df["embedding"].to_list()  # each entry already a Python list

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        emb = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        feats = torch.tensor(self.features[idx], dtype=torch.float32)

        x = torch.cat([emb, feats], dim=0)  # final input vector
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        id = self.ids[idx]

        return x, y, id
