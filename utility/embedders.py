from sentence_transformers import SentenceTransformer
import numpy as np
from meta_data_utils import meta_data_vectorrize

# lav din model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def bert_encode(texts):
    """Needs a array, in which each entry is the text"""
    embs = model.encode(texts, convert_to_numpy=True)

    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / norms

    return embs


def bert_and_metadata_encode(texts):
    """
    Returns an array where each row = [metadata_vector + bert_vector]
    """
    bert_embs = bert_encode(texts)

    metadata_embs = np.array([meta_data_vectorrize(t) for t in texts], dtype=float)

    embs = np.concatenate([metadata_embs, bert_embs], axis=1)

    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / norms

    return embs


if __name__ == "__main__":
    x = [
        "hello world dadf dafadf adf df adfad ndf kadfhi badfbhadfgufgudvadf gvuadgvuagvuadf gvuadfooadgvf oadf ogadf gvuadfo adfegvuoadf dafgadvgadf gvadf padf vguad gvupagvupdfgvupdf gvupfag adf hapgadf gadfd pgadfgvagvuad gvupdgvpagvuagvupad gvupadf dafgvgafd vgpadpvg adgvpadfgvvavgad vgpadfpgadf ad vgadf gvadf gvpadsfvgadfgvup adgvupdf gvupadf gvpadfvpadf",
        "machine learning is fun",
    ]
    embed = bert_and_metadata_encode(x)
    print(embed.shape)
    print(embed[0])
