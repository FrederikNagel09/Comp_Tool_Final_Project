import numpy as np
from meta_data_utils import calculate_metadata
from file_utils import get_csv_dataframe


def embed_text(text: str, label: int):
    """convert a text into a vector with the last dimension being the label"""
    meta_data = calculate_metadata(text)
    l = list(meta_data.values())
    l.append(label)
    return np.array(l, dtype=np.float64)


def preprocess_data(path: str):
    """Open csv file and convert all texts to vectors and combine into one matrix"""
    df = get_csv_dataframe(path)
    N = df.height

    # embed the first data entry to get the size of the vector
    text, label = df.row(0)
    vect = embed_text(text, label)

    # pre-allocate memory
    A = np.zeros((N, vect.size))
    A[0, :] = vect

    print("processing data")
    for i in range(1, N):
        text, label = df.row(i)
        vect = embed_text(text, label)
        A[i, :] = vect

        print(f"\r{i}", end="", flush=True)

    np.save("data/A.npy", A)
