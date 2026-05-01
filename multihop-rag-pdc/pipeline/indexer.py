import os
import json
import numpy as np
import faiss
from config import EMBEDDING_DIM


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))
    return index


def save_index(index: faiss.IndexFlatL2, chunks: list, path_prefix: str):
    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
    faiss.write_index(index, f"{path_prefix}.faiss")
    with open(f"{path_prefix}_chunks.json", "w") as f:
        json.dump(chunks, f)


def load_index(path_prefix: str):
    index = faiss.read_index(f"{path_prefix}.faiss")
    with open(f"{path_prefix}_chunks.json") as f:
        chunks = json.load(f)
    return index, chunks
