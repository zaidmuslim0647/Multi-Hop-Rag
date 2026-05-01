import numpy as np
from config import TOP_K


def retrieve(query: str, index, chunks: list, model, top_k: int = TOP_K) -> list:
    q_emb = model.encode([query], device="cpu").astype("float32")
    distances, indices = index.search(q_emb, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]
