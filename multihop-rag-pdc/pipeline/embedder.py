import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return _model


def embed_chunks(chunks: list, batch_size: int = 64) -> np.ndarray:
    model = get_model()
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, device="cpu")
    return embeddings.astype("float32")
