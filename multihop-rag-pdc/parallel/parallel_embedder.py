import numpy as np
from multiprocessing import get_context
from config import EMBEDDING_MODEL


def _embed_shard(args) -> np.ndarray:
    shard_texts, device = args
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    return model.encode(shard_texts, batch_size=64, show_progress_bar=False).astype("float32")


def parallel_embed(chunks: list, n_processes: int, device: str = "cpu") -> np.ndarray:
    texts = [c["text"] for c in chunks]

    if n_processes == 1:
        return _embed_shard((texts, device))

    shard_size = len(texts) // n_processes
    shards = [
        (texts[i * shard_size : (i + 1) * shard_size], device)
        for i in range(n_processes)
    ]
    # remainder into last shard
    if len(texts) % n_processes:
        remainder = texts[n_processes * shard_size :]
        shards[-1] = (shards[-1][0] + remainder, device)

    ctx = get_context("spawn")
    with ctx.Pool(processes=n_processes) as pool:
        results = pool.map(_embed_shard, shards)

    return np.vstack(results)
