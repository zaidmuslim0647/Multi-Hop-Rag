import numpy as np
from multiprocessing import get_context


def _search_partition(args) -> tuple:
    query_emb, index_path, top_k = args
    import faiss
    index = faiss.read_index(index_path)
    distances, indices = index.search(query_emb, top_k)
    return distances[0].tolist(), indices[0].tolist()


def parallel_retrieve(query_emb: np.ndarray, shard_paths: list, shard_offsets: list, top_k: int) -> list:
    """
    query_emb: (1, dim) float32
    shard_paths: list of FAISS shard file paths
    shard_offsets: global start index for each shard
    Returns list of global indices (top_k best across all shards).
    """
    args = [(query_emb, path, top_k) for path in shard_paths]
    ctx = get_context("spawn")
    with ctx.Pool(processes=len(shard_paths)) as pool:
        results = pool.map(_search_partition, args)

    all_results = []
    for shard_id, (distances, indices) in enumerate(results):
        offset = shard_offsets[shard_id]
        for d, i in zip(distances, indices):
            if i >= 0:
                all_results.append((d, offset + i))

    all_results.sort(key=lambda x: x[0])
    return [idx for _, idx in all_results[:top_k]]
