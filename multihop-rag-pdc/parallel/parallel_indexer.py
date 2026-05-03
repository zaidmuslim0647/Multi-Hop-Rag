import os
import tempfile
import numpy as np
import faiss
from multiprocessing import get_context


def _build_local_index(args) -> str:
    embeddings_shard, index_path = args
    import faiss as _faiss
    dim = embeddings_shard.shape[1]
    index = _faiss.IndexFlatL2(dim)
    index.add(embeddings_shard.astype("float32"))
    _faiss.write_index(index, index_path)
    return index_path


def parallel_build_index(embeddings: np.ndarray, n_processes: int, tmp_dir: str = None) -> faiss.Index:
    if tmp_dir is None:
        tmp_dir = tempfile.gettempdir()

    if n_processes == 1:
        return _build_single(embeddings)

    shard_size = len(embeddings) // n_processes
    args = []
    for i in range(n_processes):
        if i < n_processes - 1:
            shard = embeddings[i * shard_size : (i + 1) * shard_size]
        else:
            shard = embeddings[i * shard_size :]  # includes remainder
        path = os.path.join(tmp_dir, f"faiss_shard_{i}.index")
        args.append((shard, path))

    ctx = get_context("spawn")
    with ctx.Pool(processes=n_processes) as pool:
        shard_paths = pool.map(_build_local_index, args)

    # merge all shards — FAISS indices are not picklable; we pass file paths.
    # IndexFlatL2 has no merge_from(), so reconstruct vectors and re-add.
    merged = faiss.read_index(shard_paths[0])
    os.remove(shard_paths[0])
    for path in shard_paths[1:]:
        shard_index = faiss.read_index(path)
        if shard_index.ntotal > 0:
            shard_vecs = shard_index.reconstruct_n(0, shard_index.ntotal)
            merged.add(shard_vecs)
        os.remove(path)

    return merged


def _build_single(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))
    return index
