from multiprocessing import get_context


def _chunk_shard(shard: list) -> list:
    from pipeline.chunker import chunk_documents
    return chunk_documents(shard)


def parallel_chunk(docs: list, n_processes: int) -> list:
    if n_processes == 1:
        return _chunk_shard(docs)

    shard_size = len(docs) // n_processes
    shards = [docs[i * shard_size : (i + 1) * shard_size] for i in range(n_processes)]
    # remainder goes into last shard
    if len(docs) % n_processes:
        shards[-1].extend(docs[n_processes * shard_size :])

    ctx = get_context("spawn")
    with ctx.Pool(processes=n_processes) as pool:
        results = pool.map(_chunk_shard, shards)

    chunks = [chunk for shard_result in results for chunk in shard_result]
    # sort by chunk_id for positional alignment guarantee
    chunks.sort(key=lambda c: c["chunk_id"])
    return chunks
