import time
import os
import pandas as pd
from parallel.parallel_chunker import parallel_chunk
from parallel.parallel_embedder import parallel_embed
from parallel.parallel_indexer import parallel_build_index
from config import CORPUS_SIZES, PROCESS_COUNTS, LOG_DIR


def run_benchmark(corpus_size: int, n_processes: int, docs: list) -> dict:
    corpus = docs[:corpus_size]
    timings = {}

    t0 = time.perf_counter()
    chunks = parallel_chunk(corpus, n_processes)
    timings["chunking_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    embeddings = parallel_embed(chunks, n_processes, device="cpu")
    timings["embedding_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    parallel_build_index(embeddings, n_processes)
    timings["indexing_s"] = time.perf_counter() - t0

    timings["total_s"] = timings["chunking_s"] + timings["embedding_s"] + timings["indexing_s"]
    timings["n_processes"] = n_processes
    timings["corpus_size"] = corpus_size
    timings["n_chunks"] = len(chunks)

    return timings


def run_full_benchmark(docs: list, corpus_sizes: list = None, process_counts: list = None) -> pd.DataFrame:
    if corpus_sizes is None:
        corpus_sizes = CORPUS_SIZES
    if process_counts is None:
        process_counts = PROCESS_COUNTS

    os.makedirs(LOG_DIR, exist_ok=True)
    out_path = os.path.join(LOG_DIR, "benchmark_results.csv")
    results = []

    for corpus_size in corpus_sizes:
        for n_proc in process_counts:
            print(f"Running benchmark: corpus_size={corpus_size}, n_processes={n_proc}")
            result = run_benchmark(corpus_size, n_proc, docs)
            results.append(result)
            # save incrementally — preserves partial results if process dies
            pd.DataFrame(results).to_csv(out_path, index=False)
            print(f"  total_s={result['total_s']:.2f}  chunks={result['n_chunks']}")

    return pd.DataFrame(results)
