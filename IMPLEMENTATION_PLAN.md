# Scalable Multi-Hop RAG with Parallel Vector Indexing — Implementation Plan

## Project Overview

Build a three-phase RAG system for financial question answering that demonstrates:
1. A sequential baseline RAG pipeline (control group)
2. A decomposed multi-hop reasoning framework on top of it (NLP contribution)
3. Parallel infrastructure using multiprocessing to reduce preprocessing latency (PDC contribution)

**Datasets:**

| Dataset | GitHub | HuggingFace |
|---------|--------|-------------|
| FinQA | https://github.com/czyssrs/FinQA | https://huggingface.co/datasets/ibm-research/finqa |
| MultiHop-RAG | https://github.com/yixuantt/MultiHop-RAG | https://huggingface.co/datasets/yixuantt/MultiHopRAG |


**LLM:** Gemini 1.5 Flash (free tier, `gemini-1.5-flash`)  
**Embeddings:** `sentence-transformers` locally (`all-MiniLM-L6-v2`) — CPU-bound, validates PDC claims  
**Vector index:** FAISS (`faiss-cpu`)  
**Environment:** Local Ubuntu — i5-11th gen (4 cores / 8 threads), runs everything including benchmarks

---

## Repository Structure

```
multihop-rag-pdc/
├── IMPLEMENTATION_PLAN.md          ← this file
├── requirements.txt
├── config.py                       ← all constants, paths, hyperparams in one place
│
├── data/
│   ├── loader.py                   ← unified data loader for both datasets
│   ├── finqa_adapter.py            ← normalize FinQA schema
│   └── multihop_adapter.py         ← normalize MultiHop-RAG schema
│
├── pipeline/
│   ├── chunker.py                  ← sequential chunker (baseline)
│   ├── embedder.py                 ← sequential embedder (baseline)
│   ├── indexer.py                  ← sequential FAISS indexer (baseline)
│   ├── retriever.py                ← single-shot retriever (baseline)
│   └── generator.py               ← Gemini API wrapper with rate limiting
│
├── multihop/
│   ├── decomposer.py               ← query decomposition via Gemini
│   ├── hop_runner.py               ← iterative retrieval loop
│   ├── conditioner.py              ← contextual conditioning between hops
│   └── aggregator.py              ← evidence buffer + final generation
│
├── parallel/
│   ├── parallel_chunker.py         ← multiprocessing.Pool chunking
│   ├── parallel_embedder.py        ← parallel embedding generation
│   ├── parallel_indexer.py         ← localized FAISS + merge_from()
│   └── parallel_retriever.py       ← broadcast search across index partitions
│
├── evaluation/
│   ├── metrics.py                  ← EM, F1, latency recording
│   ├── amdahl.py                   ← speedup computation + Amdahl curve fitting
│   └── reporter.py                ← generate results tables + plots
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_eval.ipynb
│   ├── 03_multihop_eval.ipynb
│   └── 04_parallel_benchmark.ipynb
│
└── results/
    ├── latency_logs/
    ├── accuracy_logs/
    └── plots/
```

---

## Config (`config.py`)

Everything hardcoded goes here. No magic numbers anywhere else.

```python
# Paths
FINQA_DATASET = "ibm-research/finqa"          # https://huggingface.co/datasets/ibm-research/finqa
MULTIHOP_DATASET = "yixuantt/MultiHopRAG"     # https://huggingface.co/datasets/yixuantt/MultiHopRAG
INDEX_DIR = "results/indices/"
LOG_DIR = "results/latency_logs/"

# Chunking
CHUNK_SIZE = 512          # tokens
CHUNK_OVERLAP = 100       # tokens

# Embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Retrieval
TOP_K = 5                 # chunks to retrieve per hop

# Gemini
GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_RPM_LIMIT = 15     # free tier: 15 requests/minute
GEMINI_DAILY_LIMIT = 1500

# PDC benchmark — i5-11th gen has 4 cores, so N=4 is the hardware ceiling
PROCESS_COUNTS = [1, 2, 4]
CORPUS_SIZES = [1000, 5000, 20000]  # realistic ceiling given actual dataset sizes

# Evaluation
SMOKE_CORPUS_SIZE = 500   # local dev only
EVAL_SAMPLE_SIZE = 200    # questions to evaluate on (keep quota-safe)
```

---

## Milestone 1 — Environment + Dataset Setup

**Goal:** Working data pipeline, both datasets normalized to the same schema, smoke-test corpus ready.

### 1.1 Dependencies (`requirements.txt`)

```
datasets
faiss-cpu
sentence-transformers
google-generativeai
numpy
pandas
matplotlib
tqdm
tiktoken
```

Install:
```bash
pip install -r requirements.txt
```

On Kaggle, add `faiss-gpu` instead of `faiss-cpu` for faster index building, but keep all multiprocessing logic identical.

### 1.2 Unified Data Schema

Both datasets must be normalized to this structure before any pipeline code touches them:

```python
{
    "id": str,                        # unique question ID
    "question": str,                  # the question
    "answer": str,                    # ground truth answer (string)
    "answer_type": str,               # "numeric" | "entity" | "span"
    "supporting_docs": [              # list of relevant document chunks
        {
            "doc_id": str,
            "title": str,
            "text": str               # raw text of the document/passage
        }
    ],
    "dataset": str                    # "finqa" | "multihop"
}
```

### 1.3 FinQA Adapter (`data/finqa_adapter.py`)

FinQA structure: each example has `pre_text`, `post_text`, a `table`, a `qa` object with `question` and `exe_ans`.

Key normalization steps:
- Linearize the table into a string: `"Row 1: col1 | col2 | col3\nRow 2: ..."` — this becomes part of the document text
- Concatenate `pre_text + linearized_table + post_text` as the single supporting document
- `answer_type` = "numeric" for all FinQA (answers are numbers)
- Use `train` split for corpus building, `test` split for evaluation

```python
def adapt_finqa(example) -> dict:
    table_str = linearize_table(example["table"])
    doc_text = " ".join(example["pre_text"]) + "\n" + table_str + "\n" + " ".join(example["post_text"])
    return {
        "id": example["id"],
        "question": example["qa"]["question"],
        "answer": str(example["qa"]["exe_ans"]),
        "answer_type": "numeric",
        "supporting_docs": [{"doc_id": example["id"], "title": example.get("filename", ""), "text": doc_text}],
        "dataset": "finqa"
    }
```

### 1.4 MultiHop-RAG Adapter (`data/multihop_adapter.py`)

MultiHop-RAG structure: each example has `query`, `answer`, `question_type`, and `supporting_passages` (list of passages with `title` and `body`).

Key normalization steps:
- `supporting_docs` maps directly from `supporting_passages`
- `answer_type` = "entity" for most — keep as-is
- Use `train` split only (the dataset may not have a formal test split; sample 200 for eval)

```python
def adapt_multihop(example) -> dict:
    docs = [
        {"doc_id": f"{example['query_id']}_p{i}", "title": p["title"], "text": p["body"]}
        for i, p in enumerate(example["supporting_passages"])
    ]
    return {
        "id": example["query_id"],
        "question": example["query"],
        "answer": example["answer"],
        "answer_type": "entity",
        "supporting_docs": docs,
        "dataset": "multihop"
    }
```

### 1.5 Unified Loader (`data/loader.py`)

```python
def load_dataset(name: str, split: str = "train", max_samples: int = None) -> list[dict]:
    # loads from HuggingFace, applies adapter, returns list of normalized dicts
    # name: "finqa" | "multihop" | "both"
```

Also build a `build_corpus(samples, corpus_size)` function that extracts all `supporting_docs` from the sample list, deduplicates by `doc_id`, and returns a flat list of `{"doc_id", "title", "text"}` dicts. This is the corpus that gets chunked and indexed.

### 1.6 Smoke Test

Run locally with `SMOKE_CORPUS_SIZE = 500`. Confirm:
- Both adapters produce valid dicts
- Corpus has no duplicate `doc_id`s
- A sample question and its answer print correctly

---

## Milestone 2 — Phase 1: Baseline Sequential RAG

**Goal:** End-to-end working pipeline. Single-threaded. Evaluate on both datasets. Record `T_sequential`.

### 2.1 Chunker (`pipeline/chunker.py`)

Input: list of corpus docs `[{"doc_id", "title", "text"}]`  
Output: list of chunks `[{"chunk_id", "doc_id", "title", "text", "chunk_index"}]`

Algorithm:
- Tokenize with `tiktoken` (`cl100k_base` encoding) for accurate token counting
- Sliding window: advance by `CHUNK_SIZE - CHUNK_OVERLAP` tokens each step
- `chunk_id` = `f"{doc_id}_c{chunk_index}"`
- Preserve `doc_id` and `title` in every chunk for citation tracking

```python
def chunk_documents(docs: list[dict]) -> list[dict]:
    # sequential, single-threaded
    # returns list of chunk dicts
```

### 2.2 Embedder (`pipeline/embedder.py`)

Input: list of chunk dicts  
Output: numpy array of shape `(N, EMBEDDING_DIM)` + the original chunk list (index alignment guaranteed)

```python
from sentence_transformers import SentenceTransformer

def embed_chunks(chunks: list[dict], batch_size: int = 64) -> np.ndarray:
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return embeddings  # shape: (len(chunks), 384)
```

Important: `model.encode()` runs on CPU throughout. Keep `device="cpu"` explicit in all embedding calls — this is what makes the multiprocessing argument valid. Intel Iris Xe is shared memory and irrelevant here.

### 2.3 Indexer (`pipeline/indexer.py`)

Input: embeddings numpy array, chunk list  
Output: FAISS index object + chunk metadata list (positional alignment: `index.search` result `i` → `chunks[i]`)

```python
import faiss

def build_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))
    return index

def save_index(index, chunks, path):
    faiss.write_index(index, f"{path}.faiss")
    # save chunks as JSON for metadata lookup
```

Record wall-clock time wrapping `chunk_documents → embed_chunks → build_index` as `T_sequential`. Use `time.perf_counter()`, not `time.time()`.

### 2.4 Retriever (`pipeline/retriever.py`)

Input: query string, FAISS index, chunk list, top_k  
Output: list of top-k chunk dicts

```python
def retrieve(query: str, index, chunks: list[dict], model, top_k: int = TOP_K) -> list[dict]:
    q_emb = model.encode([query]).astype("float32")
    distances, indices = index.search(q_emb, top_k)
    return [chunks[i] for i in indices[0]]
```

### 2.5 Generator (`pipeline/generator.py`)

Gemini API wrapper with mandatory rate limiting.

```python
import google.generativeai as genai
import time

class GeminiGenerator:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        self._last_call = 0
        self._min_interval = 60 / GEMINI_RPM_LIMIT  # 4 seconds between calls

    def generate(self, prompt: str) -> str:
        elapsed = time.perf_counter() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        response = self.model.generate_content(prompt)
        self._last_call = time.perf_counter()
        return response.text
```

### 2.6 Baseline Pipeline (single-shot)

```python
def run_baseline(question: str, index, chunks, embed_model, generator) -> str:
    retrieved = retrieve(question, index, chunks, embed_model)
    context = "\n\n".join([f"[{c['title']}]: {c['text']}" for c in retrieved])
    prompt = f"""Answer the following financial question using only the provided context.
    
Context:
{context}

Question: {question}

Answer:"""
    return generator.generate(prompt)
```

### 2.7 Evaluation (`evaluation/metrics.py`)

```python
def exact_match(pred: str, gold: str) -> float:
    # normalize: lowercase, strip punctuation, strip whitespace
    # return 1.0 if match, 0.0 otherwise
    # for numeric: also try float comparison with tolerance

def token_f1(pred: str, gold: str) -> float:
    # tokenize both, compute precision/recall/F1 on token overlap
    # standard SQuAD-style F1

def evaluate_dataset(questions, run_fn) -> dict:
    # runs run_fn(q) for each question
    # returns {"em": float, "f1": float, "avg_latency_ms": float}
```

Evaluate on:
- 200 FinQA test questions → report Exact Match (numeric answers)
- 200 MultiHop-RAG questions → report F1 (entity/span answers)

**Key outputs from M2:**
- `T_sequential` (preprocessing time at corpus sizes 1k, 10k, 100k)
- Baseline EM on FinQA
- Baseline F1 on MultiHop-RAG
- Average per-query latency

---

## Milestone 3 — Phase 2: Decomposed Multi-Hop RAG

**Goal:** Multi-hop reasoning loop on top of the M2 index. Accuracy improves, latency worsens — that's the expected and correct result. This motivates M4.

All indexing infrastructure is identical to M2. Only the query processing changes.

### 3.1 Query Decomposer (`multihop/decomposer.py`)

Takes a complex question, returns an ordered list of sub-queries with dependency annotations.

Prompt design (few-shot, financial domain):

```python
DECOMPOSE_PROMPT = """You are a financial reasoning assistant. Break down the complex question into simple, ordered sub-questions.

Rules:
- Each sub-question must be answerable from a single piece of evidence
- If sub-question B needs the answer from sub-question A, mark it as DEPENDS_ON: 1
- Output ONLY a JSON array, no explanation

Example:
Question: "What was the goodwill adjustment for the company that acquired Hittite Microwave in 2014?"
Output:
[
  {"id": 1, "question": "Which company acquired Hittite Microwave in 2014?", "depends_on": null},
  {"id": 2, "question": "What was the goodwill adjustment for [ANSWER_1] in 2014?", "depends_on": 1}
]

Question: {question}
Output:"""

def decompose(question: str, generator: GeminiGenerator) -> list[dict]:
    prompt = DECOMPOSE_PROMPT.format(question=question)
    raw = generator.generate(prompt)
    # parse JSON from response
    # validate structure
    # return list of {"id", "question", "depends_on"}
```

Edge case handling:
- If Gemini returns malformed JSON, retry once with a stricter prompt
- If decomposition yields only 1 sub-query, fall back to baseline single-shot (log this)
- Cap at 4 hops maximum — beyond that, truncate and log a warning

### 3.2 Hop Runner (`multihop/hop_runner.py`)

Executes sub-queries in dependency order, substituting intermediate answers into downstream queries.

```python
def run_hops(sub_queries: list[dict], index, chunks, embed_model, generator) -> dict:
    """
    Returns:
    {
        "intermediate_answers": {sub_query_id: answer_str},
        "retrieved_chunks": {sub_query_id: [chunk_dicts]},
        "hop_count": int
    }
    """
    answers = {}
    retrieved = {}
    
    for sq in sub_queries:  # already ordered by dependency
        # substitute [ANSWER_N] placeholders with actual answers
        query_text = sq["question"]
        if sq["depends_on"] is not None:
            parent_answer = answers[sq["depends_on"]]
            query_text = query_text.replace(f"[ANSWER_{sq['depends_on']}]", parent_answer)
        
        # retrieve + generate intermediate answer
        chunks_retrieved = retrieve(query_text, index, chunks, embed_model)
        context = build_context_string(chunks_retrieved)
        answer = generator.generate(HOP_ANSWER_PROMPT.format(context=context, question=query_text))
        
        answers[sq["id"]] = answer
        retrieved[sq["id"]] = chunks_retrieved
    
    return {"intermediate_answers": answers, "retrieved_chunks": retrieved, "hop_count": len(sub_queries)}
```

### 3.3 Contextual Conditioner (`multihop/conditioner.py`)

The key NLP contribution. Reformulates the next sub-query using the previous answer to make it more semantically precise before embedding.

```python
CONDITION_PROMPT = """Given the intermediate finding, rewrite the follow-up question to be more specific and searchable.

Intermediate finding: {prior_answer}
Follow-up question: {next_question}

Rewritten question (one sentence, specific, searchable):"""

def condition_query(prior_answer: str, next_question: str, generator: GeminiGenerator) -> str:
    # returns a reformulated, more specific query string
    # this is what gets embedded for the next hop's FAISS search
```

Note: conditioning adds one more Gemini call per hop. With 4 hops and 2 sub-queries each that's ~8 calls per question. At 15 RPM, pace your eval runs accordingly.

### 3.4 Evidence Aggregator + Final Generation (`multihop/aggregator.py`)

```python
FINAL_ANSWER_PROMPT = """You are a financial analyst. Answer the original question using the chain of evidence gathered across multiple reasoning steps.

Original question: {question}

Evidence gathered:
{evidence_chain}

Provide a precise, grounded answer. If the answer is numeric, give only the number. If it is an entity, give only the entity name.

Answer:"""

def aggregate_and_answer(question: str, hop_results: dict, generator: GeminiGenerator) -> str:
    # build evidence_chain string from all intermediate answers + their source chunks
    evidence_lines = []
    for hop_id, answer in hop_results["intermediate_answers"].items():
        sources = [c["title"] for c in hop_results["retrieved_chunks"][hop_id]]
        evidence_lines.append(f"Step {hop_id}: {answer} (sources: {', '.join(sources)})")
    evidence_chain = "\n".join(evidence_lines)
    return generator.generate(FINAL_ANSWER_PROMPT.format(question=question, evidence_chain=evidence_chain))
```

### 3.5 Full Multi-Hop Pipeline

```python
def run_multihop(question: str, index, chunks, embed_model, generator) -> dict:
    sub_queries = decompose(question, generator)
    if len(sub_queries) <= 1:
        answer = run_baseline(question, index, chunks, embed_model, generator)
        return {"answer": answer, "hop_count": 1, "fallback": True}
    
    hop_results = run_hops(sub_queries, index, chunks, embed_model, generator)
    answer = aggregate_and_answer(question, hop_results, generator)
    return {"answer": answer, "hop_count": hop_results["hop_count"], "fallback": False}
```

### 3.6 Evaluation

Same metrics as M2. Compare directly:

| Metric | Phase 1 Baseline | Phase 2 Multi-Hop |
|--------|-----------------|-------------------|
| FinQA EM | ? | ? |
| MultiHop-RAG F1 | ? | ? |
| Avg query latency | ? | ? |

Expected result: MultiHop-RAG F1 improves significantly. FinQA EM improves modestly (it's mostly single-hop). Query latency increases — this is the correct and expected result that motivates Phase 3.

---

## Milestone 4 — Phase 3: Parallel Infrastructure

**Goal:** Parallelize chunking → embedding → indexing using `multiprocessing`. Measure speedup across N ∈ {1, 2, 4, 8} and corpus sizes ∈ {1k, 10k, 100k}. Plot empirical vs Amdahl's Law theoretical curve.

**Critical:** Run this on Kaggle with CPU-only embedding (force `device="cpu"` even on T4). The PDC contribution is about CPU multiprocessing bypassing the GIL — not GPU acceleration.

### 4.1 Parallel Chunker (`parallel/parallel_chunker.py`)

```python
from multiprocessing import Pool, get_context

def _chunk_shard(shard: list[dict]) -> list[dict]:
    """Worker function — runs in isolated process. Must be top-level (not a lambda)."""
    from pipeline.chunker import chunk_documents
    return chunk_documents(shard)

def parallel_chunk(docs: list[dict], n_processes: int) -> list[dict]:
    # partition docs into n_processes equal shards
    shard_size = len(docs) // n_processes
    shards = [docs[i*shard_size:(i+1)*shard_size] for i in range(n_processes)]
    # handle remainder
    if len(docs) % n_processes:
        shards[-1].extend(docs[n_processes * shard_size:])
    
    ctx = get_context("spawn")  # explicit spawn — safer than fork with PyTorch loaded
    with ctx.Pool(processes=n_processes) as pool:
        results = pool.map(_chunk_shard, shards)
    
    return [chunk for shard_result in results for chunk in shard_result]
```

### 4.2 Parallel Embedder (`parallel/parallel_embedder.py`)

```python
def _embed_shard(args) -> np.ndarray:
    """Worker function. Loads its own model instance — no shared state."""
    shard_texts, device = args
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    return model.encode(shard_texts, batch_size=64, show_progress_bar=False)

def parallel_embed(chunks: list[dict], n_processes: int, device: str = "cpu") -> np.ndarray:
    texts = [c["text"] for c in chunks]
    shard_size = len(texts) // n_processes
    shards = [(texts[i*shard_size:(i+1)*shard_size], device) for i in range(n_processes)]
    if len(texts) % n_processes:
        remainder = texts[n_processes * shard_size:]
        shards[-1] = (shards[-1][0] + remainder, device)
    
    ctx = get_context("spawn")
    with ctx.Pool(processes=n_processes) as pool:
        results = pool.map(_embed_shard, shards)
    
    return np.vstack(results)
```

Note: each worker loads its own `SentenceTransformer` model instance — no shared state, no GIL contention. Memory overhead is `n_processes × model_size` (~90MB each for MiniLM), so 4 processes = ~360MB. Fine on any modern machine.

### 4.3 Parallel Indexer (`parallel/parallel_indexer.py`)

The critical part. Each worker builds its own local FAISS index, main process merges.

```python
def _build_local_index(args) -> str:
    """
    Worker: builds a local IndexFlatL2 from its embedding shard.
    Saves to a temp file, returns the file path.
    Must return a path (not the index object) — FAISS indices are not picklable.
    """
    embeddings_shard, index_path = args
    import faiss, tempfile
    dim = embeddings_shard.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_shard.astype("float32"))
    faiss.write_index(index, index_path)
    return index_path

def parallel_build_index(embeddings: np.ndarray, n_processes: int, tmp_dir: str = "/tmp") -> faiss.Index:
    shard_size = len(embeddings) // n_processes
    args = []
    for i in range(n_processes):
        shard = embeddings[i*shard_size:(i+1)*shard_size]
        if i == n_processes - 1:
            shard = embeddings[i*shard_size:]  # include remainder in last shard
        path = f"{tmp_dir}/faiss_shard_{i}.index"
        args.append((shard, path))
    
    ctx = get_context("spawn")
    with ctx.Pool(processes=n_processes) as pool:
        shard_paths = pool.map(_build_local_index, args)
    
    # merge all shard indices into one
    merged = faiss.read_index(shard_paths[0])
    for path in shard_paths[1:]:
        shard_index = faiss.read_index(path)
        faiss.merge_from(merged, shard_index, merged.ntotal)
    
    return merged
```

### 4.4 Parallel Retriever (`parallel/parallel_retriever.py`)

For query-time parallelism. Partitions the index logically, broadcasts the query, merges results.

```python
def _search_partition(args) -> tuple:
    query_emb, index_path, top_k = args
    import faiss
    index = faiss.read_index(index_path)
    distances, indices = index.search(query_emb, top_k)
    return distances[0], indices[0]

def parallel_retrieve(query_emb: np.ndarray, shard_paths: list[str], top_k: int) -> list[int]:
    args = [(query_emb, path, top_k) for path in shard_paths]
    ctx = get_context("spawn")
    with ctx.Pool(processes=len(shard_paths)) as pool:
        results = pool.map(_search_partition, args)
    
    # merge: collect all (distance, global_index) pairs, sort, return top_k
    all_results = []
    for shard_id, (distances, indices) in enumerate(results):
        for d, i in zip(distances, indices):
            global_idx = shard_id * (index_total_size // len(shard_paths)) + i  # adjust offset
            all_results.append((d, global_idx))
    
    all_results.sort(key=lambda x: x[0])
    return [idx for _, idx in all_results[:top_k]]
```

### 4.5 Benchmark Runner

The benchmark is the core PDC deliverable. Must be clean, reproducible, and logged.

```python
def run_benchmark(corpus_size: int, n_processes: int, docs: list[dict]) -> dict:
    """
    Run full parallel pipeline for given corpus_size and n_processes.
    Returns timing breakdown and total time.
    """
    import time
    
    corpus = docs[:corpus_size]
    timings = {}
    
    # chunking
    t0 = time.perf_counter()
    chunks = parallel_chunk(corpus, n_processes)
    timings["chunking_s"] = time.perf_counter() - t0
    
    # embedding (CPU-only, always)
    t0 = time.perf_counter()
    embeddings = parallel_embed(chunks, n_processes, device="cpu")
    timings["embedding_s"] = time.perf_counter() - t0
    
    # indexing
    t0 = time.perf_counter()
    index = parallel_build_index(embeddings, n_processes)
    timings["indexing_s"] = time.perf_counter() - t0
    
    timings["total_s"] = sum(timings.values())
    timings["n_processes"] = n_processes
    timings["corpus_size"] = corpus_size
    timings["n_chunks"] = len(chunks)
    
    return timings

def run_full_benchmark(docs: list[dict]):
    results = []
    for corpus_size in CORPUS_SIZES:
        for n_proc in PROCESS_COUNTS:
            print(f"Running: corpus={corpus_size}, n_proc={n_proc}")
            result = run_benchmark(corpus_size, n_proc, docs)
            results.append(result)
            # save incrementally — if kernel dies mid-run, partial results are preserved
            pd.DataFrame(results).to_csv(f"{LOG_DIR}/benchmark_results.csv", index=False)
    return pd.DataFrame(results)
```

Run N=1 first for each corpus size — this is your `T_sequential` baseline for Amdahl's Law. Nine total runs (3 process counts × 3 corpus sizes), all on local machine.

---

## Milestone 5 — Evaluation + Amdahl's Law Analysis

### 5.1 Speedup Computation (`evaluation/amdahl.py`)

```python
def compute_speedup(df: pd.DataFrame) -> pd.DataFrame:
    """
    df has columns: n_processes, corpus_size, total_s
    For each corpus_size, compute speedup = T(N=1) / T(N)
    """
    results = []
    for corpus_size in df["corpus_size"].unique():
        subset = df[df["corpus_size"] == corpus_size].copy()
        t_seq = subset[subset["n_processes"] == 1]["total_s"].values[0]
        subset["speedup_empirical"] = t_seq / subset["total_s"]
        results.append(subset)
    return pd.concat(results)

def amdahl_theoretical(N: int, S: float) -> float:
    """
    S = sequential fraction (0 to 1)
    N = number of processors
    Returns theoretical speedup
    """
    return 1 / (S + (1 - S) / N)

def fit_sequential_fraction(empirical_speedups: dict) -> float:
    """
    empirical_speedups: {N: speedup}
    Fit S by minimizing MSE between empirical and Amdahl theoretical.
    Returns estimated S (sequential fraction).
    """
    from scipy.optimize import minimize_scalar
    
    def mse(S):
        return sum((amdahl_theoretical(N, S) - sp)**2 for N, sp in empirical_speedups.items())
    
    result = minimize_scalar(mse, bounds=(0.01, 0.99), method='bounded')
    return result.x
```

### 5.2 Results Plots (`evaluation/reporter.py`)

Generate three plots:

**Plot 1 — Speedup vs N (Amdahl's Law comparison):**
- X-axis: N ∈ {1, 2, 4, 8}
- Y-axis: speedup
- Lines: empirical speedup (solid) + Amdahl theoretical (dashed) for each corpus size

**Plot 2 — Preprocessing latency vs corpus size:**
- X-axis: corpus size (log scale)
- Y-axis: total wall-clock seconds
- Lines: one per N value

**Plot 3 — Accuracy comparison (NLP contribution):**
- Grouped bar chart: FinQA EM and MultiHop-RAG F1
- Groups: Phase 1 Baseline vs Phase 2 Multi-Hop

```python
import matplotlib.pyplot as plt

def plot_amdahl(df_speedup, S_fitted):
    fig, ax = plt.subplots(figsize=(8, 5))
    N_values = [1, 2, 4, 8]
    theoretical = [amdahl_theoretical(N, S_fitted) for N in N_values]
    
    for corpus_size, group in df_speedup.groupby("corpus_size"):
        ax.plot(group["n_processes"], group["speedup_empirical"], marker="o", label=f"Empirical ({corpus_size:,} docs)")
    
    ax.plot(N_values, theoretical, "k--", label=f"Amdahl (S={S_fitted:.2f})", linewidth=1.5)
    ax.set_xlabel("Number of processes (N)")
    ax.set_ylabel("Speedup (T_seq / T_par)")
    ax.set_title("Empirical vs Theoretical Speedup (Amdahl's Law)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig("results/plots/amdahl_speedup.png", dpi=150, bbox_inches="tight")
```

### 5.3 Final Results Table

Produce this table for the paper:

| Phase | FinQA EM | MultiHop F1 | Preprocessing Time (10k docs, N=1) | Preprocessing Time (10k docs, N=8) | Speedup |
|-------|----------|-------------|-------------------------------------|-------------------------------------|---------|
| Phase 1 — Sequential Baseline | ? | ? | ? s | — | 1× |
| Phase 2 — Multi-Hop (Sequential Infra) | ? | ? | ? s | — | 1× |
| Phase 3 — Multi-Hop (Parallel Infra) | same as P2 | same as P2 | — | ? s | ?× |

Note: Phase 3 accuracy numbers are identical to Phase 2 — the parallel changes are infrastructure only. The contribution is speed, not accuracy. Make this explicit in the paper.

---

## Notebook Structure

Four Jupyter notebooks, each independently runnable after the previous saves its outputs. Run all locally.

### `01_data_exploration.ipynb`
- Load both datasets, run adapters, print samples
- Build corpus at sizes 1k, 5k, 20k — save as JSON to `data/corpus/`
- Verify chunk counts, token distributions, embedding shapes

### `02_baseline_eval.ipynb`
- Load 1k corpus, run sequential pipeline, save FAISS index
- Evaluate 200 questions from each dataset
- Save: `results/accuracy_logs/baseline_results.json`
- Record and save `T_sequential` per corpus size to `results/latency_logs/`

### `03_multihop_eval.ipynb`
- Load same index from notebook 02
- Run multi-hop pipeline on same 200 questions
- Save: `results/accuracy_logs/multihop_results.json`
- Print accuracy comparison table vs baseline

### `04_parallel_benchmark.ipynb`
- Run `run_full_benchmark()` — 9 total runs (N ∈ {1,2,4} × corpus ∈ {1k,5k,20k})
- Save incrementally to `results/latency_logs/benchmark_results.csv`
- Generate and save all 3 plots to `results/plots/`

---

## Gemini Quota Management

At 15 RPM and 1500/day:
- Phase 1 eval: 200 questions × 1 call = 200 calls. Fine.
- Phase 2 eval: 200 questions × ~4 calls (decompose + 2 hops + aggregate) = ~800 calls. Fine.
- Leave 500 calls/day buffer for debugging and re-runs.
- Add a `DailyCallTracker` class that writes call count to disk and raises an exception if approaching 1400/day.

```python
class DailyCallTracker:
    def __init__(self, limit=1400, log_path="results/gemini_calls.json"):
        self.limit = limit
        self.log_path = log_path
    
    def check_and_increment(self):
        # load today's count from disk, increment, save, raise if over limit
```

---

## Critical Implementation Notes

1. **Never pass a FAISS index object through multiprocessing IPC.** FAISS indices are not picklable. Always save to disk and pass file paths between processes.

2. **Always use `get_context("spawn")` for multiprocessing.** The default `fork` context with PyTorch loaded can deadlock. Spawn is safer and more explicit.

3. **`T_sequential` measurement must use N=1 in the parallel framework**, not the raw sequential code. Otherwise speedup numbers are invalid — you'd be comparing different code paths.

4. **Chunk metadata and embeddings must stay positionally aligned.** If you shuffle or reorder chunks during parallel processing, `index.search` will return wrong chunk metadata. After `parallel_chunk`, sort by `chunk_id` before embedding.

5. **Gemini responses are not always valid JSON.** The decomposer must have a try/except with a fallback re-prompt and a final fallback to single-shot baseline. Never let a JSON parse error crash an eval run.

6. **Log everything incrementally.** After every question evaluation, append to a JSONL file. Never hold results only in memory — if the kernel dies mid-run you lose everything.

7. **The parallel similarity search (4.4) is optional** if time is tight. The main PDC contribution is the indexing speedup. The query-time parallel search is a bonus that strengthens the argument.
