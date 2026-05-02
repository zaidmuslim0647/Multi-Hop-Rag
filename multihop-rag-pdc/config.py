import os

# Dataset identifiers (HuggingFace)
FINQA_DATASET = "ibm-research/finqa"
MULTIHOP_DATASET = "yixuantt/MultiHopRAG"

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(BASE_DIR, "results/indices")
LOG_DIR = os.path.join(BASE_DIR, "results/latency_logs")
ACCURACY_DIR = os.path.join(BASE_DIR, "results/accuracy_logs")
PLOT_DIR = os.path.join(BASE_DIR, "results/plots")
CORPUS_DIR = os.path.join(BASE_DIR, "data/corpus")
LLM_CALLS_LOG = os.path.join(BASE_DIR, "results/llm_calls.json")

# Chunking
CHUNK_SIZE = 512       # tokens
CHUNK_OVERLAP = 100    # tokens

# Embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Retrieval
TOP_K = 3

# Local LLM (Ollama, OpenAI-compatible API)
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_API_KEY = "ollama"  # placeholder — Ollama ignores it but the OpenAI client requires one

# PDC benchmark — i5-11th gen: 4 cores / 8 threads
PROCESS_COUNTS = [1, 2, 4]
CORPUS_SIZES = [1000, 5000, 20000]

# Evaluation
SMOKE_CORPUS_SIZE = 500
EVAL_SAMPLE_SIZE = 200

# Multi-hop
MAX_HOPS = 2
