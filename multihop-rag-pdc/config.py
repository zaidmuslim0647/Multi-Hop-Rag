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
GEMINI_CALLS_LOG = os.path.join(BASE_DIR, "results/gemini_calls.json")

# Chunking
CHUNK_SIZE = 512       # tokens
CHUNK_OVERLAP = 100    # tokens

# Embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Retrieval
TOP_K = 5

# Gemini
GEMINI_MODEL = "gemini-1.5-flash"
GEMINI_RPM_LIMIT = 15      # free tier: 15 requests/minute
GEMINI_DAILY_LIMIT = 1500
GEMINI_SAFE_DAILY_LIMIT = 1400  # stop before hitting the real cap

# PDC benchmark — i5-11th gen: 4 cores / 8 threads
PROCESS_COUNTS = [1, 2, 4]
CORPUS_SIZES = [1000, 5000, 20000]

# Evaluation
SMOKE_CORPUS_SIZE = 500
EVAL_SAMPLE_SIZE = 200

# Multi-hop
MAX_HOPS = 4
