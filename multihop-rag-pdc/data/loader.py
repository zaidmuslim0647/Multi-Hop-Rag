import json
import os
import urllib.request
from datasets import load_dataset as hf_load_dataset
from data.finqa_adapter import adapt_finqa
from data.multihop_adapter import adapt_multihop

# FinQA GitHub raw JSON URLs — avoids the broken HuggingFace dataset script
_FINQA_URLS = {
    "train": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json",
    "test":  "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json",
    "dev":   "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/dev.json",
}


def _finqa_cache_path(split: str) -> str:
    from config import BASE_DIR
    cache_dir = os.path.join(BASE_DIR, "data", "raw", "finqa")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{split}.json")


def _load_finqa_raw(split: str) -> list:
    """Download FinQA JSON from GitHub (cached locally after first download)."""
    split = split if split in _FINQA_URLS else "train"
    cache = _finqa_cache_path(split)
    if not os.path.exists(cache):
        url = _FINQA_URLS[split]
        print(f"Downloading FinQA {split} from GitHub → {cache}")
        urllib.request.urlretrieve(url, cache)
    with open(cache) as f:
        return json.load(f)


def load_dataset(name: str, split: str = "train", max_samples: int = None) -> list:
    """
    Load and normalize dataset(s) to the unified schema.
    name: "finqa" | "multihop" | "both"
    Returns list of normalized dicts.
    """
    if name == "both":
        finqa = _load_single("finqa", split, max_samples)
        multihop = _load_single("multihop", split, max_samples)
        return finqa + multihop
    return _load_single(name, split, max_samples)


def _load_single(name: str, split: str, max_samples: int) -> list:
    from config import MULTIHOP_DATASET

    if name == "finqa":
        raw = _load_finqa_raw(split)
        if max_samples is not None:
            raw = raw[:max_samples]
        return [adapt_finqa(ex) for ex in raw]

    elif name == "multihop":
        try:
            ds = hf_load_dataset(MULTIHOP_DATASET, "MultiHopRAG", split=split)
        except Exception:
            ds = hf_load_dataset(MULTIHOP_DATASET, "MultiHopRAG", split="train")
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))
        return [adapt_multihop(ex) for ex in ds]

    else:
        raise ValueError(f"Unknown dataset name: {name!r}")


def build_corpus(samples: list, corpus_size: int = None) -> list:
    """
    Flatten all supporting_docs from samples, deduplicate by doc_id.
    Returns a list of {"doc_id", "title", "text"} dicts.
    """
    seen = set()
    corpus = []
    for sample in samples:
        for doc in sample.get("supporting_docs", []):
            if doc["doc_id"] not in seen:
                seen.add(doc["doc_id"])
                corpus.append(doc)
    if corpus_size is not None:
        corpus = corpus[:corpus_size]
    return corpus
