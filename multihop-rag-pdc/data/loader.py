from datasets import load_dataset as hf_load_dataset
from data.finqa_adapter import adapt_finqa
from data.multihop_adapter import adapt_multihop


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
    from config import FINQA_DATASET, MULTIHOP_DATASET

    if name == "finqa":
        ds = hf_load_dataset(FINQA_DATASET, split=split, trust_remote_code=True)
        adapter = adapt_finqa
    elif name == "multihop":
        try:
            ds = hf_load_dataset(MULTIHOP_DATASET, split=split, trust_remote_code=True)
        except Exception:
            # fallback — dataset may only have "train"
            ds = hf_load_dataset(MULTIHOP_DATASET, split="train", trust_remote_code=True)
        adapter = adapt_multihop
    else:
        raise ValueError(f"Unknown dataset name: {name}")

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    return [adapter(ex) for ex in ds]


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
