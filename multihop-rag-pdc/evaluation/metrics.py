import re
import string
import time
import json
import os
from collections import Counter


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[" + re.escape(string.punctuation) + r"]", " ", text)
    return " ".join(text.split())


def exact_match(pred: str, gold: str) -> float:
    pred_n = _normalize(pred)
    gold_n = _normalize(gold)
    if pred_n == gold_n:
        return 1.0
    # numeric tolerance
    try:
        return 1.0 if abs(float(pred_n) - float(gold_n)) < 1e-4 else 0.0
    except ValueError:
        return 0.0


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = _normalize(pred).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_dataset(questions: list, run_fn, log_path: str = None) -> dict:
    """
    questions: list of normalized sample dicts with "question" and "answer"
    run_fn: callable(question_str) -> answer_str
    log_path: JSONL file to append results incrementally (never hold in memory only)
    """
    em_scores = []
    f1_scores = []
    latencies_ms = []

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    for sample in questions:
        q = sample["question"]
        gold = sample["answer"]

        t0 = time.perf_counter()
        try:
            pred = run_fn(q)
        except Exception as e:
            pred = ""
            print(f"[WARN] run_fn failed for '{q[:60]}': {e}")

        latency_ms = (time.perf_counter() - t0) * 1000
        em = exact_match(pred, gold)
        f1 = token_f1(pred, gold)

        em_scores.append(em)
        f1_scores.append(f1)
        latencies_ms.append(latency_ms)

        record = {
            "id": sample.get("id", ""),
            "question": q,
            "gold": gold,
            "pred": pred,
            "em": em,
            "f1": f1,
            "latency_ms": latency_ms,
        }
        if log_path:
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")

    return {
        "em": sum(em_scores) / len(em_scores) if em_scores else 0.0,
        "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "avg_latency_ms": sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0,
        "n": len(questions),
    }
