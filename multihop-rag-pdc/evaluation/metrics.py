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


def _load_checkpoint(log_path: str) -> tuple[set, list]:
    """Returns (completed_ids, existing_records) from a JSONL checkpoint file.
    Dedupes by id — keeps the LAST record per id (so a re-run that overwrote
    a previous attempt wins). Records without an id are dropped."""
    by_id = {}
    if log_path and os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    rid = r.get("id", "")
                    if not rid:
                        continue
                    by_id[rid] = r
                except Exception:
                    pass
    return set(by_id.keys()), list(by_id.values())


def evaluate_dataset(questions: list, run_fn, log_path: str = None) -> dict:
    """
    Evaluates questions with full checkpoint/resume support.
    - Reads log_path on start, skips already-completed question IDs.
    - Writes each result immediately after completion.
    - Re-raises RuntimeError (daily quota exhausted) so the notebook stops cleanly.
    """
    # --- checkpoint resume ---
    completed_ids, prior_records = _load_checkpoint(log_path)
    if completed_ids:
        print(f"[checkpoint] resuming — {len(completed_ids)} done, "
              f"{len(questions) - len(completed_ids)} remaining")

    # seed running scores from prior records
    em_scores = [r["em"] for r in prior_records]
    f1_scores = [r["f1"] for r in prior_records]
    latencies_ms = [r["latency_ms"] for r in prior_records]
    errors = 0
    total = len(questions)
    done = len(completed_ids)

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    for sample in questions:
        if sample.get("id", "") in completed_ids:
            continue

        done += 1
        q = sample["question"]
        gold = sample["answer"]

        t0 = time.perf_counter()
        pred = ""
        try:
            pred = run_fn(q)
        except RuntimeError:
            # hard stop: daily quota exhausted — propagate so the notebook cell fails visibly
            raise
        except Exception as e:
            errors += 1
            print(f"[WARN {done}/{total}] {type(e).__name__}: {e}")

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

        if done % 10 == 0:
            print(f"  [{done}/{total}] em={sum(em_scores)/len(em_scores):.3f}  "
                  f"f1={sum(f1_scores)/len(f1_scores):.3f}  errors={errors}")

    return {
        "em": sum(em_scores) / len(em_scores) if em_scores else 0.0,
        "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "avg_latency_ms": sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0,
        "n": total,
        "errors": errors,
    }
