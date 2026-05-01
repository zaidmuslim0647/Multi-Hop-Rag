from multihop.decomposer import decompose
from multihop.hop_runner import run_hops
from multihop.aggregator import aggregate_and_answer
from pipeline.baseline import run_baseline


def run_multihop(question: str, index, chunks: list, embed_model, generator) -> dict:
    sub_queries = decompose(question, generator)

    if len(sub_queries) <= 1:
        answer = run_baseline(question, index, chunks, embed_model, generator)
        return {"answer": answer, "hop_count": 1, "fallback": True}

    hop_results = run_hops(sub_queries, index, chunks, embed_model, generator)
    answer = aggregate_and_answer(question, hop_results, generator)
    return {
        "answer": answer,
        "hop_count": hop_results["hop_count"],
        "fallback": False,
    }
