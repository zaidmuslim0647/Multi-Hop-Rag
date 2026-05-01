FINAL_ANSWER_PROMPT = """You are a financial analyst. Answer the original question using the chain of evidence gathered across multiple reasoning steps.

Original question: {question}

Evidence gathered:
{evidence_chain}

Provide a precise, grounded answer. If the answer is numeric, give only the number. If it is an entity, give only the entity name.

Answer:"""


def aggregate_and_answer(question: str, hop_results: dict, generator) -> str:
    evidence_lines = []
    for hop_id, answer in hop_results["intermediate_answers"].items():
        sources = [c["title"] for c in hop_results["retrieved_chunks"].get(hop_id, [])]
        source_str = ", ".join(sources) if sources else "unknown"
        evidence_lines.append(f"Step {hop_id}: {answer} (sources: {source_str})")
    evidence_chain = "\n".join(evidence_lines)
    return generator.generate(
        FINAL_ANSWER_PROMPT.format(question=question, evidence_chain=evidence_chain)
    )
