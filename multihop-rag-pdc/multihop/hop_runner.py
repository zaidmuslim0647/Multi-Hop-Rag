from pipeline.retriever import retrieve
from config import TOP_K

HOP_ANSWER_PROMPT = """Answer the following question using only the provided context. Reply in one short phrase — a number, name, or single sentence. Do not explain.

Context:
{context}

Question: {question}

Answer:"""


def build_context_string(chunks: list) -> str:
    return "\n\n".join(f"[{c['title']}]: {c['text']}" for c in chunks)


def run_hops(sub_queries: list, index, chunks: list, embed_model, generator) -> dict:
    """
    Executes sub-queries in dependency order.
    Returns dict with intermediate_answers, retrieved_chunks, hop_count.
    """
    answers = {}
    retrieved_map = {}

    for sq in sub_queries:
        query_text = sq["question"]

        # substitute [ANSWER_N] placeholders directly — no separate conditioner LLM call
        if sq["depends_on"] is not None:
            parent_id = sq["depends_on"]
            parent_answer = answers.get(parent_id, "")
            query_text = query_text.replace(f"[ANSWER_{parent_id}]", parent_answer)

        retrieved_chunks = retrieve(query_text, index, chunks, embed_model, TOP_K)
        context = build_context_string(retrieved_chunks)
        answer = generator.generate(
            HOP_ANSWER_PROMPT.format(context=context, question=query_text),
            max_tokens=64,
        )

        answers[sq["id"]] = answer.strip()
        retrieved_map[sq["id"]] = retrieved_chunks

    return {
        "intermediate_answers": answers,
        "retrieved_chunks": retrieved_map,
        "hop_count": len(sub_queries),
    }
