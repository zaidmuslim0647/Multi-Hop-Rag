from pipeline.retriever import retrieve
from config import TOP_K


def build_context_string(chunks: list) -> str:
    return "\n\n".join(f"[{c['title']}]: {c['text']}" for c in chunks)


def run_baseline(question: str, index, chunks: list, embed_model, generator) -> str:
    retrieved = retrieve(question, index, chunks, embed_model, TOP_K)
    context = build_context_string(retrieved)
    prompt = f"""Answer the following financial question using only the provided context.

Context:
{context}

Question: {question}

Answer:"""
    return generator.generate(prompt)
