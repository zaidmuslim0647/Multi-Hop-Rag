def adapt_multihop(example: dict) -> dict:
    passages = example.get("supporting_passages", [])
    docs = [
        {
            "doc_id": f"{example.get('query_id', str(i))}_p{i}",
            "title": p.get("title", ""),
            "text": p.get("body", ""),
        }
        for i, p in enumerate(passages)
    ]

    return {
        "id": str(example.get("query_id", "")),
        "question": example.get("query", ""),
        "answer": str(example.get("answer", "")),
        "answer_type": example.get("question_type", "entity"),
        "supporting_docs": docs,
        "dataset": "multihop",
    }
