import hashlib


def adapt_multihop(example: dict) -> dict:
    passages = example.get("supporting_passages", [])
    query = example.get("query", "")
    qid = example.get("query_id") or hashlib.md5(query.encode("utf-8")).hexdigest()[:16]

    docs = [
        {
            "doc_id": f"{qid}_p{i}",
            "title": p.get("title", ""),
            "text": p.get("body", ""),
        }
        for i, p in enumerate(passages)
    ]

    return {
        "id": f"mh_{qid}",
        "question": query,
        "answer": str(example.get("answer", "")),
        "answer_type": example.get("question_type", "entity"),
        "supporting_docs": docs,
        "dataset": "multihop",
    }
