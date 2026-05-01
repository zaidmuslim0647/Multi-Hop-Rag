def linearize_table(table: list) -> str:
    if not table:
        return ""
    rows = []
    for i, row in enumerate(table):
        rows.append(f"Row {i}: " + " | ".join(str(cell) for cell in row))
    return "\n".join(rows)


def adapt_finqa(example: dict) -> dict:
    table_str = linearize_table(example.get("table", []))
    pre = " ".join(example.get("pre_text", []))
    post = " ".join(example.get("post_text", []))
    doc_text = "\n".join(filter(None, [pre, table_str, post]))

    qa = example.get("qa", {})
    return {
        "id": str(example.get("id", "")),
        "question": qa.get("question", ""),
        "answer": str(qa.get("exe_ans", "")),
        "answer_type": "numeric",
        "supporting_docs": [
            {
                "doc_id": str(example.get("id", "")),
                "title": example.get("filename", ""),
                "text": doc_text,
            }
        ],
        "dataset": "finqa",
    }
