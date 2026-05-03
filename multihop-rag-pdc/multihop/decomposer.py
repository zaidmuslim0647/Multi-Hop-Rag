import json
import re
from config import MAX_HOPS

DECOMPOSE_PROMPT = """You are a financial reasoning assistant. Break down the complex question into simple, ordered sub-questions.

Rules:
- Each sub-question must be answerable from a single piece of evidence
- If sub-question B needs the answer from sub-question A, mark it as depends_on: 1
- Output ONLY a JSON array, no explanation, no markdown fences

Example:
Question: "What was the goodwill adjustment for the company that acquired Hittite Microwave in 2014?"
Output:
[
  {{"id": 1, "question": "Which company acquired Hittite Microwave in 2014?", "depends_on": null}},
  {{"id": 2, "question": "What was the goodwill adjustment for [ANSWER_1] in 2014?", "depends_on": 1}}
]

Question: {question}
Output:"""

DECOMPOSE_STRICT_PROMPT = """Output ONLY a valid JSON array of sub-questions. No markdown, no explanation.
Each element: {{"id": <int>, "question": "<str>", "depends_on": <int or null>}}

Question: {question}
Output:"""


def _parse_json(raw: str) -> list:
    raw = raw.strip()
    # strip markdown code fences if present
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


def _validate(sub_queries: list) -> list:
    validated = []
    for sq in sub_queries:
        if not isinstance(sq, dict):
            continue
        if "id" not in sq or "question" not in sq:
            continue
        validated.append(
            {
                "id": int(sq["id"]),
                "question": str(sq["question"]),
                "depends_on": int(sq["depends_on"]) if sq.get("depends_on") is not None else None,
            }
        )
    # sort by id and cap at MAX_HOPS
    validated.sort(key=lambda x: x["id"])
    if len(validated) > MAX_HOPS:
        print(f"[WARN] Decomposer returned {len(validated)} hops — capping at {MAX_HOPS}")
        validated = validated[:MAX_HOPS]
    return validated


def decompose(question: str, generator) -> list:
    """
    Returns list of {"id", "question", "depends_on"} dicts in dependency order.
    Falls back to single-hop on parse failure.
    """
    prompt = DECOMPOSE_PROMPT.format(question=question)
    raw = generator.generate(prompt, max_tokens=400)
    try:
        sub_queries = _validate(_parse_json(raw))
        if sub_queries:
            return sub_queries
        raise ValueError("empty list")
    except Exception as e:
        print(f"[WARN] Decompose parse failed ({e}), retrying with strict prompt")

    # retry with stricter prompt
    try:
        raw2 = generator.generate(DECOMPOSE_STRICT_PROMPT.format(question=question), max_tokens=400)
        sub_queries = _validate(_parse_json(raw2))
        if sub_queries:
            return sub_queries
    except Exception as e2:
        print(f"[WARN] Strict retry also failed ({e2}), falling back to single-hop")

    return [{"id": 1, "question": question, "depends_on": None}]
