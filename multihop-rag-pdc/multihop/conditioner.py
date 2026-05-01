CONDITION_PROMPT = """Given the intermediate finding, rewrite the follow-up question to be more specific and searchable.

Intermediate finding: {prior_answer}
Follow-up question: {next_question}

Rewritten question (one sentence, specific, searchable):"""


def condition_query(prior_answer: str, next_question: str, generator) -> str:
    prompt = CONDITION_PROMPT.format(prior_answer=prior_answer, next_question=next_question)
    try:
        result = generator.generate(prompt)
        return result.strip()
    except Exception as e:
        print(f"[WARN] Conditioner failed ({e}), using original question")
        return next_question
