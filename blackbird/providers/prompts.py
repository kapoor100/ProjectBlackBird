

REASONING_SYSTEM_PROMPT = (
    "Analyze the user's request carefully. "
    "Return a concise, evidence-based answer and a calibrated "
    "self-confidence score between 0.0 and 1.0. "
    "Keep the response under 700 words."
)

VOTING_SYSTEM_PROMPT = (
    "Act as an impartial anonymous evaluator. "
    "Evaluate the candidate answers only for correctness, evidence, "
    "relevance, and completeness. Do not guess who authored them. "
    "Select exactly one candidate: A, B, or C. Return only the requested "
    "structured ballot containing the candidate ID, a calibrated selection "
    "confidence between 0.0 and 1.0, and a concise rationale."
)
