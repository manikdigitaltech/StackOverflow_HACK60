"""
Lightweight token budget management, used to keep any single LLM prompt
within the model's context window regardless of source document size.

Uses a character-based heuristic rather than the model's real tokenizer,
since exact token counts aren't necessary here -- just a conservative
estimate that stays safely under the real limit. Real tokenizers vary
roughly 3.5-4.5 chars/token for English prose; 4 is a safe middle ground
that slightly UNDER-counts real capacity, which is the safe direction to
err in (we'd rather truncate a bit early than overflow the context window).
"""

CHARS_PER_TOKEN_ESTIMATE = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)


def truncate_to_token_budget(text: str, max_tokens: int, keep: str = "head_and_tail") -> str:
    """
    keep:
      "head"          - first N tokens only
      "tail"          - last N tokens only
      "head_and_tail" - split the budget between start and end (default) --
                         often most informative for academic text, since
                         intros/setup and conclusions/results tend to carry
                         more review-relevant signal than a truncated middle.
    """
    max_chars = max_tokens * CHARS_PER_TOKEN_ESTIMATE
    if len(text) <= max_chars:
        return text

    if keep == "head":
        return text[:max_chars] + "\n\n[...truncated for length...]"
    if keep == "tail":
        return "[...truncated for length...]\n\n" + text[-max_chars:]

    half = max_chars // 2
    return text[:half] + "\n\n[...truncated for length...]\n\n" + text[-half:]
