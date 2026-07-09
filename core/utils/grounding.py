"""
Shared helpers for verifying that an LLM's cited references actually match
retrieved literature. Used by the Novelty Agent's test now; the Citation
Agent (Step 7) will need identical logic later.

Deliberately NOT exact string equality: even a correctly-grounded LLM
response often reformats slightly (e.g. appending "(2024)" to a title that
had no year in the source data). Flagging that as a "hallucination" is a
false alarm from an overly strict check, not a real grounding failure.
"""

import re
from typing import Iterable

_TRAILING_YEAR_PATTERN = re.compile(r"\s*\(\d{4}\)\s*$")


def normalize_title(title: str) -> str:
    stripped = _TRAILING_YEAR_PATTERN.sub("", title or "")
    return stripped.strip().lower()


def is_title_grounded(cited_title: str, retrieved_titles: Iterable[str]) -> bool:
    normalized_cited = normalize_title(cited_title)
    return any(normalize_title(t) == normalized_cited for t in retrieved_titles)
