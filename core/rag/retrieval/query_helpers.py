"""Query-construction helpers that sit in front of retrieval, not inside it.

These transform an agent's raw question into better-targeted queries before
handing off to `core.rag.retrieval.tools`. Neither function touches an index
directly - they are pure query-shaping logic, which is what makes them
testable without a live index or LLM in Phase 6.

The LLM is dependency-injected as a plain `Callable[[str], str]` (prompt in,
completion out) so this module never imports an LLM client and stays
testable with a fake. With `llm=None` both functions degrade to deterministic
heuristics - reduced quality, never a hard dependency.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

LLMFn = Optional[Callable[[str], str]]

_DECOMPOSE_PROMPT = """Split the following question into 2-4 focused, independently
searchable sub-queries, one per line. Output ONLY the sub-queries, no numbering.
If the question is already a single focused query, output it unchanged.

Question: {question}"""

_HYDE_PROMPT = """Write a short (3-5 sentence) hypothetical paper abstract that would
perfectly answer the following question. Plausible academic style; invented
specifics are fine - this text is used only as a search probe, never shown to
anyone. Output ONLY the abstract.

Question: {question}"""


def decompose_query(question: str, llm: LLMFn = None) -> list[str]:
    """Expand one agent question into several targeted retrieval queries.

    Example: a citation agent asking "does this paper's related-work section
    adequately cover prior work on X?" benefits from being split into
    "prior work on X" and "related work section coverage" as two separate
    retrievals, since a single embedding of the compound question tends to
    retrieve mediocre matches for both halves rather than a good match for
    either.

    Args:
        question: the agent's original natural-language question.
        llm: injected completion function (prompt -> text). None = heuristic
            split on coordinating conjunctions only.

    Returns:
        List of 1+ sub-queries to retrieve for independently; callers should
        merge/dedupe the resulting hits themselves.
    """
    question = question.strip()
    if not question:
        return []
    if llm is not None:
        try:
            lines = [ln.strip(" -*\t") for ln in llm(_DECOMPOSE_PROMPT.format(question=question)).splitlines()]
            subs = [ln for ln in lines if ln]
            if subs:
                return subs[:4]
        except Exception:
            pass  # fall through to the heuristic - query shaping must not break retrieval
    # heuristic: split compound questions on top-level conjunctions; a simple
    # focused question passes through as itself
    parts = [p.strip(" ,;?") for p in re.split(r"\band\b|;", question) if p.strip(" ,;?")]
    return parts if len(parts) > 1 else [question]


def hyde_query(question: str, llm: LLMFn = None) -> str:
    """HyDE: generate a hypothetical ideal-answer document, to embed instead
    of the raw question, for literature (Index B) search.

    Optional / Phase 2 in priority - not required for the paper-reviewer demo
    to function, but improves literature-search recall because a fabricated
    "ideal abstract" embeds closer to real abstracts than a short question
    does.

    Args:
        question: e.g. "has anyone applied contrastive pretraining to
            time-series anomaly detection before?"
        llm: injected completion function (prompt -> text). None = a
            deterministic abstract-shaped template around the question.

    Returns:
        A short hypothetical abstract-like passage answering `question`,
        intended to be passed to `search_literature` in place of the raw
        question text.
    """
    question = question.strip()
    if llm is not None:
        try:
            passage = llm(_HYDE_PROMPT.format(question=question)).strip()
            if passage:
                return passage
        except Exception:
            pass  # fall through to the template
    topic = question.rstrip("?").strip()
    return (f"We study {topic}. We propose a method addressing this problem and "
            f"evaluate it against standard baselines, reporting quantitative results "
            f"and an ablation analysis. Our findings characterize when the approach "
            f"succeeds and its limitations.")
