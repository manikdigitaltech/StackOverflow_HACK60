"""Query-construction helpers that sit in front of retrieval, not inside it.

These transform an agent's raw question into better-targeted queries before
handing off to `core.rag.retrieval.tools`. Neither function touches an index
directly - they are pure query-shaping logic, which is what makes them
testable without a live index or LLM in Phase 6.
"""
from __future__ import annotations


def decompose_query(question: str) -> list[str]:
    """Expand one agent question into several targeted retrieval queries.

    Example: a citation agent asking "does this paper's related-work section
    adequately cover prior work on X?" benefits from being split into
    "prior work on X" and "related work section coverage" as two separate
    retrievals, since a single embedding of the compound question tends to
    retrieve mediocre matches for both halves rather than a good match for
    either.

    Args:
        question: the agent's original natural-language question.

    Returns:
        List of 1+ sub-queries to retrieve for independently; callers should
        merge/dedupe the resulting hits themselves.
    """
    # TODO(Phase 6): use the shared LLM (lives outside this package - inject
    #   via a callback/provider param once the agent layer exists) to
    #   decompose `question` into 2-4 focused sub-queries. Keep this
    #   dependency-injected rather than importing an LLM client directly, so
    #   this module stays testable with a fake.
    raise NotImplementedError


def hyde_query(question: str) -> str:
    """HyDE: generate a hypothetical ideal-answer document, to embed instead
    of the raw question, for literature (Index B) search.

    Optional / Phase 2 in priority - not required for the paper-reviewer demo
    to function, but improves literature-search recall because a fabricated
    "ideal abstract" embeds closer to real abstracts than a short question
    does.

    Args:
        question: e.g. "has anyone applied contrastive pretraining to
            time-series anomaly detection before?"

    Returns:
        A short hypothetical abstract-like passage answering `question`,
        intended to be passed to `search_literature` in place of the raw
        question text.
    """
    # TODO(Phase 6, optional): call the shared LLM to draft a plausible
    #   abstract-length answer to `question`. Same DI note as decompose_query
    #   - do not hardcode an LLM client here.
    raise NotImplementedError
