"""Thin client for the arXiv API (no auth) — a live enhancement, not a guarantee.

Same resilience contract as `semantic_scholar_client`: never raise out of
this module, always degrade to `[]` plus a logged warning.
"""
from __future__ import annotations

import logging

from core.config.rag_settings import RAG_SETTINGS
from core.rag.models import RetrievalResult

logger = logging.getLogger(__name__)


def search_arxiv(
    query: str, k: int = RAG_SETTINGS.live_sources.default_top_k
) -> list[RetrievalResult]:
    """Query arXiv's public search API for papers matching `query`.

    Args:
        query: free-text search query.
        k: max number of results to return.

    Returns:
        Up to `k` `RetrievalResult`s with source="arxiv". Returns `[]` on any
        network error, timeout, or parse failure.
    """
    # TODO(Phase 5): use the `arxiv` package's Search(query=..., max_results=k)
    #   wrapped in try/except with a timeout guard (the arxiv package doesn't
    #   expose one directly - consider running it under a thread/asyncio
    #   timeout, or fall back to raw `requests.get` against the Atom API with
    #   RAG_SETTINGS.live_sources.request_timeout_seconds). Log and return []
    #   on failure. Map each entry's title/summary/published into
    #   RetrievalResult(source="arxiv").
    raise NotImplementedError
