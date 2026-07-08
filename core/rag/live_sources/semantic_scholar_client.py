"""Thin client for the Semantic Scholar Graph API — a live enhancement, not a guarantee.

Every call must be resilient: on timeout, HTTP error, or malformed response,
log a warning and return an empty list rather than raising. The offline
FAISS indexes carry the demo; this is best-effort supplementary coverage.
"""
from __future__ import annotations

import logging

from core.config.rag_settings import RAG_SETTINGS
from core.rag.models import RetrievalResult

logger = logging.getLogger(__name__)


def search_semantic_scholar(
    query: str, k: int = RAG_SETTINGS.live_sources.default_top_k
) -> list[RetrievalResult]:
    """Query `GET /graph/v1/paper/search` for papers matching `query`.

    Args:
        query: free-text search query (e.g. a paper title or topic phrase).
        k: max number of results to return.

    Returns:
        Up to `k` `RetrievalResult`s with source="semantic_scholar". Returns
        `[]` on any network error, timeout, or non-2xx response - callers
        must not assume this list reflects true zero-result search.
    """
    # TODO(Phase 5): requests.get(f"{base_url}/paper/search", params={...},
    #   timeout=RAG_SETTINGS.live_sources.request_timeout_seconds), wrapped
    #   in try/except (requests.RequestException, ValueError) -> log
    #   logger.warning(...) and return []. Map each hit's title/abstract/year
    #   into RetrievalResult(source="semantic_scholar").
    raise NotImplementedError
