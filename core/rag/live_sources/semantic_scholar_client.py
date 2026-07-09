"""Thin client for the Semantic Scholar Graph API - a live enhancement, not a guarantee.

Every call must be resilient: on timeout, HTTP error, or malformed response,
log a warning and return an empty list rather than raising. The offline
FAISS indexes carry the demo; this is best-effort supplementary coverage.
"""
from __future__ import annotations

import logging

from core.config.rag_settings import RAG_SETTINGS
from core.config.settings import settings
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
    import requests

    api_key = settings.live_sources.semantic_scholar_api_key
    headers = {"x-api-key": api_key} if api_key else {}

    try:
        response = requests.get(
            f"{RAG_SETTINGS.live_sources.semantic_scholar_base_url}/paper/search",
            params={"query": query, "limit": k, "fields": "title,abstract,year,venue,externalIds"},
            headers=headers,
            timeout=RAG_SETTINGS.live_sources.request_timeout_seconds,
        )
        response.raise_for_status()
        hits = (response.json().get("data") or [])[:k]
    except (requests.RequestException, ValueError) as exc:
        logger.warning("semantic scholar search failed, degrading to []: %s", exc)
        return []

    results: list[RetrievalResult] = []
    for rank, hit in enumerate(hits):
        title = (hit.get("title") or "").strip()
        abstract = (hit.get("abstract") or "").strip()
        if not title:
            continue
        results.append(RetrievalResult(
            source="semantic_scholar",
            # the search endpoint returns relevance order but no score; encode rank
            score=1.0 / (rank + 1),
            content=f"{title}\n{abstract}" if abstract else title,
            metadata={"title": title, "year": hit.get("year"), "venue": hit.get("venue"),
                      "external_ids": hit.get("externalIds") or {}},
        ))
    return results
