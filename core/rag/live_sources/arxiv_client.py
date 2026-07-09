"""Thin client for the arXiv API (no auth) - a live enhancement, not a guarantee.

Same resilience contract as `semantic_scholar_client`: never raise out of
this module, always degrade to `[]` plus a logged warning.
"""
from __future__ import annotations

import logging

from core.config.rag_settings import RAG_SETTINGS
from core.rag.models import RetrievalResult
from core.utils.guardrails import sanitize_pdf_text

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
    # raw Atom API via requests instead of the `arxiv` package: the package
    # exposes no timeout, and a hung live call must never stall a review run.
    import xml.etree.ElementTree as ET

    import requests

    try:
        response = requests.get(
            "https://export.arxiv.org/api/query",
            params={"search_query": f"all:{query}", "max_results": k, "start": 0},
            timeout=RAG_SETTINGS.live_sources.request_timeout_seconds,
        )
        response.raise_for_status()
        root = ET.fromstring(response.text)
    except (requests.RequestException, ET.ParseError, ValueError) as exc:
        logger.warning("arxiv search failed, degrading to []: %s", exc)
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    results: list[RetrievalResult] = []
    for rank, entry in enumerate(root.findall("atom:entry", ns)[:k]):
        # Live-source text bypasses the PDF parser (where sanitization normally
        # runs) but still lands in agent prompts -- sanitize it here.
        title, _ = sanitize_pdf_text(" ".join((entry.findtext("atom:title", "", ns) or "").split()))
        summary, _ = sanitize_pdf_text(" ".join((entry.findtext("atom:summary", "", ns) or "").split()))
        if not title:
            continue
        results.append(RetrievalResult(
            source="arxiv",
            score=1.0 / (rank + 1),  # Atom feed is relevance-ordered, no score field
            content=f"{title}\n{summary}" if summary else title,
            metadata={"title": title,
                      "published": entry.findtext("atom:published", "", ns),
                      "url": entry.findtext("atom:id", "", ns)},
        ))
    return results
