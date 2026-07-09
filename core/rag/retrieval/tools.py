"""The four retrieval tools agents will call, wired to their backing indexes/sources.

This is the seam between the RAG subsystem and the (not-yet-built) agent
layer: agents should only ever import from here, never reach into
`indexes/` or `live_sources/` directly. That keeps the ReAct loop (Phase 7)
free to swap implementations behind these signatures without touching agent
code.
"""
from __future__ import annotations

from core.config.rag_settings import RAG_SETTINGS
from core.rag.indexes.literature_index import LiteratureIndex
from core.rag.indexes.paper_index import PaperIndex
from core.rag.live_sources.arxiv_client import search_arxiv as _search_arxiv
from core.rag.live_sources.semantic_scholar_client import (
    search_semantic_scholar as _search_semantic_scholar,
)
from core.rag.models import RetrievalResult


def retrieve_from_paper(
    index: PaperIndex,
    query: str,
    section_filter: str | None = None,
    k: int = RAG_SETTINGS.paper_index.default_top_k,
) -> list[RetrievalResult]:
    """Ground a claim in the submission paper's own text (Index A, hybrid).

    Args:
        index: the `PaperIndex` built for the paper currently under review
            (one per review run - see `graph/nodes` for where this gets
            constructed once the agent layer exists).
        query: the question to ground, e.g. "what evaluation metrics does
            the method section report?"
        section_filter: restrict to one section (e.g. "method"), or None to
            search the whole paper.
        k: number of chunks to return.

    Returns:
        Ranked list of `RetrievalResult` (source="paper_rag").
    """
    return index.retrieve(query, section_filter=section_filter, k=k)


def search_literature(
    index: LiteratureIndex,
    query: str,
    k: int = RAG_SETTINGS.literature_index.default_top_k,
    exclude_paper_id: str | None = None,
) -> list[RetrievalResult]:
    """Check whether a contribution has prior art in the persistent PeerRead corpus (Index B).

    Args:
        index: the process-wide `LiteratureIndex`, loaded once at startup.
        query: novelty question, e.g. the paper's contribution restated as
            a search query (optionally produced by `hyde_query` first).
        k: number of papers to return.
        exclude_paper_id: paper_id of the submission under review, to guard
            against self-match leakage (see README).

    Returns:
        Ranked list of `RetrievalResult` (source="literature_rag").
    """
    return index.search_literature(query, k=k, exclude_paper_id=exclude_paper_id)


def search_semantic_scholar(
    query: str, k: int = RAG_SETTINGS.live_sources.default_top_k
) -> list[RetrievalResult]:
    """Live supplementary novelty check via Semantic Scholar. Never blocks the demo.

    Args:
        query: free-text search query.
        k: number of results to return.

    Returns:
        Ranked list of `RetrievalResult` (source="semantic_scholar"), or `[]`
        if the live call fails for any reason.
    """
    return _search_semantic_scholar(query, k=k)


def search_arxiv(
    query: str, k: int = RAG_SETTINGS.live_sources.default_top_k
) -> list[RetrievalResult]:
    """Live supplementary novelty check via arXiv. Never blocks the demo.

    Args:
        query: free-text search query.
        k: number of results to return.

    Returns:
        Ranked list of `RetrievalResult` (source="arxiv"), or `[]` if the
        live call fails for any reason.
    """
    return _search_arxiv(query, k=k)
