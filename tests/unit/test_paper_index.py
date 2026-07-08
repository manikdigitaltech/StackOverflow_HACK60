"""Behavior contracts for core.rag.indexes.paper_index.PaperIndex.

Dense-only tests target Phase 2; hybrid/section-filter tests target Phase 3.
"""


def test_dense_search_returns_top_k_by_cosine_similarity():
    """search_dense returns at most k results, ordered by descending cosine score."""
    raise NotImplementedError


def test_sparse_search_favors_exact_term_matches():
    """A query containing a rare exact term (e.g. a metric name) should rank a chunk
    containing that literal term above a chunk that is only semantically related."""
    raise NotImplementedError


def test_retrieve_applies_section_filter():
    """retrieve(..., section_filter='method') must not return chunks from other sections."""
    raise NotImplementedError


def test_retrieve_returns_paper_rag_source_tag():
    """Every RetrievalResult from PaperIndex.retrieve has source == 'paper_rag'."""
    raise NotImplementedError
