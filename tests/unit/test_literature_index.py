"""Behavior contracts for core.rag.indexes.literature_index.LiteratureIndex.

Targets Phase 4. Requires a small prebuilt fixture index rather than the
full PeerRead corpus - build one from a handful of CorpusRecords in a
fixture/conftest once implementation starts.
"""


def test_load_reads_index_and_records_in_matching_order():
    """After LiteratureIndex.load(), the Nth FAISS row must correspond to the Nth records.jsonl line."""
    raise NotImplementedError


def test_literature_index_excludes_paper_under_review():
    """search_literature(..., exclude_paper_id=X) must never return a hit whose paper_id == X,
    even if that paper is the closest match in the index."""
    raise NotImplementedError


def test_search_literature_returns_literature_rag_source_tag():
    """Every RetrievalResult from search_literature has source == 'literature_rag'."""
    raise NotImplementedError

def test_search_literature_respects_k():
    """search_literature never returns more than k results even if more are available."""
    raise NotImplementedError
