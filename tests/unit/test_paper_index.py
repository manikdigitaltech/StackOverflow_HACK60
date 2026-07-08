"""Behavior contracts for core.rag.indexes.paper_index.PaperIndex.

Dense-only tests target Phase 2; hybrid/section-filter tests target Phase 3.
All tests inject the FakeEmbeddingProvider from conftest so real FAISS/BM25/
RRF logic runs without any model download.
"""
from core.rag.indexes.paper_index import PaperIndex
from core.rag.models import Chunk


def _chunk(i: int, section: str, text: str) -> Chunk:
    return Chunk(chunk_id=f"p:{section}:{i}", paper_id="p", section=section,
                 para_idx=i, text=text, token_count=len(text.split()))


CHUNKS = [
    _chunk(0, "method", "we train a transformer encoder with contrastive loss on pairs"),
    _chunk(1, "experiments", "evaluation uses the ROUGE metric on the summarization benchmark"),
    _chunk(2, "results", "our approach improves summary quality over the baseline system"),
    _chunk(3, "related_work", "prior systems studied translation with recurrent networks"),
]


def _built_index(fake_provider) -> PaperIndex:
    index = PaperIndex(embedding_provider=fake_provider)
    index.build(CHUNKS)
    return index


def test_dense_search_returns_top_k_by_cosine_similarity(fake_provider):
    """search_dense returns at most k results, ordered by descending cosine score."""
    index = _built_index(fake_provider)
    hits = index.search_dense("transformer encoder contrastive loss", k=3)
    assert 1 <= len(hits) <= 3
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
    assert hits[0][0] == 0  # the chunk sharing the most tokens with the query


def test_sparse_search_favors_exact_term_matches(fake_provider):
    """A query containing a rare exact term (e.g. a metric name) should rank a chunk
    containing that literal term above a chunk that is only semantically related."""
    index = _built_index(fake_provider)
    hits = index.search_sparse("ROUGE metric", k=4)
    assert hits[0][0] == 1  # literal "ROUGE" beats the semantically-related results chunk


def test_retrieve_applies_section_filter(fake_provider):
    """retrieve(..., section_filter='method') must not return chunks from other sections."""
    index = _built_index(fake_provider)
    results = index.retrieve("training procedure", section_filter="method", k=4)
    assert results, "filtered retrieve returned nothing"
    assert all(r.metadata["section"] == "method" for r in results)


def test_retrieve_returns_paper_rag_source_tag(fake_provider):
    """Every RetrievalResult from PaperIndex.retrieve has source == 'paper_rag'."""
    index = _built_index(fake_provider)
    results = index.retrieve("summary quality", k=4)
    assert results
    assert all(r.source == "paper_rag" for r in results)
