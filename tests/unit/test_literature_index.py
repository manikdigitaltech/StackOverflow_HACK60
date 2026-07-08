"""Behavior contracts for core.rag.indexes.literature_index.LiteratureIndex.

Targets Phase 4. Builds a small fixture index in tmp_path from a handful of
CorpusRecords (with fake embeddings) instead of the full PeerRead corpus.
"""
import faiss
import pytest

from core.rag.indexes.literature_index import LiteratureIndex
from core.rag.models import CorpusRecord

RECORDS = [
    CorpusRecord(paper_id="iclr_2017:1", title="contrastive pretraining for time series",
                 abstract="we apply contrastive pretraining to time series anomaly detection"),
    CorpusRecord(paper_id="iclr_2017:2", title="recurrent translation models",
                 abstract="sequence to sequence translation with recurrent networks"),
    CorpusRecord(paper_id="iclr_2017:3", title="image classification at scale",
                 abstract="large scale convolutional image classification study"),
]


@pytest.fixture
def fixture_index(tmp_path, fake_provider):
    vectors = fake_provider.embed([f"{r.title}\n{r.abstract}" for r in RECORDS])
    index = faiss.IndexFlatIP(fake_provider.dimension)
    index.add(vectors)
    index_path = tmp_path / "index.faiss"
    records_path = tmp_path / "records.jsonl"
    faiss.write_index(index, str(index_path))
    records_path.write_text("\n".join(r.model_dump_json() for r in RECORDS), encoding="utf-8")
    return LiteratureIndex.load(index_path, records_path, embedding_provider=fake_provider)


def test_load_reads_index_and_records_in_matching_order(fixture_index):
    """After LiteratureIndex.load(), the Nth FAISS row must correspond to the Nth records.jsonl line."""
    # querying with record N's own text must surface record N first - only
    # true if FAISS rows and records stayed in lockstep
    for r in RECORDS:
        hits = fixture_index.search_literature(f"{r.title}\n{r.abstract}", k=1)
        assert hits[0].metadata["paper_id"] == r.paper_id


def test_literature_index_excludes_paper_under_review(fixture_index):
    """search_literature(..., exclude_paper_id=X) must never return a hit whose paper_id == X,
    even if that paper is the closest match in the index."""
    query = "contrastive pretraining time series anomaly detection"
    unguarded = fixture_index.search_literature(query, k=3)
    assert unguarded[0].metadata["paper_id"] == "iclr_2017:1"  # it IS the closest match
    guarded = fixture_index.search_literature(query, k=3, exclude_paper_id="iclr_2017:1")
    assert all(h.metadata["paper_id"] != "iclr_2017:1" for h in guarded)


def test_search_literature_returns_literature_rag_source_tag(fixture_index):
    """Every RetrievalResult from search_literature has source == 'literature_rag'."""
    hits = fixture_index.search_literature("translation", k=3)
    assert hits
    assert all(h.source == "literature_rag" for h in hits)


def test_search_literature_respects_k(fixture_index):
    """search_literature never returns more than k results even if more are available."""
    assert len(fixture_index.search_literature("study", k=2)) <= 2
    assert len(fixture_index.search_literature("study", k=1)) == 1
