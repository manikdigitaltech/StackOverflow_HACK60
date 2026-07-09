"""Behavior contracts for LiteratureRAGAgent's optional live-source merge.

Live sources (arXiv, Semantic Scholar) are off by default -- these tests
pin that default, verify the merge/dedup/cap behavior when enabled, and
verify a live-source outage (empty list) never breaks the run.
"""
from unittest.mock import MagicMock

import core.agents.literature_rag_agent as literature_rag_agent
from core.agents.literature_rag_agent import LiteratureRAGAgent
from core.config.rag_settings import LiveSourceSettings, RAG_SETTINGS, RagSettings
from core.rag.models import RetrievalResult
from core.schemas.agent_output_schemas import ParsedPaper

PAPER = ParsedPaper(
    title="Contrastive Pretraining for Time Series",
    abstract="We apply contrastive pretraining to time series anomaly detection.",
    sections=[], tables=[], figures=[], references=[],
    source_pdf_path="fake.pdf",
)


def _index_result(paper_id="iclr_2017:1", title="Existing Corpus Paper", score=0.9):
    r = MagicMock()
    r.metadata = {"paper_id": paper_id, "title": title, "year": 2017}
    r.content = "chunk text"
    r.score = score
    return r


def _make_agent(index_results=None, top_k=10):
    index = MagicMock()
    index.search_literature.return_value = index_results or []
    return LiteratureRAGAgent(index=index, top_k=top_k)


def test_live_sources_enabled_by_default():
    """Live enrichment is on by default (problem statement §9 encourages it);
    safe because both clients degrade to [] on any failure."""
    assert RAG_SETTINGS.live_sources.enable_arxiv is True
    assert RAG_SETTINGS.live_sources.enable_semantic_scholar is True


def test_live_sources_not_called_when_disabled(monkeypatch):
    monkeypatch.setattr(
        literature_rag_agent, "RAG_SETTINGS",
        RagSettings(live_sources=LiveSourceSettings(enable_arxiv=False, enable_semantic_scholar=False)),
    )
    fake_arxiv = MagicMock(return_value=[])
    fake_ss = MagicMock(return_value=[])
    monkeypatch.setattr(literature_rag_agent, "search_arxiv", fake_arxiv)
    monkeypatch.setattr(literature_rag_agent, "search_semantic_scholar", fake_ss)

    agent = _make_agent(index_results=[_index_result()])
    context = agent.run({"parsed_paper": PAPER})

    fake_arxiv.assert_not_called()
    fake_ss.assert_not_called()
    assert len(context.matches) == 1
    assert context.matches[0].source == "literature_index"


def test_live_sources_merge_when_enabled(monkeypatch):
    monkeypatch.setattr(
        literature_rag_agent, "RAG_SETTINGS",
        RagSettings(live_sources=LiveSourceSettings(enable_arxiv=True, enable_semantic_scholar=True)),
    )
    fake_arxiv = MagicMock(return_value=[
        RetrievalResult(source="arxiv", score=0.5, content="arxiv chunk",
                         metadata={"title": "Live ArXiv Paper", "url": "http://arxiv.org/abs/1"}),
    ])
    fake_ss = MagicMock(return_value=[
        RetrievalResult(source="semantic_scholar", score=0.6, content="ss chunk",
                         metadata={"title": "Live SS Paper", "year": 2023}),
    ])
    monkeypatch.setattr(literature_rag_agent, "search_arxiv", fake_arxiv)
    monkeypatch.setattr(literature_rag_agent, "search_semantic_scholar", fake_ss)

    agent = _make_agent(index_results=[_index_result()], top_k=10)
    context = agent.run({"parsed_paper": PAPER})

    sources = {m.source for m in context.matches}
    assert sources == {"literature_index", "arxiv", "semantic_scholar"}
    assert len(context.matches) == 3


def test_live_sources_dedupe_by_title(monkeypatch):
    monkeypatch.setattr(
        literature_rag_agent, "RAG_SETTINGS",
        RagSettings(live_sources=LiveSourceSettings(enable_arxiv=True, enable_semantic_scholar=False)),
    )
    fake_arxiv = MagicMock(return_value=[
        RetrievalResult(source="arxiv", score=0.5, content="dup chunk",
                         metadata={"title": "Existing Corpus Paper", "url": "http://arxiv.org/abs/1"}),
    ])
    monkeypatch.setattr(literature_rag_agent, "search_arxiv", fake_arxiv)
    monkeypatch.setattr(literature_rag_agent, "search_semantic_scholar", MagicMock(return_value=[]))

    agent = _make_agent(index_results=[_index_result(title="Existing Corpus Paper")], top_k=10)
    context = agent.run({"parsed_paper": PAPER})

    assert len(context.matches) == 1
    assert context.matches[0].source == "literature_index"


def test_live_sources_capped_at_top_k(monkeypatch):
    monkeypatch.setattr(
        literature_rag_agent, "RAG_SETTINGS",
        RagSettings(live_sources=LiveSourceSettings(enable_arxiv=True, enable_semantic_scholar=False)),
    )
    fake_arxiv = MagicMock(return_value=[
        RetrievalResult(source="arxiv", score=0.5, content="c", metadata={"title": f"Arxiv Paper {i}"})
        for i in range(5)
    ])
    monkeypatch.setattr(literature_rag_agent, "search_arxiv", fake_arxiv)
    monkeypatch.setattr(literature_rag_agent, "search_semantic_scholar", MagicMock(return_value=[]))

    agent = _make_agent(index_results=[_index_result()], top_k=2)
    context = agent.run({"parsed_paper": PAPER})

    assert len(context.matches) == 2


def test_live_source_outage_does_not_break_run(monkeypatch):
    monkeypatch.setattr(
        literature_rag_agent, "RAG_SETTINGS",
        RagSettings(live_sources=LiveSourceSettings(enable_arxiv=True, enable_semantic_scholar=True)),
    )
    monkeypatch.setattr(literature_rag_agent, "search_arxiv", MagicMock(return_value=[]))
    monkeypatch.setattr(literature_rag_agent, "search_semantic_scholar", MagicMock(return_value=[]))

    agent = _make_agent(index_results=[_index_result()], top_k=10)
    context = agent.run({"parsed_paper": PAPER})

    assert len(context.matches) == 1
    assert context.matches[0].source == "literature_index"


def test_no_index_and_live_outage_returns_empty_context(monkeypatch):
    monkeypatch.setattr(literature_rag_agent, "search_arxiv", MagicMock(return_value=[]))
    monkeypatch.setattr(literature_rag_agent, "search_semantic_scholar", MagicMock(return_value=[]))

    agent = LiteratureRAGAgent(index=None, top_k=10)
    agent._load_attempted = True  # skip real disk lookup, simulate "no corpus built"
    context = agent.run({"parsed_paper": PAPER})

    assert context.matches == []
