"""Behavior contract: NoveltyAgent and CitationAgent both label live-source
literature matches distinctly from curated PeerRead-corpus matches in their
formatted prompt text, so the LLM (and a human reading the trace) can weight
a live web hit differently from a curated one."""

from core.agents.citation_agent import _format_literature_context as citation_format
from core.agents.novelty_agent import _format_literature_context as novelty_format
from core.schemas.agent_output_schemas import LiteratureContext, LiteratureMatch


def _match(source: str) -> LiteratureMatch:
    return LiteratureMatch(
        paper_id="p1", title="Some Paper", year=2023,
        chunk_text="chunk", similarity_score=0.5, source=source,
    )


def test_citation_formatter_labels_live_sources():
    context = LiteratureContext(query_text="q", matches=[
        _match("literature_index"), _match("arxiv"), _match("semantic_scholar"),
    ])
    formatted = citation_format(context)
    lines = formatted.splitlines()
    assert "[via arXiv]" not in lines[0] and "[via Semantic Scholar]" not in lines[0]
    assert "[via arXiv]" in lines[1]
    assert "[via Semantic Scholar]" in lines[2]


def test_novelty_formatter_labels_live_sources():
    context = LiteratureContext(query_text="q", matches=[_match("arxiv")])
    formatted = novelty_format(context)
    assert "[via arXiv]" in formatted


def test_formatters_handle_no_matches():
    context = LiteratureContext(query_text="q", matches=[])
    assert citation_format(context) == "No related literature was retrieved."
    assert novelty_format(context) == "No related literature was retrieved."
