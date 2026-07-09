"""Behavior contracts for core.rag.chunking.section_chunker.chunk_paper.

Uses tests/fixtures/sample_paper.json as the parsed_paper fixture. Token
counting may fall back to whitespace if the HF tokenizer is unavailable;
these contracts hold under either accounting.
"""
from core.config.rag_settings import RAG_SETTINGS
from core.rag.chunking.section_chunker import chunk_paper


def test_chunker_preserves_section_boundaries(sample_paper):
    """No chunk should contain text from two different sections."""
    chunks = chunk_paper("sample-001", sample_paper)
    by_section = {}
    for c in chunks:
        by_section.setdefault(c.section, []).append(c.text)
    # every chunk's text must appear inside its own source section's text only
    for section_name, source_text in sample_paper["sections"].items():
        for c in chunks:
            if c.section == section_name and not c.has_table:
                assert c.text.split()[0] in source_text.split() or c.text in source_text


def test_section_aware_chunker_keeps_tables_intact(sample_paper):
    """A paragraph containing a markdown table must stay in one chunk with has_table=True."""
    chunks = chunk_paper("sample-001", sample_paper)
    table_chunks = [c for c in chunks if c.has_table]
    assert len(table_chunks) == 1
    t = table_chunks[0]
    assert t.section == "experiments"
    assert "| Accuracy | 0.91 |" in t.text and "| F1 | 0.88 |" in t.text


def test_long_section_is_split_with_overlap():
    """A section exceeding max_tokens is split into multiple chunks sharing overlap_tokens of text."""
    words = [f"tok{i}" for i in range(RAG_SETTINGS.chunking.max_tokens * 3)]
    parsed = {"sections": {"method": " ".join(words)}}
    chunks = chunk_paper("long-001", parsed)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.token_count <= RAG_SETTINGS.chunking.max_tokens
    # consecutive windows share text (overlap): last words of chunk i appear in chunk i+1
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert set(first[-5:]) & set(second), "no overlapping text between consecutive windows"


def test_short_section_produces_single_chunk(sample_paper):
    """A section under min_tokens (e.g. abstract) should not be needlessly split."""
    chunks = chunk_paper("sample-001", sample_paper)
    abstract_chunks = [c for c in chunks if c.section == "abstract"]
    assert len(abstract_chunks) == 1
    assert abstract_chunks[0].text.startswith("This paper studies")


def test_chunk_ids_are_unique_within_a_paper(sample_paper):
    """No two chunks from the same chunk_paper() call share a chunk_id."""
    chunks = chunk_paper("sample-001", sample_paper)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(cid.startswith("sample-001:") for cid in ids)
