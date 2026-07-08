"""Behavior contracts for core.rag.chunking.section_chunker.chunk_paper.

Fill these in once chunk_paper is implemented (Phase 1). Use
tests/fixtures/sample_paper.json as the parsed_paper fixture.
"""


def test_chunker_preserves_section_boundaries():
    """No chunk should contain text from two different sections."""
    raise NotImplementedError


def test_section_aware_chunker_keeps_tables_intact():
    """A paragraph containing a markdown table must stay in one chunk with has_table=True."""
    raise NotImplementedError


def test_long_section_is_split_with_overlap():
    """A section exceeding max_tokens is split into multiple chunks sharing overlap_tokens of text."""
    raise NotImplementedError


def test_short_section_produces_single_chunk():
    """A section under min_tokens (e.g. abstract) should not be needlessly split."""
    raise NotImplementedError


def test_chunk_ids_are_unique_within_a_paper():
    """No two chunks from the same chunk_paper() call share a chunk_id."""
    raise NotImplementedError
