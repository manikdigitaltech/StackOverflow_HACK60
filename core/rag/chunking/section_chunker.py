"""Structure-aware chunker for Index A (Paper-RAG).

Two-pass design: first split the parsed paper on its own section boundaries
(abstract, intro, method, ...), then, within each section, split long spans
into ~300-500 token windows with overlap. Splitting on structure BEFORE size
is what keeps a chunk's `section` metadata meaningful — a naive fixed-window
split over the raw text would blend method text into results text at
boundaries and make `section_filter` retrieval useless.
"""
from __future__ import annotations

from typing import Any

from core.config.rag_settings import RAG_SETTINGS
from core.rag.models import Chunk


def chunk_paper(paper_id: str, parsed_paper: Any) -> list[Chunk]:
    """Split a parsed paper into section-aware, size-bounded chunks.

    Args:
        paper_id: identifier of the paper under review, propagated onto
            every produced `Chunk` so downstream retrieval can scope by paper.
        parsed_paper: the structured paper object produced by the parsing
            layer (title, abstract, sections, tables, references). Typed as
            `Any` here because the parsing module is out of scope for this
            RAG task; narrow this once `parsing.parsed_paper_schema` exists.

    Returns:
        Ordered list of `Chunk`, each tagged with `section`, `para_idx`,
        and `has_table`.
    """
    # TODO(Phase 1): implement the two-pass split.
    #   Pass 1 - iterate parsed_paper.sections in document order, keeping
    #     each section's own text span separate (do not concatenate first).
    #   Pass 2 - for each section, if its token count exceeds
    #     RAG_SETTINGS.chunking.max_tokens, split into windows sized between
    #     min_tokens and max_tokens with overlap_tokens of shared text
    #     between consecutive windows (so a claim split across a boundary is
    #     still retrievable from either chunk).
    #   Tables: keep any paragraph containing a table intact in one chunk
    #     (has_table=True) even if that pushes it over max_tokens - splitting
    #     a table's cells across two chunks destroys its meaning.
    raise NotImplementedError


def _split_section_into_windows(section_text: str, section_name: str) -> list[str]:
    """Token-bounded sliding-window split of a single section's text.

    Args:
        section_text: raw text of one section.
        section_name: name of the section being split, for error messages.

    Returns:
        List of overlapping text windows, each within
        [min_tokens, max_tokens] tokens (see `ChunkingSettings`).
    """
    # TODO(Phase 1): implement sliding window over a tokenizer's token ids
    #   (not naive whitespace split - token counts must match what the
    #   embedding model actually consumes) using RAG_SETTINGS.chunking.
    raise NotImplementedError


def _count_tokens(text: str) -> int:
    """Return the token count of `text` under the embedding model's tokenizer.

    Kept as its own function so the chunker and the embedding provider agree
    on what "300-500 tokens" means without duplicating tokenizer setup.
    """
    # TODO(Phase 1): back this with the tokenizer of
    #   RAG_SETTINGS.paper_index.embedding_model (e.g. via
    #   transformers.AutoTokenizer) so token accounting matches what actually
    #   gets embedded.
    raise NotImplementedError
