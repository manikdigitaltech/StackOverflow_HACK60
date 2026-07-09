"""Typed data contracts shared across the RAG subsystem.

These are the nouns every other module speaks in: a `Chunk` is what Index A
indexes, a `CorpusRecord` is what Index B indexes, and a `RetrievalResult` is
the uniform shape every retrieval tool returns, regardless of which source
produced it.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A section-aware slice of the paper under review.

    Produced by the chunker in `core.rag.chunking.section_chunker`, embedded
    with `bge-small-en-v1.5`, and stored in the ephemeral Index A (Paper-RAG).
    One `Chunk` == one row in the FAISS index and one document in the BM25
    corpus for a single review run.
    """

    chunk_id: str
    paper_id: str
    section: Literal[
        "abstract",
        "introduction",
        "method",
        "experiments",
        "results",
        "related_work",
        "conclusion",
        "other",
    ]
    para_idx: int
    text: str
    has_table: bool = False
    token_count: int


class CorpusRecord(BaseModel):
    """One paper-level record in the persistent literature corpus.

    Produced once, offline, by `core.rag.ingestion.build_corpus`, embedded
    with `specter2_base` (title + abstract concatenated), and stored in the
    persistent Index B (Literature-RAG). One `CorpusRecord` == one vector,
    unlike `Chunk` where one paper produces many vectors.
    """

    paper_id: str
    title: str
    abstract: str
    year: Optional[int] = None
    venue: Optional[str] = None
    accepted: Optional[bool] = None


class RetrievalResult(BaseModel):
    """A single ranked hit returned by any retrieval tool.

    Uniform across chunk-level (Index A) and paper-level (Index B, Semantic
    Scholar, arXiv, and the other optional live sources) so that agent code
    can consume results from any tool without branching on shape.
    """

    source: Literal[
        "paper_rag", "literature_rag", "semantic_scholar", "arxiv",
        "google_scholar", "exa", "openalex", "openreview", "dblp",
    ]
    score: float
    content: str
    metadata: dict = Field(default_factory=dict)
