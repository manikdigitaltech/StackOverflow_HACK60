"""Index A - Paper-RAG: ephemeral, chunk-level, hybrid dense+sparse.

Rebuilt fresh for every review run and discarded afterward. Backs the
`retrieve_from_paper` tool. Combines a FAISS dense index over `bge-small`
embeddings with a BM25 sparse index over the same chunks, fused with RRF
(see `core.rag.retrieval.fusion`) - dense retrieval alone misses exact
term/number matches (e.g. a specific metric name or hyperparameter value)
that reviewers care about, which is exactly what BM25 is good at.
"""
from __future__ import annotations

import re

from core.config.rag_settings import RAG_SETTINGS
from core.config.settings import settings
from core.rag.embeddings.embedding_provider import BgeSmallEmbeddingProvider
from core.rag.models import Chunk, RetrievalResult
from core.rag.retrieval.fusion import reciprocal_rank_fusion


def _tokenize(text: str) -> list[str]:
    """Shared BM25 tokenization for chunks and queries - must stay identical
    on both sides or sparse scores are meaningless."""
    return re.findall(r"[a-z0-9]+", text.lower())


class PaperIndex:
    """In-memory hybrid index over one paper's chunks. One instance per review run."""

    def __init__(self, embedding_provider: BgeSmallEmbeddingProvider | None = None):
        self._embedding_provider = embedding_provider or BgeSmallEmbeddingProvider(device=settings.embeddings.device)
        self._faiss_index = None
        self._bm25_index = None
        self._chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        """Embed and index a paper's chunks for both dense and sparse retrieval.

        Args:
            chunks: output of `core.rag.chunking.section_chunker.chunk_paper`.
        """
        import faiss
        from rank_bm25 import BM25Okapi

        if not chunks:
            raise ValueError("PaperIndex.build called with no chunks")
        self._chunks = list(chunks)
        vectors = self._embedding_provider.embed([c.text for c in self._chunks])
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        self._faiss_index = index
        # same chunk order as the FAISS rows: index i in both structures
        # refers to self._chunks[i]
        self._bm25_index = BM25Okapi([_tokenize(c.text) for c in self._chunks])

    def search_dense(self, query: str, k: int) -> list[tuple[int, float]]:
        """Dense-only search, used standalone in Phase 2 before hybrid fusion lands.

        Returns:
            List of (chunk_index, cosine_score) pairs, best first.
        """
        if self._faiss_index is None:
            raise RuntimeError("PaperIndex.build(...) must be called before searching.")
        query_vector = self._embedding_provider.embed([query])
        scores, indices = self._faiss_index.search(query_vector, min(k, len(self._chunks)))
        return [(int(i), float(s)) for s, i in zip(scores[0], indices[0]) if i >= 0]

    def search_sparse(self, query: str, k: int) -> list[tuple[int, float]]:
        """BM25-only search over the same chunk order as `search_dense`.

        Returns:
            List of (chunk_index, bm25_score) pairs, best first.
        """
        if self._bm25_index is None:
            raise RuntimeError("PaperIndex.build(...) must be called before searching.")
        scores = self._bm25_index.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(i, float(scores[i])) for i in ranked]

    def retrieve(
        self,
        query: str,
        section_filter: str | None = None,
        k: int = RAG_SETTINGS.paper_index.default_top_k,
    ) -> list[RetrievalResult]:
        """Hybrid retrieve: fuse dense + sparse rankings via RRF, optionally
        restricted to one section.

        Args:
            query: natural-language question an agent wants grounded.
            section_filter: if set, only chunks whose `section` matches are
                eligible - e.g. a methodology agent asking only within
                "method"/"experiments".
            k: number of results to return after fusion.

        Returns:
            Up to `k` `RetrievalResult`s with source="paper_rag", ranked by
            fused RRF score.
        """
        # over-fetch both rankings so post-fusion section filtering still
        # leaves k results (pre-search filtered FAISS is the future upgrade)
        fetch_k = max(k * 3, k + 10) if section_filter else k
        fused = reciprocal_rank_fusion([
            self.search_dense(query, fetch_k),
            self.search_sparse(query, fetch_k),
        ])
        results: list[RetrievalResult] = []
        for chunk_index, rrf_score in fused:
            chunk = self._chunks[chunk_index]
            if section_filter is not None and chunk.section != section_filter:
                continue
            results.append(RetrievalResult(
                source="paper_rag",
                score=rrf_score,
                content=chunk.text,
                metadata={
                    "chunk_id": chunk.chunk_id, "section": chunk.section,
                    "para_idx": chunk.para_idx, "has_table": chunk.has_table,
                },
            ))
            if len(results) == k:
                break
        return results
