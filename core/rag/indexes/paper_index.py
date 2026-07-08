"""Index A — Paper-RAG: ephemeral, chunk-level, hybrid dense+sparse.

Rebuilt fresh for every review run and discarded afterward. Backs the
`retrieve_from_paper` tool. Combines a FAISS dense index over `bge-small`
embeddings with a BM25 sparse index over the same chunks, fused with RRF
(see `core.rag.retrieval.fusion`) — dense retrieval alone misses exact
term/number matches (e.g. a specific metric name or hyperparameter value)
that reviewers care about, which is exactly what BM25 is good at.
"""
from __future__ import annotations

from core.config.rag_settings import RAG_SETTINGS
from core.rag.embeddings.embedding_provider import BgeSmallEmbeddingProvider
from core.rag.models import Chunk, RetrievalResult


class PaperIndex:
    """In-memory hybrid index over one paper's chunks. One instance per review run."""

    def __init__(self, embedding_provider: BgeSmallEmbeddingProvider | None = None):
        self._embedding_provider = embedding_provider or BgeSmallEmbeddingProvider()
        self._faiss_index = None  # TODO(Phase 2): faiss.IndexFlatIP(dim)
        self._bm25_index = None  # TODO(Phase 3): rank_bm25.BM25Okapi over tokenized chunks
        self._chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        """Embed and index a paper's chunks for both dense and sparse retrieval.

        Args:
            chunks: output of `core.rag.chunking.section_chunker.chunk_paper`.
        """
        # TODO(Phase 2): embed chunk texts via self._embedding_provider.embed,
        #   build faiss.IndexFlatIP over the normalized vectors.
        # TODO(Phase 3): tokenize chunk texts (simple whitespace/regex is
        #   fine for BM25) and build the BM25 index over the same chunk order
        #   so index i in both structures refers to the same Chunk.
        raise NotImplementedError

    def search_dense(self, query: str, k: int) -> list[tuple[int, float]]:
        """Dense-only search, used standalone in Phase 2 before hybrid fusion lands.

        Returns:
            List of (chunk_index, cosine_score) pairs, best first.
        """
        # TODO(Phase 2): embed query, faiss_index.search, zip indices+scores.
        raise NotImplementedError

    def search_sparse(self, query: str, k: int) -> list[tuple[int, float]]:
        """BM25-only search over the same chunk order as `search_dense`.

        Returns:
            List of (chunk_index, bm25_score) pairs, best first.
        """
        # TODO(Phase 3): tokenize query the same way chunks were tokenized in
        #   build(), score via self._bm25_index.get_scores, take top k.
        raise NotImplementedError

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
                eligible — e.g. a methodology agent asking only within
                "method"/"experiments".
            k: number of results to return after fusion.

        Returns:
            Up to `k` `RetrievalResult`s with source="paper_rag", ranked by
            fused RRF score.
        """
        # TODO(Phase 3): call search_dense and search_sparse (Phase 2/3 must
        #   land first), pass both rankings to
        #   core.rag.retrieval.fusion.reciprocal_rank_fusion, apply
        #   section_filter either pre-search (restrict candidate chunk ids)
        #   or post-fusion (drop non-matching hits) - pre-search is cheaper
        #   and preferred once the index supports filtered search.
        raise NotImplementedError
