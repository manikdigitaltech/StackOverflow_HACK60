"""
faiss_retriever.py

Local FAISS retrieval over a corpus of paper embeddings. Uses
``IndexFlatIP`` on L2-normalized vectors (inner product = cosine
similarity). Fully local - no network calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from .config import FAISS_INDEX_FIELD, TOP_K, get_logger
from .models import PaperEmbedding, SimilarPaper

logger = get_logger(__name__)


class FaissRetrieverError(Exception):
    """Raised on FAISS index build, save, load, or search failure."""


class FaissRetriever:
    """Builds and queries a local FAISS index over paper embeddings.

    Single responsibility: nearest-neighbour retrieval only. Does not
    know about novelty scoring or decision logic (Dependency
    Inversion - higher-level agents depend on this abstraction, not the
    reverse).

    Example:
        >>> retriever = FaissRetriever()
        >>> retriever.build(corpus_embeddings)
        >>> results = retriever.search(query_embedding, top_k=10)
    """

    def __init__(self, section: str = FAISS_INDEX_FIELD) -> None:
        self.section = section
        self._index = None
        self._paper_ids: List[str] = []
        self._dim: int = 0

    @property
    def size(self) -> int:
        return len(self._paper_ids)

    def build(self, embeddings: List[PaperEmbedding]) -> None:
        """Build the FAISS index from a corpus of paper embeddings.

        Args:
            embeddings: Embeddings for every paper in the corpus. Papers
                missing the configured section are skipped with a warning.

        Raises:
            FaissRetrieverError: If FAISS is unavailable or no vectors
                are available to index.
        """
        try:
            import faiss
        except ImportError as exc:
            raise FaissRetrieverError("faiss-cpu is not installed") from exc

        vectors, paper_ids = [], []
        for emb in embeddings:
            vec = emb.get(self.section)
            if vec is None:
                logger.warning("Paper '%s' has no '%s' embedding - excluded from index", emb.paper_id, self.section)
                continue
            vectors.append(vec)
            paper_ids.append(emb.paper_id)

        if not vectors:
            raise FaissRetrieverError(f"No '{self.section}' vectors available to build the FAISS index")

        matrix = np.vstack(vectors).astype(np.float32)
        self._dim = matrix.shape[1]

        try:
            index = faiss.IndexFlatIP(self._dim)
            index.add(matrix)
        except Exception as exc:  # noqa: BLE001
            raise FaissRetrieverError(f"Failed to build FAISS index: {exc}") from exc

        self._index = index
        self._paper_ids = paper_ids
        logger.info("Built FAISS index: %d vectors, section='%s', dim=%d", self.size, self.section, self._dim)

    def search(self, query_vector: np.ndarray, top_k: int = TOP_K, exclude_paper_id: str = None) -> List[SimilarPaper]:
        """Search the index for the top-K most similar papers.

        Args:
            query_vector: L2-normalized query vector.
            top_k: Number of neighbours to return.
            exclude_paper_id: If provided, filters out this paper_id
                from the results (useful when the query paper is itself
                part of the indexed corpus).

        Returns:
            List of ``SimilarPaper``, sorted by descending similarity
            (scaled to [0, 100]).

        Raises:
            FaissRetrieverError: If the index has not been built/loaded.
        """
        if self._index is None:
            raise FaissRetrieverError("FAISS index has not been built or loaded yet")

        try:
            query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
            k = min(top_k + (1 if exclude_paper_id else 0), self.size)
            scores, indices = self._index.search(query, k)
        except Exception as exc:  # noqa: BLE001
            raise FaissRetrieverError(f"FAISS search failed: {exc}") from exc

        results: List[SimilarPaper] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            paper_id = self._paper_ids[idx]
            if exclude_paper_id and paper_id == exclude_paper_id:
                continue
            pct = self._cosine_to_pct(float(score))
            results.append(SimilarPaper(paper_id=paper_id, similarity=pct))

        return results[:top_k]

    def save(self, directory: Path) -> None:
        """Persist the index and paper-id mapping to disk."""
        if self._index is None:
            raise FaissRetrieverError("Cannot save an unbuilt index")
        try:
            import faiss

            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(directory / f"{self.section}_index.faiss"))
            with open(directory / f"{self.section}_meta.json", "w", encoding="utf-8") as fh:
                json.dump({"section": self.section, "dim": self._dim, "paper_ids": self._paper_ids}, fh)
        except Exception as exc:  # noqa: BLE001
            raise FaissRetrieverError(f"Failed to save index to '{directory}': {exc}") from exc
        logger.info("Saved FAISS index (%d vectors) to %s", self.size, directory)

    def load(self, directory: Path) -> None:
        """Load a previously saved index and paper-id mapping."""
        directory = Path(directory)
        index_path = directory / f"{self.section}_index.faiss"
        meta_path = directory / f"{self.section}_meta.json"
        if not index_path.is_file() or not meta_path.is_file():
            raise FaissRetrieverError(f"Index files not found in '{directory}' for section '{self.section}'")
        try:
            import faiss

            self._index = faiss.read_index(str(index_path))
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            self._paper_ids = meta["paper_ids"]
            self._dim = meta["dim"]
        except Exception as exc:  # noqa: BLE001
            raise FaissRetrieverError(f"Failed to load index from '{directory}': {exc}") from exc
        logger.info("Loaded FAISS index (%d vectors) from %s", self.size, directory)

    @staticmethod
    def _cosine_to_pct(sim: float) -> float:
        """Map cosine similarity in [-1, 1] to a [0, 100] percentage scale."""
        sim = max(-1.0, min(1.0, sim))
        return round(((sim + 1.0) / 2.0) * 100.0, 4)
