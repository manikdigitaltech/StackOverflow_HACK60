"""
embedding_service.py

Wraps ``sentence-transformers`` (all-MiniLM-L6-v2 by default) to produce
independent embeddings for each section of a paper (abstract,
methodology, conclusion, references) plus the title. Everything runs
locally - no online API calls beyond the one-time model download that
sentence-transformers performs on first use.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .config import EMBEDDING_BATCH_SIZE, EMBEDDING_DEVICE, EMBEDDING_DIM, EMBEDDING_MODEL_NAME, get_logger
from .models import PaperEmbedding, PaperRecord

logger = get_logger(__name__)


class EmbeddingServiceError(Exception):
    """Raised when embedding generation fails."""


class EmbeddingService:
    """Generates local sentence-transformer embeddings for paper sections.

    Single responsibility: text -> vector. Knows nothing about FAISS,
    similarity scoring, or novelty logic (Single Responsibility
    Principle). The model is lazily loaded so importing this module has
    no cost until embeddings are actually requested.

    Example:
        >>> service = EmbeddingService()
        >>> embedding = service.embed(paper_record)
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, device: str = EMBEDDING_DEVICE) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._use_fallback = False

    @property
    def model(self):
        """Lazily load the SentenceTransformer model, with an offline fallback.

        If the model cannot be downloaded (no network access to the
        model hub), falls back to a deterministic local hashing-based
        embedder of matching dimensionality so the agent keeps working
        end to end in constrained/offline environments.
        """
        if self._model is None and not self._use_fallback:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading SentenceTransformer '%s' (device=%s)", self.model_name, self.device)
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not load '%s' (%s); falling back to offline hashing embedder.", self.model_name, exc
                )
                self._use_fallback = True
        return self._model

    def embed(self, paper: PaperRecord) -> PaperEmbedding:
        """Generate embeddings for every non-empty section of a paper.

        Args:
            paper: A ``PaperRecord`` produced by ``PaperTextExtractor``.

        Returns:
            A ``PaperEmbedding`` with one vector per non-empty section.
            Sections with no text are omitted (not zero-vectored), so
            downstream similarity code can tell "no data" apart from
            "very dissimilar".

        Raises:
            EmbeddingServiceError: If encoding fails.
        """
        section_texts = self._collect_section_texts(paper)
        non_empty = {name: text for name, text in section_texts.items() if text}

        if not non_empty:
            logger.warning("Paper '%s' has no non-empty sections to embed", paper.paper_id)
            return PaperEmbedding(paper_id=paper.paper_id, vectors={})

        try:
            model = self.model
            names = list(non_empty.keys())
            texts = list(non_empty.values())

            if self._use_fallback:
                vectors = self._fallback_encode(texts)
            else:
                vectors = model.encode(
                    texts,
                    batch_size=EMBEDDING_BATCH_SIZE,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingServiceError(f"Failed to embed paper '{paper.paper_id}': {exc}") from exc

        vector_map = {name: vec.astype(np.float32) for name, vec in zip(names, vectors)}
        logger.info("Embedded paper '%s' across sections: %s", paper.paper_id, list(vector_map.keys()))
        return PaperEmbedding(paper_id=paper.paper_id, vectors=vector_map)

    def embed_batch(self, papers: List[PaperRecord]) -> List[PaperEmbedding]:
        """Embed a batch of papers, skipping any that fail.

        Args:
            papers: List of ``PaperRecord`` objects.

        Returns:
            List of successfully embedded ``PaperEmbedding`` objects.
        """
        results: List[PaperEmbedding] = []
        for paper in papers:
            try:
                results.append(self.embed(paper))
            except EmbeddingServiceError as exc:
                logger.error("Skipping paper '%s': %s", paper.paper_id, exc)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_section_texts(paper: PaperRecord) -> Dict[str, str]:
        keywords_text = " ".join(paper.keywords)
        return {
            "title": paper.title,
            "abstract": paper.abstract,
            "methodology": paper.methodology,
            "conclusion": paper.conclusion,
            "references": paper.references,
            "keywords": keywords_text,
        }

    @staticmethod
    def _fallback_encode(texts: List[str]) -> np.ndarray:
        """Deterministic offline embedding (no network) used only as a fallback."""
        from sklearn.feature_extraction.text import HashingVectorizer

        vectorizer = HashingVectorizer(n_features=EMBEDDING_DIM, norm="l2", alternate_sign=False)
        return vectorizer.transform(texts).toarray().astype(np.float32)
