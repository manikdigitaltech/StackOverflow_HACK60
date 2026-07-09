"""
similarity_service.py

Computes section-wise cosine similarity (abstract, methodology,
conclusion, references) between a target paper and a candidate paper,
and combines them into a single weighted overall similarity score.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .config import SECTION_FIELDS, SECTION_WEIGHTS, get_logger
from .models import PaperEmbedding, SimilarityBreakdown

logger = get_logger(__name__)


class SimilarityServiceError(Exception):
    """Raised when similarity computation fails."""


class SimilarityService:
    """Computes section-wise and weighted-overall similarity between two papers.

    Single responsibility: pure similarity math. Missing sections yield
    ``None`` (not 0.0) so callers can distinguish "no data" from a
    genuinely low similarity score.

    Example:
        >>> service = SimilarityService()
        >>> breakdown, overall = service.compute(target_embedding, candidate_embedding)
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights = weights or SECTION_WEIGHTS

    def compute(self, target: PaperEmbedding, candidate: PaperEmbedding) -> tuple:
        """Compute the section-wise breakdown and weighted overall similarity.

        Args:
            target: Embeddings of the paper being evaluated.
            candidate: Embeddings of the comparison paper.

        Returns:
            Tuple of (``SimilarityBreakdown``, ``overall_similarity: float``).

        Raises:
            SimilarityServiceError: If no comparable sections exist.
        """
        try:
            scores: Dict[str, Optional[float]] = {}
            for section in SECTION_FIELDS:
                target_vec = target.get(section)
                candidate_vec = candidate.get(section)
                if target_vec is None or candidate_vec is None:
                    scores[section] = None
                    continue
                scores[section] = self._cosine_pct(target_vec, candidate_vec)

            overall = self._weighted_overall(scores)
        except Exception as exc:  # noqa: BLE001
            raise SimilarityServiceError(
                f"Failed to compute similarity between '{target.paper_id}' and '{candidate.paper_id}': {exc}"
            ) from exc

        breakdown = SimilarityBreakdown(
            abstract=scores.get("abstract"),
            methodology=scores.get("methodology"),
            conclusion=scores.get("conclusion"),
            references=scores.get("references"),
        )
        logger.info(
            "Similarity %s vs %s: overall=%.2f", target.paper_id, candidate.paper_id, overall
        )
        return breakdown, overall

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_pct(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Cosine similarity between two vectors, scaled to [0, 100]."""
        sim = float(cosine_similarity(vec_a.reshape(1, -1), vec_b.reshape(1, -1))[0][0])
        sim = max(-1.0, min(1.0, sim))
        return round(((sim + 1.0) / 2.0) * 100.0, 4)

    def _weighted_overall(self, scores: Dict[str, Optional[float]]) -> float:
        """Weighted sum over available sections, renormalized for missing ones."""
        total_weight = 0.0
        weighted_sum = 0.0
        for section, weight in self.weights.items():
            value = scores.get(section)
            if value is not None:
                weighted_sum += value * weight
                total_weight += weight
        if total_weight == 0:
            return 0.0
        return round(weighted_sum / total_weight, 4)
