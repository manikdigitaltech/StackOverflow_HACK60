"""
novelty_scorer.py

Computes the novelty score (0-100, 100=highly novel), a confidence
score (0-100), and a peer-review recommendation from a similarity
breakdown and retrieval evidence.
"""

from __future__ import annotations

from typing import List, Optional

from .config import (
    CONFIDENCE_BASE,
    CONFIDENCE_EVIDENCE_WEIGHT,
    CONFIDENCE_SPREAD_WEIGHT,
    DUPLICATE_SIMILARITY_THRESHOLD,
    NOVELTY_BANDS,
    RECOMMENDATION_MAP,
    TOP_K,
    get_logger,
)
from .models import SimilarityBreakdown, SimilarPaper

logger = get_logger(__name__)


class NoveltyScoringError(Exception):
    """Raised when novelty/confidence/recommendation scoring fails."""


class NoveltyScorer:
    """Derives novelty score, confidence, novelty band, and recommendation.

    Single responsibility: turns similarity numbers into a scientific
    judgement. Knows nothing about embeddings or FAISS.

    Example:
        >>> scorer = NoveltyScorer()
        >>> novelty, confidence, band, recommendation = scorer.score(
        ...     overall_similarity, breakdown, top_similar_papers
        ... )
    """

    def score(
        self,
        overall_similarity: float,
        breakdown: SimilarityBreakdown,
        top_similar_papers: List[SimilarPaper],
    ) -> tuple:
        """Compute novelty score, confidence, novelty band, and recommendation.

        Args:
            overall_similarity: Weighted overall similarity to the
                closest matching paper, in [0, 100].
            breakdown: Section-wise similarity breakdown vs. the closest
                matching paper.
            top_similar_papers: Ranked FAISS retrieval results, used as
                an evidence-volume signal for confidence.

        Returns:
            Tuple of (novelty_score, confidence, novelty_band, recommendation).

        Raises:
            NoveltyScoringError: If ``overall_similarity`` is out of range.
        """
        if not (0.0 <= overall_similarity <= 100.0):
            raise NoveltyScoringError(f"overall_similarity must be within [0, 100], got {overall_similarity}")

        try:
            novelty_score = round(100.0 - overall_similarity, 4)
            novelty_band = self._categorize(novelty_score)

            if overall_similarity >= DUPLICATE_SIMILARITY_THRESHOLD:
                novelty_band = "Duplicate"

            confidence = self._compute_confidence(breakdown, top_similar_papers)
            recommendation = RECOMMENDATION_MAP.get(novelty_band, "Borderline")
        except NoveltyScoringError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise NoveltyScoringError(f"Failed to score novelty: {exc}") from exc

        logger.info(
            "Novelty=%.2f band=%s confidence=%.2f recommendation=%s",
            novelty_score, novelty_band, confidence, recommendation,
        )
        return novelty_score, confidence, novelty_band, recommendation

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _categorize(novelty_score: float) -> str:
        for threshold, label in NOVELTY_BANDS:
            if novelty_score >= threshold:
                return label
        return "Very Low Novelty"

    @staticmethod
    def _compute_confidence(breakdown: SimilarityBreakdown, top_similar_papers: List[SimilarPaper]) -> float:
        """Blend signal clarity (score spread) and evidence volume (top-K coverage) into a 0-100 confidence.

        A larger spread between the most and least similar sections
        indicates a clearer, more decisive signal. A fuller top-K
        neighbourhood indicates more corroborating evidence was
        available for the judgement.
        """
        values = [v for v in (breakdown.abstract, breakdown.methodology, breakdown.conclusion, breakdown.references) if v is not None]
        spread = (max(values) - min(values)) if len(values) >= 2 else 0.0

        evidence_ratio = min(1.0, len(top_similar_papers) / TOP_K) if TOP_K else 0.0

        confidence = (
            CONFIDENCE_BASE
            + spread * CONFIDENCE_SPREAD_WEIGHT * 0.5
            + evidence_ratio * CONFIDENCE_EVIDENCE_WEIGHT
        )
        return round(max(0.0, min(100.0, confidence)), 4)
