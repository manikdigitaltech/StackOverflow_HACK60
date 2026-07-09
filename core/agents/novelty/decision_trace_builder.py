"""
decision_trace_builder.py

Builds an explainable, ordered decision trace describing why a given
novelty score/recommendation was reached, e.g.:

    High methodology similarity -> Reference overlap -> Low novelty -> Weak Reject
"""

from __future__ import annotations

from typing import Optional

from .config import HIGH_SIMILARITY_THRESHOLD, LOW_SIMILARITY_THRESHOLD, get_logger
from .models import DecisionTrace, SimilarityBreakdown

logger = get_logger(__name__)


class DecisionTraceBuilder:
    """Builds a human-readable, ordered chain of reasoning steps.

    Single responsibility: explanation only. Does not alter or
    recompute any score - it purely narrates the already-computed
    similarity breakdown, novelty band, and recommendation.

    Example:
        >>> builder = DecisionTraceBuilder()
        >>> trace = builder.build(breakdown, novelty_band="Low Novelty", recommendation="Weak Reject")
    """

    def build(
        self,
        breakdown: SimilarityBreakdown,
        novelty_band: str,
        recommendation: str,
        closest_paper_id: Optional[str] = None,
    ) -> DecisionTrace:
        """Construct the ordered decision trace.

        Args:
            breakdown: Section-wise similarity breakdown vs. the closest match.
            novelty_band: The novelty category label (e.g. "Low Novelty").
            recommendation: The final recommendation label (e.g. "Weak Reject").
            closest_paper_id: Identifier of the closest matching paper, if known.

        Returns:
            A ``DecisionTrace`` with ordered rule labels.
        """
        rules = []

        for section, value in (
            ("methodology", breakdown.methodology),
            ("abstract", breakdown.abstract),
            ("references", breakdown.references),
            ("conclusion", breakdown.conclusion),
        ):
            if value is None:
                continue
            if value >= HIGH_SIMILARITY_THRESHOLD:
                rules.append(self._high_label(section))
            elif value <= LOW_SIMILARITY_THRESHOLD:
                rules.append(self._low_label(section))

        if closest_paper_id:
            rules.append(f"Closest match: {closest_paper_id}")

        rules.append(novelty_band)
        rules.append(recommendation)

        trace = DecisionTrace(rules=rules)
        logger.info("Decision trace: %s", " -> ".join(trace.rules))
        return trace

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _high_label(section: str) -> str:
        return {
            "methodology": "High methodology similarity",
            "abstract": "High abstract similarity",
            "references": "Reference overlap",
            "conclusion": "High conclusion similarity",
        }[section]

    @staticmethod
    def _low_label(section: str) -> str:
        return {
            "methodology": "Distinct methodology",
            "abstract": "Distinct research objective",
            "references": "Low reference overlap",
            "conclusion": "Distinct conclusion",
        }[section]
