"""
models.py

Dataclasses used across the Novelty Evaluation Agent. Kept separate from
logic modules so every component depends only on these shared, stable
data contracts (Interface Segregation / Dependency Inversion).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class PaperRecord:
    """A parsed paper ready for embedding and comparison.

    Attributes:
        paper_id: Unique identifier for the paper.
        title: Paper title.
        abstract: Paper abstract text.
        keywords: List of extracted/declared keywords (may be empty).
        methodology: Concatenated methodology-section text (may be empty).
        conclusion: Concatenated conclusion-section text (may be empty).
        references: Concatenated reference strings (may be empty).
        year: Publication year, if available.
    """

    paper_id: str
    title: str = ""
    abstract: str = ""
    keywords: List[str] = field(default_factory=list)
    methodology: str = ""
    conclusion: str = ""
    references: str = ""
    year: Optional[int] = None


@dataclass
class PaperEmbedding:
    """Per-section embedding vectors for a single paper.

    Attributes:
        paper_id: Identifier of the owning paper.
        vectors: Mapping of section name -> L2-normalized embedding vector.
            An absent or all-empty section is represented by omission from
            this dict (not a zero vector), so downstream code can
            distinguish "no data" from "genuinely dissimilar".
    """

    paper_id: str
    vectors: Dict[str, np.ndarray] = field(default_factory=dict)

    def get(self, section: str) -> Optional[np.ndarray]:
        return self.vectors.get(section)


@dataclass
class SimilarPaper:
    """A single FAISS retrieval result.

    Attributes:
        paper_id: Identifier of the retrieved paper.
        similarity: Cosine similarity score, scaled to [0, 100].
    """

    paper_id: str
    similarity: float

    def to_dict(self) -> Dict:
        return {"paper_id": self.paper_id, "similarity": round(self.similarity, 2)}


@dataclass
class SimilarityBreakdown:
    """Section-wise similarity of a target paper against its closest match.

    Values are ``None`` when a section had no text to compare (rather
    than a misleading 0.0), and a [0, 100] float otherwise.
    """

    abstract: Optional[float] = None
    methodology: Optional[float] = None
    conclusion: Optional[float] = None
    references: Optional[float] = None

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "abstract": self._round(self.abstract),
            "methodology": self._round(self.methodology),
            "conclusion": self._round(self.conclusion),
            "references": self._round(self.references),
        }

    @staticmethod
    def _round(value: Optional[float]) -> Optional[float]:
        return round(value, 2) if value is not None else None


@dataclass
class DecisionTrace:
    """Explainable, ordered chain of reasoning steps behind a decision.

    Attributes:
        rules: Ordered list of human-readable rule labels that fired,
            e.g. ["High methodology similarity", "Reference overlap",
            "Low novelty", "Weak Reject"].
    """

    rules: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, List[str]]:
        return {"rules": self.rules}


@dataclass
class NoveltyReport:
    """Final structured output of the Novelty Evaluation Agent.

    Matches the required output schema exactly.
    """

    paper_id: str
    novelty_score: float
    overall_similarity: float
    confidence: float
    recommendation: str
    similarity_breakdown: SimilarityBreakdown = field(default_factory=SimilarityBreakdown)
    top_similar_papers: List[SimilarPaper] = field(default_factory=list)
    decision_trace: DecisionTrace = field(default_factory=DecisionTrace)

    def to_dict(self) -> Dict:
        return {
            "paper_id": self.paper_id,
            "novelty_score": round(self.novelty_score, 2),
            "overall_similarity": round(self.overall_similarity, 2),
            "confidence": round(self.confidence, 2),
            "recommendation": self.recommendation,
            "similarity_breakdown": self.similarity_breakdown.to_dict(),
            "top_similar_papers": [p.to_dict() for p in self.top_similar_papers],
            "decision_trace": self.decision_trace.to_dict(),
        }
