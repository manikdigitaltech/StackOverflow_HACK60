"""
config.py

Central configuration for the Novelty Evaluation Agent. No hardcoded
values live in the logic modules - everything tunable is declared here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from core.config.settings import settings

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger configured with a consistent format.

    Args:
        name: Usually ``__name__`` of the calling module.
        level: Logging level (default INFO).

    Returns:
        A configured ``logging.Logger``.
    """
    logging.basicConfig(level=level, format=LOG_FORMAT)
    return logging.getLogger(name)


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------

EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
EMBEDDING_DEVICE: str = settings.embeddings.device
EMBEDDING_BATCH_SIZE: int = 16
EMBEDDING_DIM: int = 384  # native dimensionality of all-MiniLM-L6-v2

# Sections evaluated independently for similarity.
SECTION_FIELDS: List[str] = ["abstract", "methodology", "conclusion", "references"]

# Section-header aliases used to locate methodology / conclusion text in
# the paper's `sections` list.
METHODOLOGY_ALIASES: List[str] = [
    "method", "methods", "methodology", "approach", "model",
    "proposed method", "proposed approach", "architecture",
]
CONCLUSION_ALIASES: List[str] = [
    "conclusion", "conclusions", "concluding remarks", "summary", "future work",
]

# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

TOP_K: int = 10
FAISS_INDEX_FIELD: str = "abstract"

# --------------------------------------------------------------------------
# Weighted similarity
# --------------------------------------------------------------------------

SECTION_WEIGHTS: Dict[str, float] = {
    "abstract": 0.40,
    "methodology": 0.35,
    "conclusion": 0.15,
    "references": 0.10,
}
assert abs(sum(SECTION_WEIGHTS.values()) - 1.0) < 1e-6, "SECTION_WEIGHTS must sum to 1.0"

# --------------------------------------------------------------------------
# Novelty / confidence / recommendation
# --------------------------------------------------------------------------

NOVELTY_BANDS: List[Tuple[float, str]] = [
    (80.0, "Very High Novelty"),
    (60.0, "High Novelty"),
    (40.0, "Moderate Novelty"),
    (20.0, "Low Novelty"),
    (0.0, "Very Low Novelty"),
]

# Recommendation mapping keyed by novelty band label.
RECOMMENDATION_MAP: Dict[str, str] = {
    "Very High Novelty": "Strong Accept",
    "High Novelty": "Accept",
    "Moderate Novelty": "Weak Accept",
    "Borderline Novelty": "Borderline",
    "Low Novelty": "Weak Reject",
    "Very Low Novelty": "Reject",
    "Duplicate": "Strong Reject",
}

# Threshold below which overall_similarity is considered near-duplicate,
# forcing a "Strong Reject" regardless of the novelty band.
DUPLICATE_SIMILARITY_THRESHOLD: float = 95.0

# Confidence scoring: base confidence derived from how many top-K
# neighbours were actually retrieved (evidence volume) and the spread
# between the closest and furthest section-similarity scores (signal
# clarity). Both are blended and clamped to [0, 100].
CONFIDENCE_BASE: float = 50.0
CONFIDENCE_SPREAD_WEIGHT: float = 0.5
CONFIDENCE_EVIDENCE_WEIGHT: float = 20.0  # max bonus for having a full top-K neighbourhood

# Section-similarity thresholds used for decision-trace rule labelling.
HIGH_SIMILARITY_THRESHOLD: float = 70.0
LOW_SIMILARITY_THRESHOLD: float = 30.0


@dataclass(frozen=True)
class AgentPaths:
    """Filesystem paths used by the agent.

    Namespaced under novelty_* so this agent's per-corpus FAISS index
    never collides with data/literature_index (kanishka's RAG Index B,
    specter2) or the legacy data/faiss_index (retired bge-large index) --
    three separate FAISS indexes with different embedding models and
    lifecycles living in the same repo.
    """

    corpus_dir: Path = Path("data/novelty_corpus")
    index_dir: Path = Path("data/novelty_index")


DEFAULT_PATHS = AgentPaths()
