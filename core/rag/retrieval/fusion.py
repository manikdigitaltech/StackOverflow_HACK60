"""Reciprocal Rank Fusion (RRF) — merges dense and sparse rankings for Index A.

RRF is chosen over score-normalization-and-sum because dense cosine scores
and BM25 scores live on incomparable scales; RRF sidesteps that by only ever
looking at *rank position* within each list, not the raw score magnitude.
"""
from __future__ import annotations

from core.config.rag_settings import RAG_SETTINGS


def reciprocal_rank_fusion(
    rankings: list[list[tuple[int, float]]],
    k: int = RAG_SETTINGS.paper_index.rrf_k,
) -> list[tuple[int, float]]:
    """Fuse multiple rankings of the same candidate ids into one ranking.

    Args:
        rankings: one list per ranking method (e.g. [dense_results,
            sparse_results]), each a list of (candidate_id, original_score)
            ordered best-first. The original_score is ignored by RRF - only
            position within the list matters.
        k: RRF damping constant. Higher k flattens the influence of rank
            position (top ranks matter relatively less); 60 is the standard
            default from the original RRF paper.

    Returns:
        Fused list of (candidate_id, rrf_score), ordered best-first, where
        rrf_score = sum over rankings of 1 / (k + rank_in_that_ranking).
        A candidate absent from a given ranking contributes 0 for that term.
    """
    # TODO(Phase 3): for each ranking, for each (candidate_id, _) at position
    #   `rank` (0-indexed), accumulate 1 / (k + rank + 1) into a running
    #   score dict keyed by candidate_id. Sort the merged dict by score desc.
    raise NotImplementedError
