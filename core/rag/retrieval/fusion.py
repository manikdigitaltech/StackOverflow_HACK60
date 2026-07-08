"""Reciprocal Rank Fusion (RRF) - merges dense and sparse rankings for Index A.

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
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, (candidate_id, _original_score) in enumerate(ranking):
            scores[candidate_id] = scores.get(candidate_id, 0.0) + 1.0 / (k + rank + 1)
    # tie-break on candidate_id so fusion output is fully deterministic
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
