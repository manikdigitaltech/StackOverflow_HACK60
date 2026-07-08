"""Behavior contracts for core.rag.retrieval.fusion.reciprocal_rank_fusion.

Targets Phase 3. These should be pure unit tests with hand-constructed
ranking lists - no index or embedding model required.
"""
from core.rag.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_fuses_dense_and_sparse():
    """A candidate ranked #1 in both dense and sparse lists must outrank one appearing in only one list."""
    dense = [(1, 0.9), (2, 0.5), (3, 0.4)]
    sparse = [(1, 12.0), (4, 3.0)]
    fused = reciprocal_rank_fusion([dense, sparse])
    assert fused[0][0] == 1
    ids = [cid for cid, _ in fused]
    assert ids.index(1) < ids.index(4)


def test_rrf_handles_candidate_present_in_only_one_ranking():
    """A candidate absent from one input ranking still receives a fused score (contributes 0 for that ranking)."""
    fused = reciprocal_rank_fusion([[(7, 0.8)], [(9, 5.0)]])
    ids = {cid for cid, _ in fused}
    assert ids == {7, 9}
    assert all(score > 0 for _, score in fused)


def test_rrf_is_order_preserving_for_identical_rankings():
    """If both input rankings are identical, the fused order must equal that ranking."""
    ranking = [(3, 0.9), (1, 0.7), (2, 0.2)]
    fused = reciprocal_rank_fusion([ranking, ranking])
    assert [cid for cid, _ in fused] == [3, 1, 2]


def test_rrf_k_parameter_changes_relative_weighting():
    """A smaller k should more strongly favor top-ranked candidates than a larger k."""
    dense = [(1, 0.9), (2, 0.8)]
    sparse = [(2, 9.0), (1, 8.0)]  # each candidate is rank-1 once, rank-2 once
    small = dict(reciprocal_rank_fusion([dense, [(3, 1.0)]], k=1))
    large = dict(reciprocal_rank_fusion([dense, [(3, 1.0)]], k=1000))
    # ratio between rank-1 and rank-2 scores collapses toward 1 as k grows
    assert small[1] / small[2] > large[1] / large[2]
