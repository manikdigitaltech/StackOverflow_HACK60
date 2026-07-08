"""Behavior contracts for core.rag.retrieval.query_helpers.

Targets Phase 6. Since both functions depend on an LLM, tests should inject
a fake/stub LLM callable rather than hitting a real model.
"""


def test_decompose_query_returns_multiple_subqueries_for_compound_question():
    """A compound question ('does X adequately cover Y and Z?') decomposes into >= 2 sub-queries."""
    raise NotImplementedError


def test_decompose_query_returns_original_question_for_simple_input():
    """A single already-focused question should decompose to a list containing itself (no over-splitting)."""
    raise NotImplementedError


def test_hyde_query_returns_nonempty_passage():
    """hyde_query returns a non-empty, abstract-like string distinct from the input question."""
    raise NotImplementedError
