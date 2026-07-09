"""Behavior contracts for core.rag.retrieval.query_helpers.

Targets Phase 6. Since both functions depend on an LLM, tests inject a
fake/stub LLM callable rather than hitting a real model.
"""
from core.rag.retrieval.query_helpers import decompose_query, hyde_query


def test_decompose_query_returns_multiple_subqueries_for_compound_question():
    """A compound question ('does X adequately cover Y and Z?') decomposes into >= 2 sub-queries."""
    fake_llm = lambda prompt: "prior work on X\nrelated work section coverage"
    subs = decompose_query(
        "does this paper's related-work section adequately cover prior work on X?",
        llm=fake_llm,
    )
    assert len(subs) >= 2
    # heuristic path (no LLM) must also split an explicit conjunction
    assert len(decompose_query("coverage of graph pruning and sparse attention baselines")) >= 2


def test_decompose_query_returns_original_question_for_simple_input():
    """A single already-focused question should decompose to a list containing itself (no over-splitting)."""
    question = "what dataset is used for evaluation?"
    fake_llm = lambda prompt: question  # LLM judges it already focused
    assert decompose_query(question, llm=fake_llm) == [question]
    assert decompose_query(question) == [question]  # heuristic path agrees


def test_hyde_query_returns_nonempty_passage():
    """hyde_query returns a non-empty, abstract-like string distinct from the input question."""
    question = "has anyone applied contrastive pretraining to time-series anomaly detection?"
    fake_llm = lambda prompt: ("We present a contrastive pretraining framework for "
                               "time-series anomaly detection, evaluated on standard benchmarks.")
    passage = hyde_query(question, llm=fake_llm)
    assert passage and passage != question
    # fallback path (no LLM) still produces a usable probe
    fallback = hyde_query(question)
    assert fallback and fallback != question and len(fallback.split()) > 10
