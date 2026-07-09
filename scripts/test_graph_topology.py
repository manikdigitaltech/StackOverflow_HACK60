"""
Fast structural test of the review graph: mocks every agent's .run() so this
completes in milliseconds, and specifically checks the two things that were
genuinely uncertain about LangGraph's execution semantics before this test
existed:
  1. figure_table's one-shot output (wired directly to final_review, not
     through reflection) is actually visible in final_review's inputs.
  2. the bounded revision loop actually re-runs the 4 assessment agents with
     revision_feedback set, and terminates instead of looping forever.

Run with: python -m scripts.test_graph_topology
"""
from unittest.mock import MagicMock

from core.agents.citation_agent import CitationAgent
from core.agents.evidence_reproducibility_agent import EvidenceReproducibilityAgent
from core.agents.figure_table_agent import FigureTableAgent
from core.agents.final_review_agent import FinalReviewAgent
from core.agents.literature_rag_agent import LiteratureRAGAgent
from core.agents.methodology_agent import MethodologyAgent
from core.agents.novelty_agent import NoveltyAgent
from core.agents.paper_understanding_agent import PaperUnderstandingAgent
from core.agents.reflection_agent import ReflectionAgent
from core.schemas.agent_output_schemas import (
    CitationAssessment, EvidenceReproducibilityAssessment, FigureTableSummary,
    FinalReview, LiteratureContext, MethodologyAssessment, NoveltyAssessment,
    ParsedPaper, PaperUnderstandingOutput, ReflectionFlag, ReflectionNotes,
)

call_log = []
citation_calls = []


def _mk_reflection_notes(revision_feedback_seen):
    # First call (no feedback yet): flag a major issue -> forces one revision.
    # Second call (feedback present): clean -> proceeds to final_review.
    if not revision_feedback_seen:
        return ReflectionNotes(
            flags=[ReflectionFlag(source_agent="citation", flagged_item="X", issue="unsupported", severity="major")],
            needs_revision=True, overall_confidence="medium", summary="one major flag",
        )
    return ReflectionNotes(flags=[], needs_revision=False, overall_confidence="high", summary="clean")


PaperUnderstandingAgent.run = lambda self, inputs: (
    call_log.append("paper_understanding"),
    PaperUnderstandingOutput(summary="s", stated_contributions=["c1"], key_terms=["t1"]),
)[1]
LiteratureRAGAgent.run = lambda self, inputs: (
    call_log.append("literature_rag"), LiteratureContext(query_text="q", matches=[]),
)[1]
FigureTableAgent.run = lambda self, inputs: (
    call_log.append("figure_table"),
    FigureTableSummary(figure_summaries=[], table_summaries=[], extraction_consistency_note="ok"),
)[1]
NoveltyAgent.run = lambda self, inputs: (
    call_log.append("novelty"),
    NoveltyAssessment(contribution_verdicts=[], overlapping_work=[], novelty_rating="medium", reasoning="r"),
)[1]
MethodologyAgent.run = lambda self, inputs: (
    call_log.append("methodology"),
    MethodologyAssessment(aspect_verdicts=[], missing_baselines=[], soundness_rating="good", reasoning="r"),
)[1]


def _citation_run(self, inputs):
    call_log.append("citation")
    citation_calls.append(inputs.get("revision_feedback"))
    return CitationAssessment(coverage_verdicts=[], citation_quality_rating="good", reasoning="r")


CitationAgent.run = _citation_run
EvidenceReproducibilityAgent.run = lambda self, inputs: (
    call_log.append("evidence_reproducibility"),
    EvidenceReproducibilityAssessment(claim_verdicts=[], reproducibility_verdicts=[], overall_rating="good", reasoning="r"),
)[1]


def _reflection_run(self, inputs):
    call_log.append("reflection")
    seen_feedback = citation_calls and citation_calls[-1]
    return _mk_reflection_notes(seen_feedback)


ReflectionAgent.run = _reflection_run


def _final_review_run(self, inputs):
    call_log.append("final_review")
    assert inputs["figure_table_summary"] is not None, "figure_table_summary missing at final_review!"
    return FinalReview(
        paper_summary="s", strengths=[], weaknesses=[], questions_for_authors=[],
        novelty_analysis="n", citation_quality="c", reproducibility="r", evidence_mapping="e",
        missing_baselines=[], final_recommendation="borderline", confidence="medium",
    )


FinalReviewAgent.run = _final_review_run

from core.graph.build_graph import build_review_graph  # noqa: E402 (patches must apply first)

parsed_paper = ParsedPaper(
    title="T", abstract="A", sections=[], tables=[], figures=[], references=[], source_pdf_path="x.pdf",
)

graph = build_review_graph(llm=MagicMock(), prompt_manager=MagicMock())
result = graph.invoke({"parsed_paper": parsed_paper}, config={"configurable": {"thread_id": "test-1"}})

print("Call order:", call_log)
print("Citation calls' revision_feedback values:", citation_calls)
print("revision_count in final state:", result.get("revision_count"))
print("final_review present:", result.get("final_review") is not None)

assert call_log.count("citation") == 2, f"expected citation to run twice (initial + 1 revision), got {call_log.count('citation')}"
assert citation_calls[0] is None, "first citation call should have no revision_feedback"
assert citation_calls[1], "second citation call should have revision_feedback set"
assert call_log.count("figure_table") == 1, "figure_table should run exactly once, never re-triggered by the revision loop"
assert result["revision_count"] == 1
assert result["final_review"] is not None

print("\nALL GRAPH TOPOLOGY ASSERTIONS PASSED")
