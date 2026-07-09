"""
Fast, fully-mocked verification of the human-in-the-loop approval gate. Mocks
every agent's .run() so this runs in milliseconds and focuses on the one thing
that's genuinely new: the review run pauses at human_approval (interrupt()),
parks on the checkpointer, and resumes correctly for each kind of decision --

  1. approved -> recommendation issued unchanged.
  2. rejected -> decision recorded, recommendation left as drafted.
  3. revised  -> human override rewrites final_recommendation.

Run with: python -m scripts.test_human_approval
"""
from unittest.mock import MagicMock

from langgraph.types import Command

from core.agents.adversarial_critic_agent import AdversarialCriticAgent
from core.agents.citation_agent import CitationAgent
from core.agents.evidence_reproducibility_agent import EvidenceReproducibilityAgent
from core.agents.figure_table_agent import FigureTableAgent
from core.agents.final_review_agent import FinalReviewAgent
from core.agents.literature_rag_agent import LiteratureRAGAgent
from core.agents.methodology_agent import MethodologyAgent
from core.agents.novelty_agent import NoveltyAgent
from core.agents.paper_understanding_agent import PaperUnderstandingAgent
from core.agents.reference_usage_agent import ReferenceUsageAgent
from core.agents.reflection_agent import ReflectionAgent
from core.schemas.agent_output_schemas import (
    AdversarialCritique, CitationAssessment, EvidenceReproducibilityAssessment,
    FigureTableSummary, FinalReview, LiteratureContext, MethodologyAssessment,
    NoveltyAssessment, ParsedPaper, PaperUnderstandingOutput, ReferenceUsageAssessment,
    ReflectionNotes,
)

# --- Mock every agent so no LLM/retrieval is touched. Reflection returns a
#     clean verdict so we go straight to synthesis (the revision loop is
#     covered by test_graph_topology.py; here we only care about approval). ---
PaperUnderstandingAgent.run = lambda self, inputs: PaperUnderstandingOutput(
    summary="s", stated_contributions=["c1"], key_terms=["t1"])
LiteratureRAGAgent.run = lambda self, inputs: LiteratureContext(query_text="q", matches=[])
FigureTableAgent.run = lambda self, inputs: FigureTableSummary(
    figure_summaries=[], table_summaries=[], extraction_consistency_note="ok")
ReferenceUsageAgent.run = lambda self, inputs: ReferenceUsageAssessment(
    reference_verdicts=[], overall_rating="good", summary="s")
NoveltyAgent.run = lambda self, inputs: NoveltyAssessment(
    contribution_verdicts=[], overlapping_work=[], novelty_rating="high", reasoning="r")
MethodologyAgent.run = lambda self, inputs: MethodologyAssessment(
    aspect_verdicts=[], missing_baselines=[], soundness_rating="good", reasoning="r")
CitationAgent.run = lambda self, inputs: CitationAssessment(
    coverage_verdicts=[], citation_quality_rating="good", reasoning="r")
EvidenceReproducibilityAgent.run = lambda self, inputs: EvidenceReproducibilityAssessment(
    claim_verdicts=[], reproducibility_verdicts=[], overall_rating="good", reasoning="r")
AdversarialCriticAgent.run = lambda self, inputs: AdversarialCritique(
    attacks=[], weakest_agent="methodology", summary="nothing worth attacking")
ReflectionAgent.run = lambda self, inputs: ReflectionNotes(
    flags=[], needs_revision=False, overall_confidence="high", summary="clean")
FinalReviewAgent.run = lambda self, inputs: FinalReview(
    paper_summary="s", strengths=["good idea"], weaknesses=["thin ablations"],
    questions_for_authors=["why no baseline X?"], novelty_analysis="n",
    citation_quality="c", reference_usage_quality="ru", reproducibility="r", evidence_mapping="e",
    missing_baselines=[], final_recommendation="weak_accept", confidence="medium")

from core.graph.build_graph import build_review_graph  # noqa: E402 (patches must apply first)

PARSED = ParsedPaper(
    title="T", abstract="A", sections=[], tables=[], figures=[], references=[], source_pdf_path="x.pdf")


def _run_until_interrupt(graph, thread_id):
    """Invoke a fresh run; it should pause at the approval interrupt."""
    result = graph.invoke({"parsed_paper": PARSED}, config={"configurable": {"thread_id": thread_id}})
    assert "__interrupt__" in result, "run should have paused at human_approval, but it ran to completion"
    payload = result["__interrupt__"][0].value
    assert payload["type"] == "approval_request"
    assert payload["draft_recommendation"] == "weak_accept", payload["draft_recommendation"]
    # The draft exists but has NOT been issued -- it's parked awaiting a human.
    assert result["final_review"].final_recommendation == "weak_accept"
    assert "human_approval" not in result, "approval must not exist before a human decides"
    return payload


def _resume(graph, thread_id, decision):
    return graph.invoke(Command(resume=decision), config={"configurable": {"thread_id": thread_id}})


graph = build_review_graph(llm=MagicMock(), prompt_manager=MagicMock())

# --- Scenario 1: approve as-drafted ---
_run_until_interrupt(graph, "approve-run")
final = _resume(graph, "approve-run", {"decision": "approve", "approver": "dr. reviewer"})
approval = final["human_approval"]
assert approval.decision == "approved", approval.decision
assert approval.approver == "dr. reviewer"
assert approval.decided_at is not None, "decided_at should be auto-stamped"
assert final["final_review"].final_recommendation == "weak_accept", "approve must not change the recommendation"
print(f"[approve]  decision={approval.decision!r}  recommendation={final['final_review'].final_recommendation!r}  at={approval.decided_at}")

# --- Scenario 2: reject outright ---
_run_until_interrupt(graph, "reject-run")
final = _resume(graph, "reject-run", {"decision": "reject", "comment": "review is too speculative"})
approval = final["human_approval"]
assert approval.decision == "rejected", approval.decision
assert approval.comment == "review is too speculative"
assert final["final_review"].final_recommendation == "weak_accept", "reject records the decision, leaves the draft as-is"
print(f"[reject]   decision={approval.decision!r}  comment={approval.comment!r}")

# --- Scenario 3: human overrides the recommendation ---
_run_until_interrupt(graph, "override-run")
final = _resume(graph, "override-run",
                {"decision": "revised", "override_recommendation": "reject", "comment": "unsupported headline claim"})
approval = final["human_approval"]
assert approval.decision == "revised", approval.decision
assert approval.override_recommendation == "reject"
assert final["final_review"].final_recommendation == "reject", "override MUST rewrite the issued recommendation"
print(f"[override] model said 'weak_accept' -> human issued {final['final_review'].final_recommendation!r}")

# --- Scenario 4: terse bare-string resume ("approve") still works ---
_run_until_interrupt(graph, "terse-run")
final = _resume(graph, "terse-run", "approve")
assert final["human_approval"].decision == "approved"
print(f"[terse]    bare-string 'approve' -> decision={final['human_approval'].decision!r}")

print("\nALL HUMAN-IN-THE-LOOP APPROVAL ASSERTIONS PASSED")
