"""
Fast, fully-mocked verification of rebuttal-aware re-review (problem statement
section 8). Mocks every agent so it runs in milliseconds and checks the one
thing that matters: seeding the graph with an author rebuttal actually reaches
the assessment agents, and the revised verdict + before/after comparison come
out correctly.

The mocked FinalReviewAgent reads whether a rebuttal was present (threaded
through to it here via a module flag the mocked assessment agents set) and
returns a more favorable recommendation on the rebuttal pass -- standing in for
a real LLM being persuaded by the authors' response.

Run with: python -m scripts.test_rebuttal_rereview
"""
from unittest.mock import MagicMock

from core.agents.adversarial_critic_agent import AdversarialCriticAgent
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
    AdversarialAttack, AdversarialCritique, CitationAssessment,
    EvidenceReproducibilityAssessment, FigureTableSummary,
    FinalReview, LiteratureContext, MethodologyAssessment, NoveltyAssessment,
    ParsedPaper, PaperUnderstandingOutput, ReflectionNotes,
)

# Records whether the last methodology run saw a rebuttal -- lets the mocked
# final-review stand in for "the rebuttal changed the assessment".
_seen = {"rebuttal": False}

PaperUnderstandingAgent.run = lambda self, inputs: PaperUnderstandingOutput(
    summary="s", stated_contributions=["c1"], key_terms=["t1"])
LiteratureRAGAgent.run = lambda self, inputs: LiteratureContext(query_text="q", matches=[])
FigureTableAgent.run = lambda self, inputs: FigureTableSummary(
    figure_summaries=[], table_summaries=[], extraction_consistency_note="ok")
NoveltyAgent.run = lambda self, inputs: NoveltyAssessment(
    contribution_verdicts=[], overlapping_work=[], novelty_rating="medium", reasoning="r")


def _methodology_run(self, inputs):
    # Assert the rebuttal actually reaches the assessment layer on a re-review,
    # and stays absent on a first-pass review.
    _seen["rebuttal"] = bool(inputs.get("rebuttal_text"))
    return MethodologyAssessment(
        aspect_verdicts=[], missing_baselines=[], soundness_rating="good", reasoning="r")


MethodologyAgent.run = _methodology_run
CitationAgent.run = lambda self, inputs: CitationAssessment(
    coverage_verdicts=[], citation_quality_rating="good", reasoning="r")
EvidenceReproducibilityAgent.run = lambda self, inputs: EvidenceReproducibilityAssessment(
    claim_verdicts=[], reproducibility_verdicts=[], overall_rating="good", reasoning="r")
AdversarialCriticAgent.run = lambda self, inputs: AdversarialCritique(
    attacks=[AdversarialAttack(
        source_agent="methodology", attacked_verdict="v", counter_argument="c", severity="moderate")],
    weakest_agent="methodology", summary="s")
ReflectionAgent.run = lambda self, inputs: ReflectionNotes(
    flags=[], needs_revision=False, overall_confidence="high", summary="clean")


def _final_review_run(self, inputs):
    # First pass: weak_reject. Rebuttal pass (rebuttal seen): the authors'
    # response resolves the concern -> weak_accept.
    rec = "weak_accept" if _seen["rebuttal"] else "weak_reject"
    return FinalReview(
        paper_summary="s", strengths=[], weaknesses=[], questions_for_authors=[],
        novelty_analysis="n", citation_quality="c", reproducibility="r", evidence_mapping="e",
        missing_baselines=[], final_recommendation=rec, confidence="medium")


FinalReviewAgent.run = _final_review_run

from core.graph.build_graph import build_review_graph  # noqa: E402 (patches must apply first)
from core.graph.rebuttal import compare_recommendations, run_rebuttal_rereview  # noqa: E402

PARSED = ParsedPaper(
    title="T", abstract="A", sections=[], tables=[], figures=[], references=[], source_pdf_path="x.pdf")

graph = build_review_graph(llm=MagicMock(), prompt_manager=MagicMock())

# --- Original review (no rebuttal): parks at approval with weak_reject ---
original_state = graph.invoke({"parsed_paper": PARSED}, config={"configurable": {"thread_id": "orig"}})
assert _seen["rebuttal"] is False, "first-pass review must NOT see a rebuttal"
original_rec = original_state["final_review"].final_recommendation
assert original_rec == "weak_reject", original_rec
print(f"[original] recommendation = {original_rec!r}")

# --- Rebuttal-aware re-review: same graph, seeded with the author rebuttal ---
result = run_rebuttal_rereview(
    PARSED,
    rebuttal_text="We have added the missing ImageNet baseline (Table 4) and released code; "
                  "the reviewer's reproducibility concern is now addressed.",
    graph=graph,
    original_recommendation=original_rec,
)
assert _seen["rebuttal"] is True, "re-review MUST propagate rebuttal_text to the assessment agents"
revised = result["revised_review"]
assert revised is not None and result["awaiting_approval"], "revised verdict should be drafted, parked at approval"
assert revised.final_recommendation == "weak_accept", revised.final_recommendation

cmp = result["comparison"]
assert cmp["changed"] and cmp["direction"] == "more_favorable" and cmp["steps"] == 2, cmp
print(f"[rebuttal] recommendation = {revised.final_recommendation!r}  "
      f"(changed={cmp['changed']}, direction={cmp['direction']}, steps={cmp['steps']:+d})")

# --- Direction helper sanity checks ---
assert compare_recommendations("reject", "reject")["direction"] == "unchanged"
assert compare_recommendations("accept", "borderline")["direction"] == "less_favorable"

print("\nALL REBUTTAL RE-REVIEW ASSERTIONS PASSED")
