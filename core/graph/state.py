"""
Shared state schema for the review orchestration graph.

Each agent's real output type is the value type for its own key -- the graph's
job is deciding WHEN each agent runs and what of the accumulated state it gets
to see, not transforming data between agents. Parallel branches never write
the same key (verified against each node's actual return in nodes.py), which
is what lets LangGraph merge concurrent branch updates into one state dict
without needing custom reducers.
"""
from __future__ import annotations

from typing import Optional, TypedDict

from core.schemas.agent_output_schemas import (
    AdversarialCritique,
    CitationAssessment,
    EvidenceReproducibilityAssessment,
    FigureTableSummary,
    FinalReview,
    HumanApproval,
    LiteratureContext,
    MethodologyAssessment,
    NoveltyAssessment,
    ParsedPaper,
    PaperUnderstandingOutput,
    ReflectionNotes,
)


class ReviewGraphState(TypedDict, total=False):
    # --- Input ---
    parsed_paper: ParsedPaper

    # --- Stage 1: parallel, each only needs parsed_paper ---
    paper_understanding: PaperUnderstandingOutput
    literature_context: LiteratureContext
    figure_table_summary: FigureTableSummary

    # --- Stage 2: parallel assessments ---
    novelty_assessment: NoveltyAssessment
    methodology_assessment: MethodologyAssessment
    citation_assessment: CitationAssessment
    evidence_assessment: EvidenceReproducibilityAssessment

    # --- Stage 3: self-reflection + bounded revision loop ---
    # Runs in parallel with reflection, attacking only methodology/citation/
    # evidence (not novelty -- out of scope by design). Feeds INTO reflection
    # as an extra input (see nodes.py/build_graph.py), not a separate branch
    # of the revision loop.
    adversarial_critique: AdversarialCritique
    reflection_notes: ReflectionNotes
    revision_count: int
    # Set only on a revision pass; the 4 assessment agents fold this into
    # their prompt when present so a second pass is an actual revision,
    # not an identical re-run of the same inputs.
    revision_feedback: Optional[str]

    # Set only on a rebuttal-aware re-review (core/graph/rebuttal.py): the
    # author's written response to the original review. The 4 assessment agents
    # fold it into their prompt so they reconsider their verdict against the
    # rebuttal. Independent of revision_feedback -- a re-review can carry both.
    rebuttal_text: Optional[str]

    # --- Stage 4: synthesis ---
    final_review: FinalReview

    # --- Stage 5: human-in-the-loop approval (gates the final recommendation) ---
    # Set once a human resumes the interrupted run with their decision. A
    # "revised" decision with an override_recommendation also rewrites
    # final_review.final_recommendation (see nodes.human_approval).
    human_approval: HumanApproval
