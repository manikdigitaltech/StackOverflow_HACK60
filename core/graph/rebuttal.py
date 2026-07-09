"""
Rebuttal-aware re-review (problem statement section 8, "brownie points": the
agent revises its verdict after a simulated author rebuttal).

A re-review is just a second run of the same compiled review graph, seeded with
the author's `rebuttal_text`. The four assessment agents fold the rebuttal into
their prompts (via core/agents/revision.rebuttal_feedback_block) and reconsider
their verdicts; reflection and final_review then synthesize a revised
recommendation, which -- like any recommendation -- parks at the mandatory
human-approval interrupt before it is issued. Reusing the whole graph (rather
than a bespoke re-run path) means the revised verdict goes through the exact
same grounding, self-reflection, and human-approval discipline as the first.

`compare_recommendations` reports how the verdict moved, which is the actual
point of the exercise: did the rebuttal change the reviewer's mind, and which
way?
"""
from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from core.schemas.agent_output_schemas import ParsedPaper

# Least-to-most favorable, matching FinalReview.final_recommendation's 5-way
# scale -- lets us report not just "changed" but the direction and magnitude.
RATING_ORDER = ["reject", "weak_reject", "borderline", "weak_accept", "accept"]


def compare_recommendations(original: str, revised: str) -> dict:
    """How the recommendation moved between the original review and the
    rebuttal-aware re-review."""
    oi, ri = RATING_ORDER.index(original), RATING_ORDER.index(revised)
    if ri > oi:
        direction = "more_favorable"
    elif ri < oi:
        direction = "less_favorable"
    else:
        direction = "unchanged"
    return {
        "original_recommendation": original,
        "revised_recommendation": revised,
        "changed": original != revised,
        "direction": direction,
        "steps": ri - oi,  # signed: +ve = moved toward accept, -ve = toward reject
    }


def run_rebuttal_rereview(
    parsed_paper: ParsedPaper,
    rebuttal_text: str,
    *,
    graph: Any = None,
    thread_id: Optional[str] = None,
    original_recommendation: Optional[str] = None,
) -> dict:
    """Re-run the full review graph with an author rebuttal folded in.

    Args:
        parsed_paper: the same ParsedPaper the original review ran on.
        rebuttal_text: the author's written response to the original review.
        graph: a compiled review graph (defaults to a fresh build_review_graph()).
            Injectable so tests can pass a mocked-agent graph.
        thread_id: checkpoint thread for this re-review; a fresh id by default so
            it never collides with the original run's parked/finished thread.
        original_recommendation: the first review's final_recommendation; when
            given, the result includes a `comparison` of how the verdict moved.

    Returns a dict with the revised FinalReview (drafted, parked at the
    human-approval gate), the thread_id to resume it on, and the comparison.
    """
    if not (rebuttal_text and rebuttal_text.strip()):
        raise ValueError("rebuttal_text must be a non-empty author rebuttal.")

    if graph is None:
        from core.graph.build_graph import build_review_graph  # local import: avoids a heavy import at module load
        graph = build_review_graph()

    thread_id = thread_id or f"rebuttal-{uuid4().hex[:8]}"
    state = graph.invoke(
        {"parsed_paper": parsed_paper, "rebuttal_text": rebuttal_text},
        config={"configurable": {"thread_id": thread_id}},
    )

    revised = state.get("final_review")
    comparison = None
    if original_recommendation and revised is not None:
        comparison = compare_recommendations(original_recommendation, revised.final_recommendation)

    return {
        "thread_id": thread_id,
        "revised_review": revised,
        # The revised verdict, like the original, is not issued until a human
        # signs off -- resume this thread_id with an approval decision.
        "awaiting_approval": "__interrupt__" in state,
        "comparison": comparison,
        "state": state,
    }
