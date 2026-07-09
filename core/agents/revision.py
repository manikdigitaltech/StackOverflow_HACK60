"""
Shared helper for injecting self-reflection feedback into a revision pass.

Used only by the four assessment agents the graph's bounded revision loop
re-runs (Novelty, Methodology, Citation, Evidence & Reproducibility) --
centralized so all four render the exact same feedback format instead of
drifting into four slightly different phrasings.
"""
from typing import Any, Dict


def revision_feedback_block(inputs: Dict[str, Any]) -> str:
    """Returns a prompt-ready feedback block, or "" on a first pass.

    inputs["revision_feedback"] is only ever set by the graph's revision node
    (core/graph/nodes.py), never by a first pass, so "" here means exactly
    "this is not a revision" -- callers can always pass this straight into
    PromptManager.render() without a None-check.
    """
    feedback = inputs.get("revision_feedback")
    if not feedback:
        return ""
    return (
        "\n\nREVISION NOTE: A prior pass of this review was flagged by the "
        "self-reflection step. Address the following before finalizing your "
        "assessment this time:\n" + feedback
    )
