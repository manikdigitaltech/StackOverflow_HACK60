"""
Shared helpers for injecting extra context into the four assessment agents the
graph re-runs (Novelty, Methodology, Citation, Evidence & Reproducibility) --
centralized so all four render the exact same block format instead of drifting
into four slightly different phrasings. Two distinct channels:

  - revision_feedback_block: self-reflection's critique on a bounded revision
    pass ("you were flagged, fix it").
  - rebuttal_feedback_block: an author's rebuttal on a re-review pass
    ("the authors responded -- reconsider your verdict").

Both return "" when their trigger key is absent, so callers can pass the
result straight into PromptManager.render() without a None-check.
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


def rebuttal_feedback_block(inputs: Dict[str, Any]) -> str:
    """Returns a prompt-ready author-rebuttal block, or "" when there is none.

    inputs["rebuttal_text"] is set only on a rebuttal-aware re-review (the
    author's written response to the original review; see
    core/graph/rebuttal.py), never on a first-pass review, so "" here means
    exactly "no rebuttal to weigh." Deliberately distinct from the
    self-reflection revision block above: that one says "you were flagged, fix
    it"; this one says "the authors responded -- reconsider your verdict,
    conceding only where the rebuttal genuinely resolves the concern."
    """
    rebuttal = inputs.get("rebuttal_text")
    if not rebuttal:
        return ""
    return (
        "\n\nAUTHOR REBUTTAL: The authors submitted the following rebuttal in "
        "response to the initial review. Reconsider your assessment in light of "
        "it -- revise a verdict only where the rebuttal genuinely resolves the "
        "concern with evidence, and hold your ground where it does not (do not "
        "concede a point merely because the authors pushed back):\n" + rebuttal
    )
