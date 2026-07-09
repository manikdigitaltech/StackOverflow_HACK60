"""
Node functions for the review orchestration graph. Each node wraps exactly
one agent's .run() call and returns a partial state update (a dict with only
the key(s) that node is responsible for) -- LangGraph merges these into the
running ReviewGraphState. Parallel nodes never write the same key, which is
what lets them run concurrently without a custom reducer.

Agents are constructed once (in ReviewGraphNodes.__init__), not per-call, so
a review run doesn't pay LLM-client construction cost at every node.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from langgraph.types import interrupt

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
from core.config.settings import settings
from core.graph.state import ReviewGraphState
from core.schemas.agent_output_schemas import HumanApproval

_MAX_FLAGS_IN_FEEDBACK = 8  # bound the feedback block; a wall of flags is worse than a focused one

# Accept common synonyms for a decision so the resume payload can be terse
# ("approve", "accept") or explicit ("approved") without the node caring.
_DECISION_SYNONYMS = {
    "approve": "approved", "approved": "approved", "accept": "approved", "ok": "approved",
    "reject": "rejected", "rejected": "rejected", "deny": "rejected",
    "revise": "revised", "revised": "revised", "override": "revised", "edit": "revised",
}


def _normalize_decision(value: Any) -> str:
    return _DECISION_SYNONYMS.get(str(value).strip().lower(), "approved")


def _coerce_approval(payload: Any) -> HumanApproval:
    """Turn whatever a human/client passed to Command(resume=...) into a
    validated HumanApproval. Tolerates a bare string ("approved"), a dict, or
    an already-built HumanApproval, and stamps decided_at if the caller didn't."""
    if isinstance(payload, HumanApproval):
        approval = payload
    elif isinstance(payload, str):
        approval = HumanApproval(decision=_normalize_decision(payload))
    elif isinstance(payload, dict):
        data = {k: v for k, v in payload.items() if k in HumanApproval.model_fields}
        data["decision"] = _normalize_decision(payload.get("decision", "approved"))
        approval = HumanApproval(**data)
    else:
        approval = HumanApproval(decision="approved")

    if approval.decided_at is None:
        approval = approval.model_copy(update={"decided_at": datetime.now(timezone.utc).isoformat()})
    return approval


class ReviewGraphNodes:
    def __init__(self, llm, prompt_manager):
        self.paper_understanding_agent = PaperUnderstandingAgent(llm, prompt_manager)
        self.literature_rag_agent = LiteratureRAGAgent()
        self.figure_table_agent = FigureTableAgent(llm, prompt_manager)
        self.novelty_agent = NoveltyAgent(llm, prompt_manager)
        self.methodology_agent = MethodologyAgent(llm, prompt_manager)
        self.citation_agent = CitationAgent(llm, prompt_manager)
        self.reference_usage_agent = ReferenceUsageAgent(llm, prompt_manager)
        self.evidence_agent = EvidenceReproducibilityAgent(llm, prompt_manager)
        self.adversarial_critic_agent = AdversarialCriticAgent(llm, prompt_manager)
        self.reflection_agent = ReflectionAgent(llm, prompt_manager)
        self.final_review_agent = FinalReviewAgent(llm, prompt_manager)

    # --- Stage 1: parallel, each only needs parsed_paper ---

    def paper_understanding(self, state: ReviewGraphState) -> dict:
        result = self.paper_understanding_agent.run({"parsed_paper": state["parsed_paper"]})
        return {"paper_understanding": result}

    def literature_rag(self, state: ReviewGraphState) -> dict:
        result = self.literature_rag_agent.run({"parsed_paper": state["parsed_paper"]})
        return {"literature_context": result}

    def figure_table(self, state: ReviewGraphState) -> dict:
        result = self.figure_table_agent.run({"parsed_paper": state["parsed_paper"]})
        return {"figure_table_summary": result}

    def reference_usage(self, state: ReviewGraphState) -> dict:
        """One-shot like figure_table -- checks how the paper uses its OWN
        bibliography (inverse of citation, which checks external literature
        coverage), so it never needs a revision re-run."""
        result = self.reference_usage_agent.run({"parsed_paper": state["parsed_paper"]})
        return {"reference_usage_assessment": result}

    # --- Stage 2: parallel assessments (re-run on a bounded revision pass) ---

    def novelty(self, state: ReviewGraphState) -> dict:
        result = self.novelty_agent.run({
            "paper_understanding": state["paper_understanding"],
            "literature_context": state["literature_context"],
            "revision_feedback": state.get("revision_feedback"),
        })
        return {"novelty_assessment": result}

    def methodology(self, state: ReviewGraphState) -> dict:
        result = self.methodology_agent.run({
            "parsed_paper": state["parsed_paper"],
            "revision_feedback": state.get("revision_feedback"),
        })
        return {"methodology_assessment": result}

    def citation(self, state: ReviewGraphState) -> dict:
        result = self.citation_agent.run({
            "parsed_paper": state["parsed_paper"],
            "literature_context": state["literature_context"],
            "revision_feedback": state.get("revision_feedback"),
        })
        return {"citation_assessment": result}

    def evidence_reproducibility(self, state: ReviewGraphState) -> dict:
        result = self.evidence_agent.run({
            "parsed_paper": state["parsed_paper"],
            "revision_feedback": state.get("revision_feedback"),
        })
        return {"evidence_assessment": result}

    # --- Stage 3: self-reflection + bounded revision loop ---

    def adversarial_critic(self, state: ReviewGraphState) -> dict:
        """Attacks only methodology/citation/evidence's own verdicts (never
        novelty -- out of scope by design). Runs alongside reflection, off
        the same three assessments, and re-runs on a revision pass exactly
        because its AND-join sources (see build_graph.py) are re-triggered
        by prepare_revision -- so it always attacks the CURRENT pass's
        verdicts, not stale ones from before a revision."""
        result = self.adversarial_critic_agent.run({
            "parsed_paper": state["parsed_paper"],
            "methodology_assessment": state["methodology_assessment"],
            "citation_assessment": state["citation_assessment"],
            "evidence_assessment": state["evidence_assessment"],
        })
        return {"adversarial_critique": result}

    def reflection(self, state: ReviewGraphState) -> dict:
        result = self.reflection_agent.run({
            "parsed_paper": state["parsed_paper"],
            "novelty_assessment": state["novelty_assessment"],
            "methodology_assessment": state["methodology_assessment"],
            "citation_assessment": state["citation_assessment"],
            "evidence_assessment": state["evidence_assessment"],
            "adversarial_critique": state["adversarial_critique"],
        })
        return {"reflection_notes": result}

    def route_after_reflection(self, state: ReviewGraphState) -> str:
        """Conditional edge: revise once (bounded by settings.reflection.max_revision_passes)
        if reflection found a major issue, otherwise go straight to synthesis."""
        notes = state["reflection_notes"]
        revision_count = state.get("revision_count", 0)
        if notes.needs_revision and revision_count < settings.reflection.max_revision_passes:
            return "revise"
        return "proceed"

    def ready_for_synthesis(self, state: ReviewGraphState) -> dict:
        """Pure pass-through sync point -- exists only so figure_table's
        one-shot output and reflection's "proceed" decision can be combined
        into a single real AND-join before final_review (see build_graph.py's
        module docstring: add_edge/add_conditional_edges into the same node
        are independent OR-triggers, not a join, unless the sources are
        listed together in one add_edge([...]) call -- and a conditional
        edge's dynamic target can't be one of those list entries directly).
        Runs at most once (nothing loops back to it), so there's no ambiguity
        about whether it needs a "fresh" figure_table update on a later pass."""
        return {}

    def prepare_revision(self, state: ReviewGraphState) -> dict:
        """Builds the feedback text the 4 assessment agents fold into their
        prompt on the re-run, and increments the bounded counter. Without this,
        a 'revision' pass would just be an identical re-run of the same
        inputs -- this is what makes the loop an actual revision."""
        notes = state["reflection_notes"]
        flags = notes.flags[:_MAX_FLAGS_IN_FEEDBACK]
        feedback_lines = [
            f"- [{f.severity}] ({f.source_agent}) {f.flagged_item}: {f.issue}" for f in flags
        ]
        feedback = "\n".join(feedback_lines) or "General concerns were raised; review your assessment for unsupported claims."
        return {
            "revision_feedback": feedback,
            "revision_count": state.get("revision_count", 0) + 1,
        }

    # --- Stage 4: synthesis ---

    def final_review(self, state: ReviewGraphState) -> dict:
        result = self.final_review_agent.run({
            "paper_understanding": state["paper_understanding"],
            "figure_table_summary": state["figure_table_summary"],
            "novelty_assessment": state["novelty_assessment"],
            "methodology_assessment": state["methodology_assessment"],
            "citation_assessment": state["citation_assessment"],
            "reference_usage_assessment": state["reference_usage_assessment"],
            "evidence_assessment": state["evidence_assessment"],
            "reflection_notes": state["reflection_notes"],
        })
        return {"final_review": result}

    # --- Stage 5: human-in-the-loop approval (mandatory gate) ---

    def human_approval(self, state: ReviewGraphState) -> dict:
        """Pause the run and require a human to sign off before the final
        recommendation is issued (problem statement section 11: "Final decision
        requires human-in-the-loop approval").

        interrupt() halts execution here and surfaces the drafted review to the
        caller; the run stays parked (on the checkpointer, keyed by thread_id)
        until it's resumed with `Command(resume=<decision>)`, at which point
        this node re-runs from the top and interrupt() returns that decision.
        Everything above the interrupt() call is a pure read of already-computed
        state, so re-running on resume is harmless.

        A "revised" decision carrying an override_recommendation replaces the
        model's final_recommendation with the human's -- the human, not the
        model, has the last word on the decision.
        """
        draft = state["final_review"]

        decision_payload = interrupt({
            "type": "approval_request",
            "message": "Human approval required before issuing the final recommendation.",
            "draft_recommendation": draft.final_recommendation,
            "draft_confidence": draft.confidence,
            "paper_summary": draft.paper_summary,
            "strengths": draft.strengths,
            "weaknesses": draft.weaknesses,
            "questions_for_authors": draft.questions_for_authors,
        })

        approval = _coerce_approval(decision_payload)
        updates: dict = {"human_approval": approval}

        if approval.decision == "revised" and approval.override_recommendation:
            updates["final_review"] = draft.model_copy(
                update={"final_recommendation": approval.override_recommendation}
            )

        return updates
