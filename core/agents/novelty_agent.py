"""
Novelty Agent: judges what's genuinely novel about the paper versus what
overlaps with retrieved literature. The first agent that combines two
prior agents' outputs (Paper Understanding + Literature RAG), and the
first where grounding actually matters -- a hallucinated "this overlaps
with paper X" citation would be a real, visible failure, not just a
rough edge.
"""

from typing import Any, Dict
from core.agents.base_agent import BaseAgent, AgentExecutionError
from core.agents.revision import revision_feedback_block
from core.schemas.agent_output_schemas import PaperUnderstandingOutput, LiteratureContext, NoveltyAssessment
from core.llm.structured_output import invoke_for_json, StructuredOutputError


def _format_literature_context(context: LiteratureContext, max_chars_per_chunk: int = 400) -> str:
    if not context.matches:
        return "No related literature was retrieved."
    lines = []
    for m in context.matches:
        lines.append(f'- "{m.title}" ({m.year}): {m.chunk_text[:max_chars_per_chunk]}')
    return "\n".join(lines)


class NoveltyAgent(BaseAgent):
    def run(self, inputs: Dict[str, Any]) -> NoveltyAssessment:
        understanding: PaperUnderstandingOutput = inputs["paper_understanding"]
        literature_context: LiteratureContext = inputs["literature_context"]

        contributions_text = "\n".join(
            f"{i + 1}. {c}" for i, c in enumerate(understanding.stated_contributions)
        )
        literature_text = _format_literature_context(literature_context)

        system, user = self._prompts.render(
            "novelty_agent",
            summary=understanding.summary,
            contributions=contributions_text,
            literature_matches=literature_text,
            revision_feedback=revision_feedback_block(inputs),
        )

        try:
            self._log("Calling LLM for novelty assessment...")
            result = invoke_for_json(self._llm, system, user, NoveltyAssessment)
            self._log(f"Novelty rating: {result.novelty_rating}, "
                      f"{len(result.contribution_verdicts)} contribution verdicts, "
                      f"{len(result.overlapping_work)} overlaps cited.")
            return result
        except StructuredOutputError as e:
            self._log(f"FAILED: {e}")
            raise AgentExecutionError(str(e)) from e