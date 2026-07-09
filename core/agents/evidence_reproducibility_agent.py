"""
Evidence & Reproducibility Agent: checks (1) whether the paper's headline
quantitative claims are actually backed up by numbers in its own tables,
and (2) whether someone could realistically reproduce the work from what's
stated. The first agent that needs TABLE data, not just section text --
build_paper_context() alone doesn't include tables, so this agent adds
its own table-formatting on top, same pattern as Novelty/Citation's local
context helpers.
"""

from typing import Any, Dict
from core.agents.base_agent import BaseAgent, AgentExecutionError
from core.agents.revision import revision_feedback_block
from core.schemas.agent_output_schemas import ParsedPaper, EvidenceReproducibilityAssessment
from core.parsing.context_builder import build_paper_context
from core.llm.structured_output import invoke_for_json, StructuredOutputError
from core.config.settings import settings

_MAX_CHARS_PER_TABLE = 600


def _format_tables(parsed_paper: ParsedPaper) -> str:
    if not parsed_paper.tables:
        return "No tables were extracted from this paper."
    parts = []
    for t in parsed_paper.tables:
        caption = t.caption or "(no caption)"
        parts.append(f"[{t.table_id}] {caption}\n{t.markdown[:_MAX_CHARS_PER_TABLE]}")
    return "\n\n".join(parts)


class EvidenceReproducibilityAgent(BaseAgent):
    def run(self, inputs: Dict[str, Any]) -> EvidenceReproducibilityAssessment:
        parsed_paper: ParsedPaper = inputs["parsed_paper"]

        # Leave room for the tables block on top of the usual section budget,
        # rather than letting section text alone consume the entire budget.
        paper_context = build_paper_context(
            parsed_paper, max_tokens=int(settings.parsing.prompt_token_budget * 0.7)
        )
        tables_text = _format_tables(parsed_paper)

        system, user = self._prompts.render(
            "evidence_reproducibility_agent",
            paper_context=paper_context,
            tables=tables_text,
            revision_feedback=revision_feedback_block(inputs),
        )

        try:
            self._log("Calling LLM for evidence & reproducibility assessment...")
            result = invoke_for_json(self._llm, system, user, EvidenceReproducibilityAssessment)
            self._log(f"Overall rating: {result.overall_rating}, "
                      f"{len(result.claim_verdicts)} claims checked, "
                      f"{len(result.reproducibility_verdicts)} reproducibility aspects checked.")
            return result
        except StructuredOutputError as e:
            self._log(f"FAILED: {e}")
            raise AgentExecutionError(str(e)) from e
