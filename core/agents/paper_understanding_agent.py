"""
Paper Understanding Agent: reads the token-budgeted paper context and
produces a condensed briefing (summary, stated contributions, key terms)
that downstream agents (Novelty, Methodology, etc.) consume instead of
re-reading the full paper each time.
"""

from typing import Any, Dict
from core.agents.base_agent import BaseAgent, AgentExecutionError
from core.schemas.agent_output_schemas import ParsedPaper, PaperUnderstandingOutput
from core.parsing.context_builder import build_paper_context
from core.llm.structured_output import invoke_for_json, StructuredOutputError
from core.config.settings import settings


class PaperUnderstandingAgent(BaseAgent):
    def run(self, inputs: Dict[str, Any]) -> PaperUnderstandingOutput:
        parsed_paper: ParsedPaper = inputs["parsed_paper"]

        paper_context = build_paper_context(
            parsed_paper, max_tokens=settings.parsing.prompt_token_budget
        )

        system, user = self._prompts.render(
            "paper_understanding_agent", paper_context=paper_context
        )

        try:
            self._log("Calling LLM for paper understanding...")
            result = invoke_for_json(self._llm, system, user, PaperUnderstandingOutput)
            self._log(f"Got summary ({len(result.summary)} chars), "
                      f"{len(result.stated_contributions)} contributions, "
                      f"{len(result.key_terms)} key terms.")
            return result
        except StructuredOutputError as e:
            self._log(f"FAILED: {e}")
            raise AgentExecutionError(str(e)) from e
