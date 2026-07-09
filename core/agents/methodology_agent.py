"""
Methodology Agent: evaluates methodological soundness -- baseline
comparisons, ablation coverage, hyperparameter justification, experimental
setup clarity, and statistical rigor. Unlike Novelty/Citation, this agent
needs only the paper itself, no literature comparison, so no grounding
concerns to verify -- just whether its judgments cite real evidence from
the paper.
"""

from typing import Any, Dict
from core.agents.base_agent import BaseAgent, AgentExecutionError
from core.agents.revision import revision_feedback_block
from core.schemas.agent_output_schemas import ParsedPaper, MethodologyAssessment
from core.parsing.context_builder import build_paper_context
from core.llm.structured_output import invoke_for_json, StructuredOutputError
from core.config.settings import settings


class MethodologyAgent(BaseAgent):
    def run(self, inputs: Dict[str, Any]) -> MethodologyAssessment:
        parsed_paper: ParsedPaper = inputs["parsed_paper"]

        paper_context = build_paper_context(
            parsed_paper, max_tokens=settings.parsing.prompt_token_budget
        )

        system, user = self._prompts.render(
            "methodology_agent", paper_context=paper_context,
            revision_feedback=revision_feedback_block(inputs),
        )

        try:
            self._log("Calling LLM for methodology assessment...")
            result = invoke_for_json(self._llm, system, user, MethodologyAssessment)
            self._log(f"Soundness rating: {result.soundness_rating}, "
                      f"{len(result.aspect_verdicts)} aspect verdicts, "
                      f"{len(result.missing_baselines)} missing baselines flagged.")
            return result
        except StructuredOutputError as e:
            self._log(f"FAILED: {e}")
            raise AgentExecutionError(str(e)) from e
