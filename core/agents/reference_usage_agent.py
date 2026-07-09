"""
Reference Usage Agent: evaluates how effectively the paper uses the
references already in its OWN bibliography -- the inverse of Citation
Agent, which checks whether EXTERNAL literature is missing from that same
bibliography. This agent instead takes each entry already in the reference
list and checks whether the paper's body actually engages with it, how
(background, related work, baseline, method, dataset, tool, or claim
support), and how useful that engagement is -- grounded in the body text,
not citation formatting, and it never flags references as missing.

One-shot, like Figure & Table Agent: needs only parsed_paper, so it isn't
re-run on a revision pass.
"""

from typing import Any, Dict
from core.agents.base_agent import BaseAgent, AgentExecutionError
from core.schemas.agent_output_schemas import ParsedPaper, ReferenceUsageAssessment
from core.parsing.context_builder import build_paper_context, build_reference_summary
from core.llm.structured_output import invoke_for_json, StructuredOutputError
from core.config.settings import settings

_MAX_REFERENCES = 60   # matches build_reference_summary's own default cap


class ReferenceUsageAgent(BaseAgent):
    def run(self, inputs: Dict[str, Any]) -> ReferenceUsageAssessment:
        parsed_paper: ParsedPaper = inputs["parsed_paper"]

        if not parsed_paper.references:
            self._log("No references extracted -- skipping the LLM call entirely.")
            return ReferenceUsageAssessment(
                reference_verdicts=[],
                overall_rating="fair",
                summary=(
                    "No references were extracted from this paper, so reference "
                    "usage could not be assessed."
                ),
            )

        paper_context = build_paper_context(parsed_paper, max_tokens=settings.parsing.prompt_token_budget)
        references_text = build_reference_summary(parsed_paper, max_references=_MAX_REFERENCES)

        system, user = self._prompts.render(
            "reference_usage_agent",
            paper_context=paper_context,
            paper_references=references_text,
            reference_count=min(len(parsed_paper.references), _MAX_REFERENCES),
        )

        try:
            self._log("Calling LLM for reference usage assessment...")
            result = invoke_for_json(self._llm, system, user, ReferenceUsageAssessment)
            not_cited = sum(1 for v in result.reference_verdicts if not v.cited_in_body)
            self._log(f"Overall rating: {result.overall_rating}, "
                      f"{len(result.reference_verdicts)} reference(s) checked, "
                      f"{not_cited} not meaningfully cited.")
            return result
        except StructuredOutputError as e:
            self._log(f"FAILED: {e}")
            raise AgentExecutionError(str(e)) from e
