"""
Citation Agent: checks whether retrieved literature that appears highly
relevant to this paper's topic is missing from the paper's OWN reference
list -- a citation-coverage gap check, not a citation-formatting check.

Needs only ParsedPaper + LiteratureContext (matching the original
blueprint's interface) -- deliberately does NOT use build_paper_context(),
to keep the prompt focused and leave token budget for the paper's own
(potentially long) reference list instead.
"""

from typing import Any, Dict
from core.agents.base_agent import BaseAgent, AgentExecutionError
from core.schemas.agent_output_schemas import ParsedPaper, LiteratureContext, CitationAssessment
from core.llm.structured_output import invoke_for_json, StructuredOutputError

_MAX_REFERENCES = 80          # PeerRead/arXiv-style reference lists rarely exceed this
_MAX_CHARS_PER_REFERENCE = 180
_MAX_CHARS_PER_LIT_CHUNK = 400


def _format_literature_context(context: LiteratureContext) -> str:
    if not context.matches:
        return "No related literature was retrieved."
    lines = []
    for m in context.matches:
        lines.append(f'- "{m.title}" ({m.year}): {m.chunk_text[:_MAX_CHARS_PER_LIT_CHUNK]}')
    return "\n".join(lines)


def _format_paper_references(parsed_paper: ParsedPaper) -> str:
    refs = parsed_paper.references[:_MAX_REFERENCES]
    if not refs:
        return "(no references were extracted from this paper)"
    return "\n".join(f"- {r.raw_text[:_MAX_CHARS_PER_REFERENCE]}" for r in refs)


class CitationAgent(BaseAgent):
    def run(self, inputs: Dict[str, Any]) -> CitationAssessment:
        parsed_paper: ParsedPaper = inputs["parsed_paper"]
        literature_context: LiteratureContext = inputs["literature_context"]

        literature_text = _format_literature_context(literature_context)
        references_text = _format_paper_references(parsed_paper)

        system, user = self._prompts.render(
            "citation_agent",
            title=parsed_paper.title,
            abstract=parsed_paper.abstract,
            literature_matches=literature_text,
            literature_count=len(literature_context.matches),
            reference_count=len(parsed_paper.references),
            paper_references=references_text,
        )

        try:
            self._log("Calling LLM for citation assessment...")
            result = invoke_for_json(self._llm, system, user, CitationAssessment)
            not_cited_count = sum(1 for v in result.coverage_verdicts if not v.cited)
            self._log(f"Citation quality: {result.citation_quality_rating}, "
                      f"{len(result.coverage_verdicts)} papers checked, "
                      f"{not_cited_count} not cited.")
            return result
        except StructuredOutputError as e:
            self._log(f"FAILED: {e}")
            raise AgentExecutionError(str(e)) from e
