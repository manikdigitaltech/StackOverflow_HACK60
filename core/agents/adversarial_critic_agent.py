"""
Adversarial Critic Agent: a sharper, targeted attacker of the Methodology,
Citation, and Evidence & Reproducibility agents' own verdicts -- NOT a
general-purpose second opinion like Reflection, and deliberately excludes
Novelty (out of scope by design).

Where Reflection asks a fairly generic "is this verdict well-supported?"
across all four assessments, this agent's ONLY job is to find the single
weakest point in each of the three target assessments and construct a real
counter-argument against it -- forcing each attacked verdict to be defended
or revised rather than left as a comfortable, uncontested "fair"/"adequate"
rating. Its output does not feed the revision loop directly; it feeds
Reflection (see reflection_agent.py), which weighs the attacks alongside its
own independent checks when deciding needs_revision/severity. This keeps the
existing conditional-edge revision-loop topology unchanged -- only
Reflection's inputs grow by one.

Runs in parallel with Reflection off the same three upstream assessments
(see build_graph.py's AND-join comments for why this is a genuine,
non-racy dependency, and why Reflection's own join grows to include this
agent's output rather than reading it out-of-band).
"""

from typing import Any, Dict
from core.agents.base_agent import BaseAgent, AgentExecutionError
from core.agents.assessment_formatters import (
    format_methodology, format_citation, format_evidence_repro,
)
from core.schemas.agent_output_schemas import (
    ParsedPaper, MethodologyAssessment, CitationAssessment,
    EvidenceReproducibilityAssessment, AdversarialCritique,
)
from core.parsing.context_builder import build_paper_context
from core.llm.structured_output import invoke_for_json, StructuredOutputError
from core.config.settings import settings


class AdversarialCriticAgent(BaseAgent):
    def run(self, inputs: Dict[str, Any]) -> AdversarialCritique:
        parsed_paper: ParsedPaper = inputs["parsed_paper"]
        methodology: MethodologyAssessment = inputs["methodology_assessment"]
        citation: CitationAssessment = inputs["citation_assessment"]
        evidence: EvidenceReproducibilityAssessment = inputs["evidence_assessment"]

        # Same reduced share Reflection uses -- three assessment summaries
        # are being added on top of the paper context in this same prompt.
        paper_context = build_paper_context(
            parsed_paper, max_tokens=int(settings.parsing.prompt_token_budget * 0.5)
        )

        system, user = self._prompts.render(
            "adversarial_critic_agent",
            paper_context=paper_context,
            methodology_summary=format_methodology(methodology),
            citation_summary=format_citation(citation),
            evidence_summary=format_evidence_repro(evidence),
        )

        try:
            self._log("Calling LLM for adversarial critique...")
            result = invoke_for_json(self._llm, system, user, AdversarialCritique)
            self._log(f"{len(result.attacks)} attack(s) raised, "
                      f"weakest_agent={result.weakest_agent}")
            return result
        except StructuredOutputError as e:
            self._log(f"FAILED: {e}")
            raise AgentExecutionError(str(e)) from e
