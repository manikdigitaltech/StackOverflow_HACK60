"""
Reflection Agent: the self-critique/verifier step. Reads all four prior
assessments (Novelty, Methodology, Citation, Evidence & Reproducibility)
plus the original paper, and flags anything speculative, unsupported, or
inconsistent -- the safety net for exactly the kind of issue we've been
catching by hand throughout Day 2 (e.g. Novelty's ungrounded "might be
implied" CUDA-overlap claim).

This agent does NOT re-run retrieval or re-parse the paper -- it only
reviews what the other four agents already concluded.
"""

from typing import Any, Dict, Optional
from core.agents.base_agent import BaseAgent, AgentExecutionError
from core.agents.assessment_formatters import format_adversarial_critique
from core.schemas.agent_output_schemas import (
    ParsedPaper, NoveltyAssessment, MethodologyAssessment,
    CitationAssessment, EvidenceReproducibilityAssessment, ReflectionNotes,
    AdversarialCritique,
)
from core.parsing.context_builder import build_paper_context
from core.llm.structured_output import invoke_for_json, StructuredOutputError
from core.config.settings import settings


def _format_novelty(a: NoveltyAssessment) -> str:
    lines = [f"Novelty rating: {a.novelty_rating}"]
    for v in a.contribution_verdicts:
        lines.append(f"  - [{v.verdict}] {v.contribution}: {v.note}")
    for o in a.overlapping_work:
        lines.append(f'  - Overlap cited: "{o.compared_paper_title}": {o.similarity_note}')
    lines.append(f"Reasoning: {a.reasoning}")
    return "\n".join(lines)


def _format_methodology(a: MethodologyAssessment) -> str:
    lines = [f"Soundness rating: {a.soundness_rating}"]
    for v in a.aspect_verdicts:
        lines.append(f"  - [{v.assessment}] {v.aspect}: {v.note}")
    if a.missing_baselines:
        lines.append(f"Missing baselines: {', '.join(a.missing_baselines)}")
    lines.append(f"Reasoning: {a.reasoning}")
    return "\n".join(lines)


def _format_citation(a: CitationAssessment) -> str:
    lines = [f"Citation quality: {a.citation_quality_rating}"]
    for v in a.coverage_verdicts:
        status = "cited" if v.cited else "NOT CITED"
        lines.append(f"  - [{status}] {v.related_paper_title}: {v.note}")
    lines.append(f"Reasoning: {a.reasoning}")
    return "\n".join(lines)


def _format_evidence_repro(a: EvidenceReproducibilityAssessment) -> str:
    lines = [f"Overall rating: {a.overall_rating}"]
    for c in a.claim_verdicts:
        lines.append(f"  - [{c.verdict}] Claim: {c.claim} -- {c.note}")
    for v in a.reproducibility_verdicts:
        lines.append(f"  - [{v.assessment}] {v.aspect}: {v.note}")
    lines.append(f"Reasoning: {a.reasoning}")
    return "\n".join(lines)


class ReflectionAgent(BaseAgent):
    def run(self, inputs: Dict[str, Any]) -> ReflectionNotes:
        parsed_paper: ParsedPaper = inputs["parsed_paper"]
        novelty: NoveltyAssessment = inputs["novelty_assessment"]
        methodology: MethodologyAssessment = inputs["methodology_assessment"]
        citation: CitationAssessment = inputs["citation_assessment"]
        evidence: EvidenceReproducibilityAssessment = inputs["evidence_assessment"]
        # Optional (.get, not []) so this agent stays callable from older
        # callers/tests that don't yet supply it -- the graph always
        # supplies a real one via its AND-join (see build_graph.py).
        critique: Optional[AdversarialCritique] = inputs.get("adversarial_critique")

        # Smaller share than standalone agents get -- four assessment
        # summaries (plus the adversarial critique) are being added on top
        # of the paper context in this same prompt, so the paper text
        # itself gets less room.
        paper_context = build_paper_context(
            parsed_paper, max_tokens=int(settings.parsing.prompt_token_budget * 0.5)
        )

        system, user = self._prompts.render(
            "reflection_agent",
            paper_context=paper_context,
            novelty_summary=_format_novelty(novelty),
            methodology_summary=_format_methodology(methodology),
            citation_summary=_format_citation(citation),
            evidence_summary=_format_evidence_repro(evidence),
            adversarial_critique_summary=(
                format_adversarial_critique(critique) if critique is not None
                else "(no adversarial critique available for this pass)"
            ),
        )

        try:
            self._log("Calling LLM for reflection/self-critique...")
            result = invoke_for_json(self._llm, system, user, ReflectionNotes)
            self._log(f"Confidence: {result.overall_confidence}, "
                      f"{len(result.flags)} flag(s), needs_revision={result.needs_revision}")
            return result
        except StructuredOutputError as e:
            self._log(f"FAILED: {e}")
            raise AgentExecutionError(str(e)) from e
