"""
Final Review Generator: the last of the 11 agents. Synthesizes ALL prior
assessments (Paper Understanding, Figure & Table, Novelty, Methodology,
Citation, Reference Usage, Evidence & Reproducibility, Reflection) into the
structured review format the whole project was originally scoped to
produce: Paper Summary, Strengths, Weaknesses, Questions for Authors,
Novelty Analysis, Citation Quality, Reference Usage Quality,
Reproducibility, Evidence Mapping, Missing Baselines, Final Recommendation,
Confidence.

Deliberately does NOT re-read the paper's own text -- all paper-grounding
work already happened in the upstream agents; this agent only synthesizes
from their outputs, which keeps this prompt smaller than Reflection's
despite covering more sources (8 vs. 4).

"missing_baselines" is set deterministically from MethodologyAssessment's
own grounded list, not re-derived by the LLM -- avoids any risk of drift
between what Methodology actually found and what Final Review reports.
"""

from typing import Any, Dict
from core.agents.base_agent import BaseAgent, AgentExecutionError
from core.schemas.agent_output_schemas import (
    PaperUnderstandingOutput, FigureTableSummary, NoveltyAssessment,
    MethodologyAssessment, CitationAssessment, ReferenceUsageAssessment,
    EvidenceReproducibilityAssessment, ReflectionNotes, FinalReview,
)
from core.agents.assessment_formatters import (
    format_understanding, format_figure_table, format_novelty,
    format_methodology, format_citation, format_reference_usage,
    format_evidence_repro, format_reflection,
)
from core.llm.structured_output import invoke_for_json, StructuredOutputError


class FinalReviewAgent(BaseAgent):
    def run(self, inputs: Dict[str, Any]) -> FinalReview:
        understanding: PaperUnderstandingOutput = inputs["paper_understanding"]
        figure_table: FigureTableSummary = inputs["figure_table_summary"]
        novelty: NoveltyAssessment = inputs["novelty_assessment"]
        methodology: MethodologyAssessment = inputs["methodology_assessment"]
        citation: CitationAssessment = inputs["citation_assessment"]
        reference_usage: ReferenceUsageAssessment = inputs["reference_usage_assessment"]
        evidence: EvidenceReproducibilityAssessment = inputs["evidence_assessment"]
        reflection: ReflectionNotes = inputs["reflection_notes"]

        system, user = self._prompts.render(
            "final_review_agent",
            understanding_summary=format_understanding(understanding),
            figure_table_summary=format_figure_table(figure_table),
            novelty_summary=format_novelty(novelty),
            methodology_summary=format_methodology(methodology),
            citation_summary=format_citation(citation),
            reference_usage_summary=format_reference_usage(reference_usage),
            evidence_summary=format_evidence_repro(evidence),
            reflection_summary=format_reflection(reflection),
        )

        try:
            self._log("Calling LLM for final review synthesis...")
            result = invoke_for_json(self._llm, system, user, FinalReview)
            # Deterministic override -- avoids any drift from Methodology's own grounded list.
            result.missing_baselines = methodology.missing_baselines
            self._log(f"Final recommendation: {result.final_recommendation}, "
                      f"confidence: {result.confidence}")
            return result
        except StructuredOutputError as e:
            self._log(f"FAILED: {e}")
            raise AgentExecutionError(str(e)) from e
