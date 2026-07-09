"""
Visual Reference Agent: verifies how the paper's prose actually USES its
figures/tables/charts/plots/diagrams -- the piece Figure & Table Agent
doesn't cover (that agent only summarizes captions/table data in isolation,
it never looks at the surrounding body text or how many times/where a visual
is invoked). For each figure/table number mentioned in-text, this agent:

  1. deterministically resolves whether that target was actually extracted
     (core/parsing/visual_reference_extractor.py -- existence is a fact, not
     an LLM judgment call, same philosophy as figure_table_agent's own
     deterministic consistency note),
  2. asks the LLM to judge, from the surrounding prose excerpts, whether the
     reference meaningfully explains/supports the point being made and
     classify its purpose (result support / method explanation / comparison
     / ablation / other),
  3. flags weak, misleading, or uncertain usage with grounding evidence.

References whose target doesn't exist at all are never sent to the LLM --
there is no visual for it to judge text about -- and are instead reported as
a deterministic "missing_target" verdict.

One-shot, like Figure & Table Agent and Reference Usage Agent: needs only
parsed_paper, so it isn't re-run on a revision pass.
"""

from typing import Any, Dict, List

from core.agents.base_agent import AgentExecutionError, BaseAgent
from core.llm.structured_output import StructuredOutputError, invoke_for_json
from core.parsing.visual_reference_extractor import ResolvedReference, extract_visual_references
from core.schemas.agent_output_schemas import (
    ParsedPaper,
    VisualReferenceAssessment,
    VisualReferenceVerdict,
)

_MAX_TARGETS_TO_JUDGE = 25   # bound prompt size on reference-heavy papers


def _format_targets(resolved: List[ResolvedReference]) -> str:
    lines = []
    for r in resolved:
        label = f"{r.kind.capitalize()} {r.number}"
        contexts = "\n      ".join(f'- "...{c}..."' for c in r.contexts)
        lines.append(f'[{r.target_id}] referenced in-text as "{label}":\n      {contexts}')
    return "\n".join(lines) if lines else "No in-text figure/table references found."


def _format_extracted(parsed_paper: ParsedPaper) -> str:
    lines = [f"[{f.figure_id}] Caption: {f.caption or '(no caption)'}" for f in parsed_paper.figures]
    lines += [f"[{t.table_id}] Caption: {t.caption or '(no caption)'}" for t in parsed_paper.tables]
    return "\n".join(lines) if lines else "No figures or tables were extracted."


def _missing_verdicts(missing: List[ResolvedReference]) -> List[VisualReferenceVerdict]:
    return [
        VisualReferenceVerdict(
            mention=r.mention, target_id=None, exists=False,
            purpose="other", verdict="missing_target",
            evidence=r.contexts[0] if r.contexts else "",
            note=(
                f"The paper's prose references {r.mention}, but no matching "
                f"{r.kind} was extracted from the document -- either the "
                f"paper mislabels it or extraction missed it."
            ),
        )
        for r in missing
    ]


class VisualReferenceAgent(BaseAgent):
    def run(self, inputs: Dict[str, Any]) -> VisualReferenceAssessment:
        parsed_paper: ParsedPaper = inputs["parsed_paper"]

        resolved = extract_visual_references(parsed_paper)
        if not resolved:
            self._log("No in-text figure/table references found -- skipping the LLM call entirely.")
            return VisualReferenceAssessment(
                reference_verdicts=[], unresolved_mentions=[],
                overall_quality="fair",
                summary="The paper's prose does not reference any figures or tables by number.",
            )

        existing = [r for r in resolved if r.exists][:_MAX_TARGETS_TO_JUDGE]
        missing = [r for r in resolved if not r.exists]
        missing_verdicts = _missing_verdicts(missing)

        if not existing:
            self._log(f"All {len(missing)} reference(s) point to non-existent targets -- skipping the LLM call.")
            return VisualReferenceAssessment(
                reference_verdicts=missing_verdicts,
                unresolved_mentions=[r.mention for r in missing],
                overall_quality="poor",
                summary="Every in-text figure/table reference points to a target that was never extracted.",
            )

        system, user = self._prompts.render(
            "visual_reference_agent",
            targets=_format_targets(existing),
            extracted_visuals=_format_extracted(parsed_paper),
        )

        try:
            self._log("Calling LLM for visual reference verification...")
            result = invoke_for_json(self._llm, system, user, VisualReferenceAssessment)
            # Deterministic additions the LLM never sees/produces -- keeps
            # existence-checking honest and immune to hallucination.
            result.reference_verdicts = result.reference_verdicts + missing_verdicts
            result.unresolved_mentions = [r.mention for r in missing]
            if missing_verdicts and result.overall_quality in ("good", "excellent"):
                result.overall_quality = "fair"  # missing targets cap the rating regardless of LLM optimism
            self._log(f"{len(result.reference_verdicts)} reference verdict(s), "
                      f"{len(missing_verdicts)} missing target(s).")
            return result
        except StructuredOutputError as e:
            self._log(f"FAILED: {e}")
            raise AgentExecutionError(str(e)) from e
