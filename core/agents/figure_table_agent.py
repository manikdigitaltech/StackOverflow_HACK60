"""
Figure & Table Agent: summarizes what each figure/table shows and whether
its caption is self-contained, using caption text + table markdown data --
NOT actual visual interpretation (that's the Vision-Optional extension
from the blueprint, which is unbuilt; this agent runs in caption-mode
only, matching VISION__ENABLED=false).

Also runs a deterministic (non-LLM) consistency check: does the paper's
prose reference more figures/tables than we actually extracted? A cheap,
reliable way to catch a parsing gap before ever blaming the model for
missing content it was never given in the first place.
"""

import re
from typing import Any, Dict
from core.agents.base_agent import BaseAgent, AgentExecutionError
from core.schemas.agent_output_schemas import ParsedPaper, FigureTableSummary
from core.llm.structured_output import invoke_for_json, StructuredOutputError

_FIGURE_REF_PATTERN = re.compile(r"\bFigure\s+(\d+)\b", re.IGNORECASE)
_TABLE_REF_PATTERN = re.compile(r"\bTable\s+(\d+)\b", re.IGNORECASE)
_MAX_CHARS_PER_TABLE = 500


def _check_extraction_consistency(parsed_paper: ParsedPaper) -> str:
    full_text = "\n".join(s.text for s in parsed_paper.sections)
    fig_refs = [int(m) for m in _FIGURE_REF_PATTERN.findall(full_text)]
    table_refs = [int(m) for m in _TABLE_REF_PATTERN.findall(full_text)]

    notes = []
    max_fig_ref = max(fig_refs) if fig_refs else 0
    if max_fig_ref > len(parsed_paper.figures):
        notes.append(f"Paper text references up to Figure {max_fig_ref}, but only "
                      f"{len(parsed_paper.figures)} figure(s) were extracted -- possible parsing gap.")
    max_table_ref = max(table_refs) if table_refs else 0
    if max_table_ref > len(parsed_paper.tables):
        notes.append(f"Paper text references up to Table {max_table_ref}, but only "
                      f"{len(parsed_paper.tables)} table(s) were extracted -- possible parsing gap.")

    return " ".join(notes) if notes else "No figure/table reference mismatches detected."


def _format_figures(parsed_paper: ParsedPaper) -> str:
    if not parsed_paper.figures:
        return "No figures were extracted from this paper."
    return "\n".join(
        f"[{f.figure_id}] Caption: {f.caption or '(no caption)'}" for f in parsed_paper.figures
    )


def _format_tables(parsed_paper: ParsedPaper) -> str:
    if not parsed_paper.tables:
        return "No tables were extracted from this paper."
    parts = []
    for t in parsed_paper.tables:
        parts.append(f"[{t.table_id}] Caption: {t.caption or '(no caption)'}\n{t.markdown[:_MAX_CHARS_PER_TABLE]}")
    return "\n\n".join(parts)


class FigureTableAgent(BaseAgent):
    def run(self, inputs: Dict[str, Any]) -> FigureTableSummary:
        parsed_paper: ParsedPaper = inputs["parsed_paper"]

        if not parsed_paper.figures and not parsed_paper.tables:
            self._log("No figures or tables extracted -- skipping the LLM call entirely.")
            return FigureTableSummary(
                figure_summaries=[], table_summaries=[],
                extraction_consistency_note="No figures or tables were extracted from this paper.",
            )

        figures_text = _format_figures(parsed_paper)
        tables_text = _format_tables(parsed_paper)
        consistency_note = _check_extraction_consistency(parsed_paper)

        system, user = self._prompts.render(
            "figure_table_agent", figures=figures_text, tables=tables_text,
        )

        try:
            self._log("Calling LLM for figure & table summary...")
            result = invoke_for_json(self._llm, system, user, FigureTableSummary)
            # Set deterministically -- this doesn't need (or get) LLM judgment.
            result.extraction_consistency_note = consistency_note
            self._log(f"{len(result.figure_summaries)} figure(s), "
                      f"{len(result.table_summaries)} table(s) summarized.")
            return result
        except StructuredOutputError as e:
            self._log(f"FAILED: {e}")
            raise AgentExecutionError(str(e)) from e
