"""
Deterministic (non-LLM) extraction of in-text figure/table/chart/plot/diagram
references, with surrounding context and resolved target existence. Used by
VisualReferenceAgent so the LLM only has to judge *how* each reference is
used -- not re-derive which references exist and whether their target does,
which is exactly the kind of thing a model can hallucinate about (see
figure_table_agent.py's own consistency check for the same philosophy).

Chart/Plot/Diagram mentions are treated as "figure" references -- the
parsed-paper model only distinguishes figures from tables, so a chart/plot/
diagram in prose is assumed to refer to an extracted figure.
"""

from __future__ import annotations

import re
from typing import Dict, List, NamedTuple, Optional, Tuple

from core.schemas.agent_output_schemas import ParsedPaper

_REF_PATTERN = re.compile(r"\b(Fig(?:ure)?|Table|Chart|Plot|Diagram)s?\.?\s+(\d+)\b", re.IGNORECASE)
_CONTEXT_CHARS = 240   # total window (split before/after the mention)
_MAX_CONTEXTS_PER_TARGET = 4   # bound prompt size -- a paper can say "Figure 2" a dozen times

_KIND_MAP = {"fig": "figure", "figure": "figure", "table": "table",
             "chart": "figure", "plot": "figure", "diagram": "figure"}


class ResolvedReference(NamedTuple):
    mention: str              # the first-seen exact in-text form, e.g. "Figure 3"
    kind: str                  # "figure" or "table"
    number: int
    target_id: Optional[str]  # resolved figure_id/table_id, or None if it doesn't exist
    exists: bool
    contexts: List[str]        # up to _MAX_CONTEXTS_PER_TARGET surrounding-text snippets


def extract_visual_references(parsed_paper: ParsedPaper) -> List[ResolvedReference]:
    """Scans the paper's body prose for "Figure N" / "Table N" (and Chart/
    Plot/Diagram N as figure synonyms), groups repeated mentions of the same
    target together, and resolves each against the figures/tables actually
    extracted -- using the same "figure_N" / "table_N" 1-indexed id convention
    docling_parser.py assigns in extraction order."""
    full_text = "\n".join(s.text for s in parsed_paper.sections)
    figure_ids = {f.figure_id for f in parsed_paper.figures}
    table_ids = {t.table_id for t in parsed_paper.tables}

    grouped: Dict[Tuple[str, int], Dict] = {}
    for m in _REF_PATTERN.finditer(full_text):
        word, num_str = m.group(1), m.group(2)
        kind = _KIND_MAP[word.lower().rstrip(".")]
        number = int(num_str)
        key = (kind, number)

        start = max(0, m.start() - _CONTEXT_CHARS // 2)
        end = min(len(full_text), m.end() + _CONTEXT_CHARS // 2)
        snippet = " ".join(full_text[start:end].split())

        entry = grouped.setdefault(key, {"mention": m.group(0), "contexts": []})
        if len(entry["contexts"]) < _MAX_CONTEXTS_PER_TARGET:
            entry["contexts"].append(snippet)

    resolved: List[ResolvedReference] = []
    for (kind, number), data in sorted(grouped.items()):
        target_id = f"{kind}_{number}"
        exists = target_id in (figure_ids if kind == "figure" else table_ids)
        resolved.append(ResolvedReference(
            mention=data["mention"],
            kind=kind,
            number=number,
            target_id=target_id if exists else None,
            exists=exists,
            contexts=data["contexts"],
        ))
    return resolved
