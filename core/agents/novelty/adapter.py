"""Bridges manik's parsing-layer output to this agent's PeerRead-shaped input.

PaperTextExtractor.extract() expects a dict shaped like PeerRead's raw JSON
(title/abstract/sections: [{heading, text}]/references/year). ParsedPaper is
a pydantic object with a typed List[Section] and List[Reference] -- this is
the one conversion needed to run the agent against a paper this project just
parsed with Docling, instead of only against pre-existing PeerRead JSON files.
"""
from __future__ import annotations

from typing import Any, Dict

from core.schemas.agent_output_schemas import ParsedPaper


def parsed_paper_to_novelty_input(parsed_paper: ParsedPaper) -> Dict[str, Any]:
    """Convert a `ParsedPaper` into the dict shape `PaperTextExtractor.extract()` expects."""
    return {
        "title": parsed_paper.title,
        "abstract": parsed_paper.abstract,
        "sections": [{"heading": s.name, "text": s.text} for s in parsed_paper.sections],
        "references": [r.raw_text for r in parsed_paper.references],
    }
