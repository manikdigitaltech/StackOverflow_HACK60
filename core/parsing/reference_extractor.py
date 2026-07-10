"""
Best-effort extraction of a structured reference list from the paper's
References/Bibliography section. This deliberately doesn't try to parse
full citation metadata (author lists, venues, etc.) - just enough
(raw text + year, when detectable) to support the Citation Agent later.
"""

import re
from typing import List, Tuple, Optional
from core.schemas.agent_output_schemas import Reference

_YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
# Matches common numbered reference-list styles: "[1] ...", "1. ...", "1) ..."
_REF_SPLIT_PATTERN = re.compile(r"(?:^|\n)\s*(?:\[\d+\]|\d+\.|\d+\))\s+")


def extract_references(text_items: List[Tuple[str, str, Optional[int]]]) -> List[Reference]:
    ref_text_parts = []
    in_references = False

    for label, text, _page in text_items:
        text = (text or "").strip()
        lowered = text.lower()
        is_heading = "head" in (label or "").lower() or "title" in (label or "").lower() or "section" in (label or "").lower()

        if is_heading and ("reference" in lowered or "bibliography" in lowered):
            in_references = True
            continue
        if is_heading and in_references:
            # Hit a new heading after References started (e.g. Appendix) - stop collecting.
            break
        if in_references and text:
            ref_text_parts.append(text)

    full_text = "\n".join(ref_text_parts).strip()
    if not full_text:
        return []

    raw_entries = [e.strip() for e in _REF_SPLIT_PATTERN.split(full_text) if e.strip()]
    if len(raw_entries) <= 1:
        # No numbered markers found - fall back to one entry per line.
        raw_entries = [e.strip() for e in full_text.split("\n") if e.strip()]

    references = []
    for entry in raw_entries:
        year_match = _YEAR_PATTERN.search(entry)
        year = int(year_match.group()) if year_match else None
        references.append(Reference(raw_text=entry, year=year))
    return references
