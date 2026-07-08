"""
Groups Docling's flat, reading-order text-item stream into named Sections,
and identifies the paper's title and abstract along the way.

Docling doesn't hand you "the abstract" as a labeled field — it hands you a
stream of (label, text) pairs in reading order. This module is the glue that
turns that stream into something section-shaped.
"""

from typing import List, Tuple, Optional
from core.schemas.agent_output_schemas import Section

_CANONICAL_SECTION_KEYWORDS = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "related work": "Related Work",
    "background": "Background",
    "methodology": "Method",
    "method": "Method",
    "approach": "Method",
    "experiment": "Experiments",
    "result": "Results",
    "evaluation": "Results",
    "discussion": "Discussion",
    "conclusion": "Conclusion",
    "limitation": "Limitations",
    "reference": "References",
    "bibliography": "References",
    "appendix": "Appendix",
    "acknowledg": "Acknowledgements",
}


def _canonical_name(heading_text: str) -> str:
    lowered = heading_text.strip().lower()
    for keyword, canonical in _CANONICAL_SECTION_KEYWORDS.items():
        if keyword in lowered:
            return canonical
    return heading_text.strip() or "Untitled Section"


def _is_heading_label(label: str) -> bool:
    label = (label or "").lower()
    return "head" in label or "title" in label or "section" in label


def segment_sections(
    text_items: List[Tuple[str, str, Optional[int]]]
) -> Tuple[List[Section], str, str]:
    """
    text_items: list of (label, text, page_no) tuples, in reading order,
    as produced by docling_parser.py.

    Returns: (sections, title, abstract)
    """
    sections: List[Section] = []
    title = ""
    current_heading = None
    current_raw_heading = None
    current_text_parts: List[str] = []
    current_page_start = None

    def flush():
        if current_heading is not None and current_text_parts:
            sections.append(Section(
                name=current_heading,
                raw_heading=current_raw_heading,
                text="\n".join(current_text_parts).strip(),
                page_start=current_page_start,
            ))

    for label, text, page_no in text_items:
        text = (text or "").strip()
        if not text:
            continue

        if not title and (label or "").lower() == "title":
            title = text
            continue

        if _is_heading_label(label):
            flush()
            current_raw_heading = text
            current_heading = _canonical_name(text)
            current_text_parts = []
            current_page_start = page_no
        else:
            if current_heading is None:
                # Text before any detected heading — keep it, don't drop it silently.
                current_heading = "Preamble"
                current_raw_heading = None
                current_page_start = page_no
            current_text_parts.append(text)

    flush()

    if not title:
        for _label, text, _page in text_items:
            if text and text.strip():
                title = text.strip()
                break

    abstract = ""
    for section in sections:
        if section.name == "Abstract":
            abstract = section.text
            break

    return sections, (title or "Untitled Paper"), abstract
