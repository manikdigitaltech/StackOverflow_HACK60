"""Bridges manik's parsing-layer output to kanishka's RAG-layer input.

`core.schemas.agent_output_schemas.ParsedPaper.sections` is a `List[Section]`
of pydantic objects (each with `.name`/`.text`). `chunk_paper()` in
`core.rag.chunking.section_chunker` wants a `{section_name: text}` mapping (or
an iterable of `(name, text)` pairs) - passing a `ParsedPaper` straight through
breaks, since iterating its `.sections` list yields `Section` instances, not
`(name, text)` tuples. This is the one conversion needed to wire the two
halves of the merged pipeline together.
"""
from __future__ import annotations

from core.schemas.agent_output_schemas import ParsedPaper

# Chunk metadata, not review content - References get their own structured
# handling in Citation Agent / build_reference_summary(), not prose chunking.
_EXCLUDED_SECTIONS = {"References", "Acknowledgements"}


def parsed_paper_to_chunker_input(parsed_paper: ParsedPaper) -> dict[str, dict[str, str]]:
    """Convert a `ParsedPaper` into the `{"sections": {name: text}}` shape `chunk_paper()` expects.

    Args:
        parsed_paper: output of `core.parsing.docling_parser`.

    Returns:
        A dict with a top-level `"sections"` key (chunk_paper's `dict` branch
        does `parsed_paper.get("sections", {})`, so the mapping must be nested,
        not passed as the top-level dict itself). Includes an `"abstract"`
        entry; chunk_paper's own `normalize_section_name` maps arbitrary
        headings to its canonical set, so manik's canonical names like
        "Method"/"Related Work" pass through correctly without renaming here.
    """
    sections: dict[str, str] = {"abstract": parsed_paper.abstract}
    for section in parsed_paper.sections:
        if section.name in _EXCLUDED_SECTIONS or not section.text.strip():
            continue
        sections[section.name] = section.text
    return {"sections": sections}
