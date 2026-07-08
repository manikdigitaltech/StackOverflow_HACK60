"""
Builds a token-bounded text representation of a ParsedPaper for injection
into an agent's prompt -- regardless of how long the source paper is.

The core idea: not all sections deserve equal space. Method/Experiments/
Results carry the most review-relevant signal; Related Work and Preamble
carry less; References are handled as structured data elsewhere (the
Citation Agent checks claims against the Reference list directly -- it
doesn't need the reference list's prose dumped into a prompt).

This is what makes the pipeline scale from a 9-page conference paper to
a 60-page thesis chapter without ever silently overflowing the LLM's
context window.
"""

from typing import Dict, List
from core.schemas.agent_output_schemas import ParsedPaper
from core.utils.token_budget import estimate_tokens, truncate_to_token_budget

# Relative importance weight per canonical section name. Sections not listed
# here (e.g. an unrecognized raw heading) default to WEIGHT_DEFAULT.
SECTION_PRIORITY_WEIGHTS: Dict[str, float] = {
    "Abstract": 1.0,
    "Method": 1.0,
    "Experiments": 1.0,
    "Results": 1.0,
    "Ablation Study": 0.8,
    "Introduction": 0.6,
    "Related Work": 0.5,
    "Background": 0.5,
    "Discussion": 0.6,
    "Limitations": 0.6,
    "Conclusion": 0.5,
    "Future Work": 0.3,
    "Preamble": 0.1,
    "Appendix": 0.2,
    "Acknowledgements": 0.0,
    "References": 0.0,   # deliberately excluded from prose context -- see module docstring
}
WEIGHT_DEFAULT = 0.4


def build_paper_context(parsed_paper: ParsedPaper, max_tokens: int) -> str:
    """
    Returns a single string combining title + abstract + section text,
    truncated per-section according to priority weight, guaranteed to stay
    within max_tokens (approximately -- see token_budget.py's estimate).
    """
    sections = [s for s in parsed_paper.sections if s.name != "References"]

    # Abstract is always included in full (and is small) regardless of budget math.
    abstract_tokens = estimate_tokens(parsed_paper.abstract)
    remaining_budget = max(0, max_tokens - abstract_tokens - 100)  # ~100 tokens reserved for title/headers

    total_weight = sum(SECTION_PRIORITY_WEIGHTS.get(s.name, WEIGHT_DEFAULT) for s in sections) or 1.0

    parts: List[str] = [f"TITLE: {parsed_paper.title}", f"\nABSTRACT:\n{parsed_paper.abstract}"]

    for section in sections:
        weight = SECTION_PRIORITY_WEIGHTS.get(section.name, WEIGHT_DEFAULT)
        if weight <= 0:
            continue
        section_budget = int(remaining_budget * (weight / total_weight))
        if section_budget < 20:  # not worth including a sliver
            continue
        truncated = truncate_to_token_budget(section.text, section_budget, keep="head_and_tail")
        parts.append(f"\n{section.name.upper()}:\n{truncated}")

    return "\n".join(parts)


def build_reference_summary(parsed_paper: ParsedPaper, max_references: int = 60) -> str:
    """
    Structured (not prose) reference summary for agents that check citation
    presence/quality rather than read the bibliography narratively.
    """
    refs = parsed_paper.references[:max_references]
    lines = [f"- {r.raw_text[:150]}" for r in refs]
    suffix = ""
    if len(parsed_paper.references) > max_references:
        suffix = f"\n... and {len(parsed_paper.references) - max_references} more references not shown."
    return f"REFERENCE LIST ({len(parsed_paper.references)} total):\n" + "\n".join(lines) + suffix
