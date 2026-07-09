"""
Crops each detected formula region to a PNG using the same PyMuPDF-based
crop_figure() figures use. Formula bbox/page come for free from Docling's
base layout model (no extra model needed -- see docling_parser.py); this
module only closes the "no pixels yet" gap, exactly like figure_analyzer.py
does for figures. Recognized LaTeX/plaintext (Formula.text) is filled in
during parsing itself, not here -- it's Docling's do_formula_enrichment
output, gated by settings.formula.enabled.

Fails soft per-formula: a bad crop is logged and that one formula is left
without an image_path -- one bad region should never block the rest of the
paper's review.
"""
from __future__ import annotations

import logging
from pathlib import Path

from core.config.settings import settings
from core.parsing.figure_cropper import crop_figure
from core.schemas.agent_output_schemas import Formula, ParsedPaper

logger = logging.getLogger(__name__)

# Same directory figure_analyzer.py crops into -- server/main.py serves both
# from a single /figure_crops static mount.
_CROPS_DIR = Path("data") / "figure_crops"


def crop_formulas(parsed_paper: ParsedPaper) -> ParsedPaper:
    """Crops up to settings.formula.max_formulas_per_paper formula regions.

    Returns a new ParsedPaper with `image_path` populated on the formulas
    that were successfully cropped. No-op if there are no formulas.
    """
    if not parsed_paper.formulas:
        return parsed_paper

    updated_formulas: list[Formula] = []
    cropped = 0
    for formula in parsed_paper.formulas:
        can_crop = formula.bbox is not None and formula.page is not None
        if not can_crop or cropped >= settings.formula.max_formulas_per_paper:
            updated_formulas.append(formula)
            continue

        try:
            image_path = _crop(parsed_paper.source_pdf_path, formula)
            updated_formulas.append(formula.model_copy(update={"image_path": image_path}))
            cropped += 1
        except Exception as exc:
            logger.warning("Formula crop failed for %s: %s", formula.formula_id, exc)
            updated_formulas.append(formula)

    return parsed_paper.model_copy(update={"formulas": updated_formulas})


def _crop(pdf_path: str, formula: Formula) -> str:
    out_path = _CROPS_DIR / f"{Path(pdf_path).stem}_{formula.formula_id}.png"
    return crop_figure(pdf_path, formula.page, formula.bbox, str(out_path), dpi=settings.formula.crop_dpi)
