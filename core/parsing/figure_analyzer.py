"""
Populates Figure.image_path (via figure_cropper.crop_figure) and Figure.ocr_text
(via a local Ollama vision model) for a ParsedPaper's figures.

docling_parser.py deliberately stops at bbox/page/caption for figures -- no
pixels, no analysis (see its module docstring, "no pixels here"). This module
is the step that closes that gap: crop each figure to a PNG, then ask a local
VLM to describe it, so downstream agents (e.g. Evidence, Methodology) can
reason about a chart's content in the same text prompt as the rest of the
paper, instead of only seeing a caption string.

Gated by settings.vision.enabled (default False) so the rest of the pipeline
behaves identically with or without a vision model pulled locally.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from core.config.settings import settings
from core.llm.llm_provider import get_vision_llm
from core.llm.prompt_manager import PromptManager
from core.parsing.figure_cropper import crop_figure
from core.schemas.agent_output_schemas import Figure, ParsedPaper

logger = logging.getLogger(__name__)

_CROPS_DIR = Path("data") / "figure_crops"


def analyze_figures(parsed_paper: ParsedPaper) -> ParsedPaper:
    """Crop + VLM-describe up to settings.vision.max_figures_per_paper figures.

    Returns a new ParsedPaper with `image_path`/`ocr_text` populated on the
    figures that were processed. No-op (returns parsed_paper unchanged) when
    settings.vision.enabled is False. Fails soft per-figure: a crop or VLM
    error is logged and that one figure is left as-is -- one bad crop should
    never block the rest of the paper's review.
    """
    if not settings.vision.enabled:
        return parsed_paper

    llm = get_vision_llm()
    prompt_manager = PromptManager()

    updated_figures: list[Figure] = []
    analyzed = 0
    for figure in parsed_paper.figures:
        can_process = figure.bbox is not None and figure.page is not None
        if not can_process or analyzed >= settings.vision.max_figures_per_paper:
            updated_figures.append(figure)
            continue

        try:
            image_path = _crop(parsed_paper.source_pdf_path, figure)
            description = _describe(llm, prompt_manager, image_path, figure.caption)
            updated_figures.append(
                figure.model_copy(update={"image_path": image_path, "ocr_text": description})
            )
            analyzed += 1
        except Exception as exc:
            logger.warning("Vision analysis failed for %s: %s", figure.figure_id, exc)
            updated_figures.append(figure)

    return parsed_paper.model_copy(update={"figures": updated_figures})


def _crop(pdf_path: str, figure: Figure) -> str:
    out_path = _CROPS_DIR / f"{Path(pdf_path).stem}_{figure.figure_id}.png"
    return crop_figure(pdf_path, figure.page, figure.bbox, str(out_path), dpi=settings.vision.crop_dpi)


def _describe(llm, prompt_manager: PromptManager, image_path: str, caption: Optional[str]) -> str:
    system, user = prompt_manager.render("figure_analysis", caption=caption or "(no caption detected)")
    image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")

    messages = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=[
        {"type": "text", "text": user},
        {"type": "image_url", "image_url": f"data:image/png;base64,{image_b64}"},
    ]))

    response = llm.invoke(messages)
    return response.content
