"""
Crops a figure's bounding box to a high-resolution PNG using PyMuPDF, given
the page number and bbox that Docling detected.

Coordinate systems differ between the two libraries: Docling's bbox is
typically in the PDF's native coordinate space (origin bottom-left, y
increasing upward), while PyMuPDF's page rect uses origin top-left, y
increasing downward. This module converts between the two.

If crops come out upside-down, empty, or offset from the actual figure,
this conversion is the first place to check - print the raw bbox and
page.rect to compare against what actually got cropped.
"""

from pathlib import Path
from typing import List
import fitz  # PyMuPDF


def crop_figure(pdf_path: str, page_no: int, bbox: List[float], out_path: str, dpi: int = 300) -> str:
    """
    bbox: [l, t, r, b] as reported by Docling (native PDF coordinate space).
    page_no: 1-indexed page number, matching Docling's convention.
    """
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_no - 1]
        page_height = page.rect.height

        l, t, r, b = bbox
        # Convert bottom-left-origin (PDF native) -> top-left-origin (fitz.Rect)
        fitz_rect = fitz.Rect(l, page_height - t, r, page_height - b).normalize()

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        pix = page.get_pixmap(dpi=dpi, clip=fitz_rect)
        pix.save(out_path)
        return out_path
    finally:
        doc.close()
