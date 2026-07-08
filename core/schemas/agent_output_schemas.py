"""
Pydantic models for parsed paper content. These are what docling_parser.py
produces and what every downstream agent consumes.
"""

from typing import Optional, List
from pydantic import BaseModel


class Reference(BaseModel):
    raw_text: str
    title: Optional[str] = None
    year: Optional[int] = None


class Section(BaseModel):
    name: str                          # canonical name if recognized (Abstract, Method, ...), else raw heading text
    raw_heading: Optional[str] = None
    text: str
    page_start: Optional[int] = None


class Table(BaseModel):
    table_id: str
    page: Optional[int] = None
    markdown: str
    caption: Optional[str] = None


class Figure(BaseModel):
    figure_id: str
    page: Optional[int] = None
    bbox: Optional[List[float]] = None   # [l, t, r, b] in Docling's native PDF coordinate space
    image_path: Optional[str] = None     # populated by figure_cropper.py, not by docling_parser.py
    caption: Optional[str] = None
    ocr_text: Optional[str] = None       # reserved for the vision-optional extension


class ParsedPaper(BaseModel):
    title: str
    abstract: str
    sections: List[Section]
    tables: List[Table]
    figures: List[Figure]
    references: List[Reference]
    source_pdf_path: str
