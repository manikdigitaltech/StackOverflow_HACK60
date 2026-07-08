"""
Docling-based PDF parser: layout analysis -> TableFormer -> OCR-if-needed,
producing a ParsedPaper.

Note: this module does NOT extract figure images. It only captures each
figure's bounding box, page number, and caption reference. Actual pixel
extraction is figure_cropper.py's job (PyMuPDF) — kept separate so this
module stays focused on structure/understanding, not image I/O.
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional

from docling.document_converter import DocumentConverter
from docling_core.types.doc import TextItem, TableItem, PictureItem

from core.schemas.agent_output_schemas import ParsedPaper, Table, Figure
from core.parsing.section_segmenter import segment_sections
from core.parsing.reference_extractor import extract_references


class DoclingParser:
    def __init__(self):
        self._converter = DocumentConverter()

    def parse(self, pdf_path: str) -> ParsedPaper:
        pdf_path = str(pdf_path)
        result = self._converter.convert(pdf_path)
        doc = result.document

        # --- Flatten the reading-order stream of text items ---
        text_items: List[Tuple[str, str, Optional[int]]] = []
        for item, _level in doc.iterate_items():
            if isinstance(item, TextItem):
                page_no = item.prov[0].page_no if getattr(item, "prov", None) else None
                text_items.append((getattr(item, "label", "text"), item.text, page_no))

        sections, title, abstract = segment_sections(text_items)

        # --- Tables (TableFormer structure, via doc.tables) ---
        tables: List[Table] = []
        for i, table_item in enumerate(doc.tables):
            markdown = self._table_to_markdown(table_item, doc)
            page_no = table_item.prov[0].page_no if getattr(table_item, "prov", None) else None
            tables.append(Table(
                table_id=f"table_{i + 1}",
                page=page_no,
                markdown=markdown,
                caption=self._resolve_caption(table_item, doc),
            ))

        # --- Figures (bbox + page + caption only — no pixels here) ---
        figures: List[Figure] = []
        for i, picture_item in enumerate(doc.pictures):
            page_no = picture_item.prov[0].page_no if getattr(picture_item, "prov", None) else None
            bbox = picture_item.prov[0].bbox if getattr(picture_item, "prov", None) else None
            figures.append(Figure(
                figure_id=f"figure_{i + 1}",
                page=page_no,
                bbox=self._bbox_to_list(bbox),
                caption=self._resolve_caption(picture_item, doc),
            ))

        references = extract_references(text_items)

        return ParsedPaper(
            title=title,
            abstract=abstract,
            sections=sections,
            tables=tables,
            figures=figures,
            references=references,
            source_pdf_path=pdf_path,
        )

    @staticmethod
    def _table_to_markdown(table_item: TableItem, doc) -> str:
        try:
            df = table_item.export_to_dataframe(doc=doc)
            return df.to_markdown(index=False)
        except Exception:
            # Fallback if `tabulate` isn't installed, or export_to_dataframe's
            # signature differs on your installed docling version.
            try:
                return table_item.export_to_html()
            except Exception:
                return "(table content unavailable — check docling version)"

    @staticmethod
    def _resolve_caption(item, doc) -> Optional[str]:
        """
        Docling stores captions as REFERENCES to other TextItems, not inline
        text — this resolves that reference. If your installed docling
        version exposes captions differently, this is the function to fix
        based on the actual error/output you see.
        """
        captions = getattr(item, "captions", None)
        if not captions:
            return None
        resolved_texts = []
        for ref in captions:
            try:
                ref_item = ref.resolve(doc) if hasattr(ref, "resolve") else None
                if ref_item is not None and hasattr(ref_item, "text"):
                    resolved_texts.append(ref_item.text)
            except Exception:
                continue
        return " ".join(resolved_texts) if resolved_texts else None

    @staticmethod
    def _bbox_to_list(bbox) -> Optional[List[float]]:
        if bbox is None:
            return None
        for attrs in (("l", "t", "r", "b"), ("x0", "y0", "x1", "y1")):
            if all(hasattr(bbox, a) for a in attrs):
                return [float(getattr(bbox, a)) for a in attrs]
        return None


if __name__ == "__main__":
    # Quick manual test: python -m core.parsing.docling_parser path/to/sample.pdf
    if len(sys.argv) < 2:
        print("Usage: python -m core.parsing.docling_parser <path_to_pdf>")
        sys.exit(1)

    parser = DoclingParser()
    parsed = parser.parse(sys.argv[1])

    print(f"Title: {parsed.title}")
    print(f"Abstract ({len(parsed.abstract)} chars): {parsed.abstract[:200]}...")
    print(f"Sections ({len(parsed.sections)}): {[s.name for s in parsed.sections]}")
    print(f"Tables: {len(parsed.tables)}")
    print(f"Figures: {len(parsed.figures)}")
    print(f"References: {len(parsed.references)}")
