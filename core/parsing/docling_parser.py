"""
Docling-based PDF parser: layout analysis -> TableFormer -> OCR-if-needed,
producing a ParsedPaper.

Note: this module does NOT extract figure/formula images. It only captures
each figure/formula's bounding box, page number (and, for figures, caption
reference). Actual pixel extraction is figure_cropper.py's job (PyMuPDF) --
kept separate so this module stays focused on structure/understanding, not
image I/O.

Formula regions are detected by Docling's base layout model with no extra
model needed (DocItemLabel.FORMULA is just another layout class, like
"table" or "picture"), so bbox/page are always populated. The recognized
LaTeX/plaintext (FormulaItem.text) is only filled in when
settings.formula.enabled turns on Docling's do_formula_enrichment, which
downloads and runs its CodeFormulaV2 model on first use -- same
off-by-default, real-code-path philosophy as VISION__ENABLED for figure
description.
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional

import fitz  # PyMuPDF -- used here only for a cheap page-count check
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import TextItem, TableItem, PictureItem, FormulaItem

from core.config.settings import settings
from core.schemas.agent_output_schemas import ParsedPaper, Table, Figure, Formula
from core.parsing.section_segmenter import segment_sections
from core.parsing.reference_extractor import extract_references
from core.utils.guardrails import sanitize_pdf_text


class DoclingParser:
    def __init__(self):
        pipeline_options = PdfPipelineOptions(do_formula_enrichment=settings.formula.enabled)
        self._converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

    def parse(self, pdf_path: str) -> ParsedPaper:
        pdf_path = str(pdf_path)

        # Cheap page-count check BEFORE running Docling's full pipeline --
        # a safety valve against an accidental huge upload (e.g. a 300-page
        # dissertation) hanging the whole review for minutes on CPU.
        quick_doc = fitz.open(pdf_path)
        total_pages = quick_doc.page_count
        quick_doc.close()

        convert_kwargs = {}
        max_pages = settings.parsing.max_pages_hard_cap
        if max_pages and total_pages > max_pages:
            print(
                f"[DoclingParser] WARNING: '{pdf_path}' has {total_pages} pages, "
                f"exceeding max_pages_hard_cap={max_pages}. Processing only the "
                f"first {max_pages} pages -- adjust PARSING__MAX_PAGES_HARD_CAP "
                f"in .env if this document should be processed in full."
            )
            convert_kwargs["page_range"] = (1, max_pages)

        result = self._converter.convert(pdf_path, **convert_kwargs)
        doc = result.document

        # --- Flatten the reading-order stream of text items ---
        # FormulaItem is a TextItem subclass, but is collected separately
        # below (like tables/pictures) rather than folded into section
        # prose -- an equation's raw/LaTeX text reads as noise in prose
        # context, and downstream agents don't need it there.
        text_items: List[Tuple[str, str, Optional[int]]] = []
        formula_items: List[FormulaItem] = []
        for item, _level in doc.iterate_items():
            if isinstance(item, FormulaItem):
                formula_items.append(item)
            elif isinstance(item, TextItem):
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

        # --- Figures (bbox + page + caption only - no pixels here) ---
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

        # --- Formulas (bbox + page always; recognized text only if enabled) ---
        formulas: List[Formula] = []
        for i, formula_item in enumerate(formula_items):
            page_no = formula_item.prov[0].page_no if getattr(formula_item, "prov", None) else None
            bbox = formula_item.prov[0].bbox if getattr(formula_item, "prov", None) else None
            formulas.append(Formula(
                formula_id=f"formula_{i + 1}",
                page=page_no,
                bbox=self._bbox_to_list(bbox),
                text=formula_item.text or None,
            ))

        references = extract_references(text_items)

        parsed_paper = ParsedPaper(
            title=title,
            abstract=abstract,
            sections=sections,
            tables=tables,
            figures=figures,
            formulas=formulas,
            references=references,
            source_pdf_path=pdf_path,
        )
        self._sanitize_in_place(parsed_paper, pdf_path)
        return parsed_paper

    @staticmethod
    def _sanitize_in_place(paper: ParsedPaper, pdf_path: str) -> None:
        """Strips prompt-injection patterns from every free-text field pulled
        out of the PDF, once, here -- so no downstream agent prompt has to
        think about untrusted content. See core/utils/guardrails.py."""
        any_triggered = False

        paper.title, triggered = sanitize_pdf_text(paper.title)
        any_triggered = any_triggered or triggered
        paper.abstract, triggered = sanitize_pdf_text(paper.abstract)
        any_triggered = any_triggered or triggered

        for section in paper.sections:
            section.text, triggered = sanitize_pdf_text(section.text)
            any_triggered = any_triggered or triggered

        for table in paper.tables:
            table.markdown, triggered = sanitize_pdf_text(table.markdown)
            any_triggered = any_triggered or triggered
            if table.caption:
                table.caption, triggered = sanitize_pdf_text(table.caption)
                any_triggered = any_triggered or triggered

        for figure in paper.figures:
            if figure.caption:
                figure.caption, triggered = sanitize_pdf_text(figure.caption)
                any_triggered = any_triggered or triggered

        for formula in paper.formulas:
            if formula.text:
                formula.text, triggered = sanitize_pdf_text(formula.text)
                any_triggered = any_triggered or triggered

        for reference in paper.references:
            reference.raw_text, triggered = sanitize_pdf_text(reference.raw_text)
            any_triggered = any_triggered or triggered

        if any_triggered:
            print(f"[DoclingParser] Guardrail: adversarial pattern(s) stripped from '{pdf_path}'.")

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
                return "(table content unavailable - check docling version)"

    @staticmethod
    def _resolve_caption(item, doc) -> Optional[str]:
        """
        Docling stores captions as REFERENCES to other TextItems, not inline
        text - this resolves that reference. If your installed docling
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
    print(f"Formulas: {len(parsed.formulas)}")
    print(f"References: {len(parsed.references)}")
