"""
text_extractor.py

Extracts title, abstract, keywords, and methodology/conclusion/reference
text from an already-parsed paper JSON dict (PeerRead schema). No PDF
parsing - JSON input only, per project scope.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .config import CONCLUSION_ALIASES, METHODOLOGY_ALIASES, get_logger
from .models import PaperRecord

logger = get_logger(__name__)


class PaperExtractionError(Exception):
    """Raised when a paper JSON cannot be turned into a ``PaperRecord``."""


class PaperTextExtractor:
    """Extracts a normalized ``PaperRecord`` from a raw paper JSON dict.

    Single responsibility: text extraction and normalization only - no
    embedding, no similarity, no scoring.

    Example:
        >>> extractor = PaperTextExtractor()
        >>> record = extractor.extract(paper_json, paper_id="304.pdf")
    """

    def __init__(
        self,
        methodology_aliases: Optional[List[str]] = None,
        conclusion_aliases: Optional[List[str]] = None,
    ) -> None:
        self._methodology_aliases = [a.lower() for a in (methodology_aliases or METHODOLOGY_ALIASES)]
        self._conclusion_aliases = [a.lower() for a in (conclusion_aliases or CONCLUSION_ALIASES)]

    def extract(self, paper_json: Dict[str, Any], paper_id: str) -> PaperRecord:
        """Extract a ``PaperRecord`` from a raw paper JSON dict.

        Args:
            paper_json: Parsed PeerRead-style paper JSON. May nest fields
                under a ``metadata`` key.
            paper_id: Identifier to assign to the resulting record.

        Returns:
            A populated ``PaperRecord``.

        Raises:
            PaperExtractionError: If the input has no usable title,
                abstract, or sections.
        """
        if not isinstance(paper_json, dict):
            raise PaperExtractionError(f"Expected dict, got {type(paper_json).__name__}")

        metadata = paper_json.get("metadata", paper_json)

        try:
            title = self._normalize(metadata.get("title") or paper_json.get("title") or "")
            abstract = self._normalize(
                metadata.get("abstractText") or metadata.get("abstract") or paper_json.get("abstract") or ""
            )
            sections = metadata.get("sections") or paper_json.get("sections") or []
            references = metadata.get("references") or paper_json.get("references") or []
            year = metadata.get("year") or paper_json.get("year")

            methodology = self._extract_section_group(sections, self._methodology_aliases)
            conclusion = self._extract_section_group(sections, self._conclusion_aliases)
            keywords = self._extract_keywords(metadata, paper_json, title, abstract)
            references_text = self._flatten_references(references)

            if not title and not abstract and not sections:
                raise PaperExtractionError(f"Paper '{paper_id}' has no title, abstract, or sections")

            record = PaperRecord(
                paper_id=paper_id,
                title=title,
                abstract=abstract,
                keywords=keywords,
                methodology=self._normalize(methodology),
                conclusion=self._normalize(conclusion),
                references=references_text,
                year=int(year) if isinstance(year, (int, float)) else None,
            )
        except PaperExtractionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PaperExtractionError(f"Failed to extract paper '{paper_id}': {exc}") from exc

        logger.info(
            "Extracted paper '%s': title_len=%d abstract_len=%d methodology_len=%d keywords=%d",
            paper_id, len(record.title), len(record.abstract), len(record.methodology), len(record.keywords),
        )
        return record

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        if not text:
            return ""
        return re.sub(r"\s+", " ", str(text)).strip()

    def _extract_section_group(self, sections: List[Any], aliases: List[str]) -> str:
        matched = []
        for sec in sections:
            if not isinstance(sec, dict):
                continue
            heading = str(sec.get("heading") or sec.get("title") or "").lower().strip()
            text = sec.get("text") or sec.get("content") or ""
            if any(alias in heading for alias in aliases):
                matched.append(str(text))
        return " ".join(matched)

    def _extract_keywords(
        self, metadata: Dict[str, Any], paper_json: Dict[str, Any], title: str, abstract: str
    ) -> List[str]:
        """Use declared keywords if present, otherwise a light heuristic fallback."""
        declared = metadata.get("keywords") or paper_json.get("keywords")
        if declared:
            if isinstance(declared, str):
                return [k.strip() for k in re.split(r"[,;]", declared) if k.strip()]
            if isinstance(declared, list):
                return [str(k).strip() for k in declared if str(k).strip()]

        # Fallback: no declared keywords available - return empty list.
        # (Keyword inference from title/abstract is out of scope for
        # this extractor; downstream embedding uses title+abstract text
        # directly when keywords are unavailable.)
        return []

    def _flatten_references(self, references: List[Any]) -> str:
        if not references:
            return ""
        texts = []
        for ref in references:
            if isinstance(ref, str):
                texts.append(ref)
            elif isinstance(ref, dict):
                texts.append(str(ref.get("title") or ref.get("text") or ""))
        return self._normalize(" ".join(t for t in texts if t))
