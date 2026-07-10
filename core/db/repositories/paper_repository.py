"""Repository for the `papers` table - literature papers ingested into KB1."""

from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import select

from core.db.models import Paper, PaperSource


class PaperRepository:
    def __init__(self, session: Session):
        self._session = session

    def get_or_create(
        self,
        source: PaperSource,
        external_id: str,
        title: str,
        abstract: Optional[str] = None,
        authors: Optional[str] = None,
        year: Optional[int] = None,
        url: Optional[str] = None,
    ) -> Paper:
        existing = self._session.execute(
            select(Paper).where(Paper.source == source, Paper.external_id == external_id)
        ).scalar_one_or_none()
        if existing:
            return existing

        paper = Paper(
            source=source, external_id=external_id, title=title,
            abstract=abstract, authors=authors, year=year, url=url,
        )
        self._session.add(paper)
        self._session.flush()  # populates paper.id without committing yet
        return paper

    def get_by_id(self, paper_id: int) -> Optional[Paper]:
        return self._session.get(Paper, paper_id)
