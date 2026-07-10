"""
Repository for `reviewed_papers` and `review_assessments` - the core audit
trail that every LangGraph node writes to as it completes.
"""

from typing import Optional, List, Any
from sqlalchemy.orm import Session
from sqlalchemy import select, desc, func

from core.db.models import ReviewedPaper, ReviewAssessment, ReviewedPaperStatus


class ReviewRepository:
    def __init__(self, session: Session):
        self._session = session

    def create_reviewed_paper(self, trace_id: str, uploaded_filename: str) -> ReviewedPaper:
        reviewed_paper = ReviewedPaper(
            trace_id=trace_id,
            uploaded_filename=uploaded_filename,
            status=ReviewedPaperStatus.in_progress,
        )
        self._session.add(reviewed_paper)
        self._session.flush()
        return reviewed_paper

    def get_by_trace_id(self, trace_id: str) -> Optional[ReviewedPaper]:
        return self._session.execute(
            select(ReviewedPaper).where(ReviewedPaper.trace_id == trace_id)
        ).scalar_one_or_none()

    def update_parsed_title(self, reviewed_paper_id: int, title: str) -> None:
        rp = self._session.get(ReviewedPaper, reviewed_paper_id)
        if rp:
            rp.parsed_title = title
            self._session.flush()

    def update_status(self, reviewed_paper_id: int, status: ReviewedPaperStatus) -> None:
        rp = self._session.get(ReviewedPaper, reviewed_paper_id)
        if rp:
            rp.status = status
            if status == ReviewedPaperStatus.completed:
                # func.now(), not datetime.utcnow(): every other timestamp
                # column here uses MySQL's own NOW() via server_default,
                # which returns the DB server's local time, not UTC -- mixing
                # the two produced a 5.5-hour-skewed completed_at (IST vs UTC)
                # that couldn't be compared against created_at/decided_at.
                rp.completed_at = func.now()
            self._session.flush()

    def save_assessment(
        self,
        reviewed_paper_id: int,
        agent_name: str,
        output_json: dict[str, Any],
        revision_pass: int = 0,
    ) -> ReviewAssessment:
        assessment = ReviewAssessment(
            reviewed_paper_id=reviewed_paper_id,
            agent_name=agent_name,
            output_json=output_json,
            revision_pass=revision_pass,
        )
        self._session.add(assessment)
        self._session.flush()
        return assessment

    def get_assessments(self, reviewed_paper_id: int) -> List[ReviewAssessment]:
        return list(
            self._session.execute(
                select(ReviewAssessment)
                .where(ReviewAssessment.reviewed_paper_id == reviewed_paper_id)
                .order_by(ReviewAssessment.created_at)
            ).scalars()
        )

    def get_history(self, limit: int = 20, offset: int = 0) -> List[ReviewedPaper]:
        return list(
            self._session.execute(
                select(ReviewedPaper)
                .order_by(desc(ReviewedPaper.created_at))
                .limit(limit)
                .offset(offset)
            ).scalars()
        )
