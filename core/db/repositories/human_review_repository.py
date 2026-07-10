"""
Repository for `human_reviews` -- an independent, structured human-reviewer
submission for a reviewed paper, separate from the AI's own FinalReview and
from HumanApproval's approve/reject/revise decision on that AI draft.
"""

from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import select

from core.db.models import HumanReview


class HumanReviewRepository:
    def __init__(self, session: Session):
        self._session = session

    def get_by_reviewed_paper_id(self, reviewed_paper_id: int) -> Optional[HumanReview]:
        return self._session.execute(
            select(HumanReview).where(HumanReview.reviewed_paper_id == reviewed_paper_id)
        ).scalar_one_or_none()

    def upsert(self, reviewed_paper_id: int, **fields) -> HumanReview:
        """One row per reviewed_paper_id -- a resubmission updates the
        existing row's fields in place rather than inserting a duplicate."""
        existing = self.get_by_reviewed_paper_id(reviewed_paper_id)
        if existing is None:
            review = HumanReview(reviewed_paper_id=reviewed_paper_id, **fields)
            self._session.add(review)
        else:
            for key, value in fields.items():
                setattr(existing, key, value)
            review = existing
        self._session.flush()
        return review
