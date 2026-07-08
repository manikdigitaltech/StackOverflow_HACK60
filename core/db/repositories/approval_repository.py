"""Repository for `human_approvals` — records the reviewer's decision."""

from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import select

from core.db.models import HumanApproval, ApprovalDecision


class ApprovalRepository:
    def __init__(self, session: Session):
        self._session = session

    def save_decision(
        self,
        reviewed_paper_id: int,
        decision: ApprovalDecision,
        feedback: Optional[str] = None,
        decided_by: Optional[str] = None,
    ) -> HumanApproval:
        approval = HumanApproval(
            reviewed_paper_id=reviewed_paper_id,
            decision=decision,
            feedback=feedback,
            decided_by=decided_by,
        )
        self._session.add(approval)
        self._session.flush()
        return approval

    def get_latest(self, reviewed_paper_id: int) -> Optional[HumanApproval]:
        return self._session.execute(
            select(HumanApproval)
            .where(HumanApproval.reviewed_paper_id == reviewed_paper_id)
            .order_by(HumanApproval.decided_at.desc())
            .limit(1)
        ).scalar_one_or_none()
