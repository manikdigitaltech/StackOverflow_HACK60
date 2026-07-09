"""Repository for `reflection_flags` — issues the Reflection Agent raised."""

from typing import List
from sqlalchemy.orm import Session
from sqlalchemy import select

from core.db.models import ReflectionFlag


class ReflectionRepository:
    def __init__(self, session: Session):
        self._session = session

    def save_flags(self, reviewed_paper_id: int, flagged_agent: str, issues: List[str]) -> List[ReflectionFlag]:
        flags = []
        for issue_text in issues:
            flag = ReflectionFlag(
                reviewed_paper_id=reviewed_paper_id,
                flagged_agent=flagged_agent,
                issue_text=issue_text,
            )
            self._session.add(flag)
            flags.append(flag)
        self._session.flush()
        return flags

    def get_unresolved(self, reviewed_paper_id: int) -> List[ReflectionFlag]:
        return list(
            self._session.execute(
                select(ReflectionFlag).where(
                    ReflectionFlag.reviewed_paper_id == reviewed_paper_id,
                    ReflectionFlag.resolved == False,  # noqa: E712
                )
            ).scalars()
        )
