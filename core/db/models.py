"""
ORM models matching the MySQL schema in the architecture blueprint:
papers, chunks, reviewed_papers, review_assessments, human_approvals, reflection_flags.
"""

import enum
import datetime
from typing import Optional, List

from sqlalchemy import (
    BigInteger, String, Text, Integer, Boolean, DateTime, JSON,
    ForeignKey, Enum, UniqueConstraint, Index, func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.db.base import Base


# --- Enums ---

class PaperSource(str, enum.Enum):
    arxiv = "arxiv"
    semantic_scholar = "semantic_scholar"
    peerread = "peerread"


class ChunkStatus(str, enum.Enum):
    active = "active"
    stale = "stale"
    deleted = "deleted"


class ReviewedPaperStatus(str, enum.Enum):
    in_progress = "in_progress"
    awaiting_approval = "awaiting_approval"
    completed = "completed"
    failed = "failed"


class ApprovalDecision(str, enum.Enum):
    approve = "approve"
    revise = "revise"
    reject = "reject"


# --- Tables ---

class Paper(Base):
    """A literature paper ingested into KB1 (currently: PeerRead only)."""
    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source: Mapped[PaperSource] = mapped_column(Enum(PaperSource), nullable=False)
    external_id: Mapped[str] = mapped_column(String(128), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    abstract: Mapped[Optional[str]] = mapped_column(Text)
    authors: Mapped[Optional[str]] = mapped_column(Text)
    year: Mapped[Optional[int]] = mapped_column(Integer)
    url: Mapped[Optional[str]] = mapped_column(String(512))
    ingested_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())

    chunks: Mapped[List["Chunk"]] = relationship(back_populates="paper")

    __table_args__ = (
        UniqueConstraint("source", "external_id", name="uq_source_external"),
    )


class Chunk(Base):
    """One embedded passage of a Paper — the bridge row between MySQL and FAISS."""
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    paper_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("papers.id"), nullable=False)
    faiss_id: Mapped[int] = mapped_column(BigInteger, nullable=False)  # bridge to the FAISS vector store
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    section_type: Mapped[Optional[str]] = mapped_column(String(64))
    status: Mapped[ChunkStatus] = mapped_column(Enum(ChunkStatus), default=ChunkStatus.active)
    embedding_model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())

    paper: Mapped["Paper"] = relationship(back_populates="chunks")

    __table_args__ = (
        Index("idx_faiss_id", "faiss_id"),
        Index("idx_status", "status"),
    )


class ReviewedPaper(Base):
    """One paper submitted by a user to be reviewed (KB2 — never added to FAISS)."""
    __tablename__ = "reviewed_papers"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    uploaded_filename: Mapped[Optional[str]] = mapped_column(String(255))
    status: Mapped[ReviewedPaperStatus] = mapped_column(
        Enum(ReviewedPaperStatus), default=ReviewedPaperStatus.in_progress
    )
    parsed_title: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())
    completed_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime, nullable=True)

    assessments: Mapped[List["ReviewAssessment"]] = relationship(back_populates="reviewed_paper")
    approvals: Mapped[List["HumanApproval"]] = relationship(back_populates="reviewed_paper")
    reflection_flags: Mapped[List["ReflectionFlag"]] = relationship(back_populates="reviewed_paper")


class ReviewAssessment(Base):
    """One agent's output for one reviewed paper (novelty / methodology / citation / evidence / final)."""
    __tablename__ = "review_assessments"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    reviewed_paper_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("reviewed_papers.id"), nullable=False)
    agent_name: Mapped[str] = mapped_column(String(64), nullable=False)
    output_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    revision_pass: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())

    reviewed_paper: Mapped["ReviewedPaper"] = relationship(back_populates="assessments")

    __table_args__ = (
        Index("idx_reviewed_paper", "reviewed_paper_id"),
    )


class HumanApproval(Base):
    """A human reviewer's decision on a reviewed paper."""
    __tablename__ = "human_approvals"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    reviewed_paper_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("reviewed_papers.id"), nullable=False)
    decision: Mapped[ApprovalDecision] = mapped_column(Enum(ApprovalDecision), nullable=False)
    feedback: Mapped[Optional[str]] = mapped_column(Text)
    decided_by: Mapped[Optional[str]] = mapped_column(String(128))
    decided_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())

    reviewed_paper: Mapped["ReviewedPaper"] = relationship(back_populates="approvals")


class ReflectionFlag(Base):
    """An issue the Reflection Agent flagged in another agent's output."""
    __tablename__ = "reflection_flags"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    reviewed_paper_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("reviewed_papers.id"), nullable=False)
    flagged_agent: Mapped[str] = mapped_column(String(64), nullable=False)
    issue_text: Mapped[str] = mapped_column(Text, nullable=False)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())

    reviewed_paper: Mapped["ReviewedPaper"] = relationship(back_populates="reflection_flags")
