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
    """One embedded passage of a Paper - the bridge row between MySQL and FAISS."""
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
    """One paper submitted by a user to be reviewed (KB2 - never added to FAISS)."""
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


# --- Human reviewer submission (independent of the AI's own FinalReview /
# HumanApproval flow -- see core/schemas/agent_output_schemas.py::FinalReview
# for the AI's own 5-point Recommendation enum, which this deliberately does
# NOT reuse, so this feature can never affect the graph or the eval harness) ---

class Venue(str, enum.Enum):
    miccai = "MICCAI"
    bmvc = "BMVC"
    neurips = "NeurIPS"
    ijcb = "IJCB"
    cvpr = "CVPR"
    other = "Other"


class ConflictOfInterest(str, enum.Enum):
    none = "none"
    declared = "declared"


class ReviewerExpertise(str, enum.Enum):
    expert = "expert"
    knowledgeable = "knowledgeable"
    passing_familiarity = "passing_familiarity"


class HumanReviewRating(str, enum.Enum):
    """The template's 6-point scale -- distinct from Recommendation (the
    AI's own 5-point enum used throughout core/graph and the eval harness)."""
    reject = "R"
    weak_reject = "WR"
    borderline_reject = "BR"
    borderline_accept = "BA"
    weak_accept = "WA"
    accept = "A"


class HumanReview(Base):
    """One human reviewer's independent, structured review of a reviewed
    paper, following Generic_Review_Template_Agentic_AI.docx's fields.
    One row per reviewed_paper_id -- see HumanReviewRepository.upsert,
    resubmission updates this same row rather than inserting a duplicate."""
    __tablename__ = "human_reviews"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    reviewed_paper_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("reviewed_papers.id"), nullable=False, unique=True
    )

    # --- Metadata ---
    paper_id_display: Mapped[Optional[str]] = mapped_column(String(128))
    paper_title: Mapped[Optional[str]] = mapped_column(Text)
    venue: Mapped[Optional[Venue]] = mapped_column(Enum(Venue))
    venue_other: Mapped[Optional[str]] = mapped_column(String(128))
    reviewer_name: Mapped[Optional[str]] = mapped_column(String(128))
    conflict_of_interest: Mapped[Optional[ConflictOfInterest]] = mapped_column(Enum(ConflictOfInterest))
    reviewer_expertise: Mapped[Optional[ReviewerExpertise]] = mapped_column(Enum(ReviewerExpertise))

    # --- Template sections A-G ---
    summary: Mapped[Optional[str]] = mapped_column(Text)                    # A) Summary of the Paper
    strengths: Mapped[Optional[list]] = mapped_column(JSON)                 # B) Strengths
    weaknesses_major: Mapped[Optional[list]] = mapped_column(JSON)          # C) Weaknesses -- Major/Technical
    weaknesses_minor: Mapped[Optional[list]] = mapped_column(JSON)          # C) Weaknesses -- Minor/Presentation
    questions_for_rebuttal: Mapped[Optional[list]] = mapped_column(JSON)    # D) Questions for Rebuttal
    final_conclusion: Mapped[Optional[str]] = mapped_column(Text)           # E) Final Review Conclusion
    rating: Mapped[Optional[HumanReviewRating]] = mapped_column(Enum(HumanReviewRating))  # F) Review Rating
    confidence: Mapped[Optional[int]] = mapped_column(Integer)              # F) Reviewer Confidence, 1-5
    rating_justification: Mapped[Optional[str]] = mapped_column(Text)       # G) Justify the Review Rating

    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    reviewed_paper: Mapped["ReviewedPaper"] = relationship()
