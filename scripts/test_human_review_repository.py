"""
Verifies HumanReviewRepository's upsert semantics against real MySQL: an
independent, structured human-reviewer submission (see
Generic_Review_Template_Agentic_AI.docx / docs and ai_paper_reviewer_ui.html's
"Submit Review" tab) that lives in its own `human_reviews` table, separate
from the AI's FinalReview/HumanApproval flow.

Run with: python -m scripts.test_human_review_repository

Expected: prints OK lines with no AssertionError. Safe to run repeatedly --
it creates a fresh reviewed_paper (unique trace_id) each run.
"""
import uuid

from core.db.session import get_session
from core.db.repositories.review_repository import ReviewRepository
from core.db.repositories.human_review_repository import HumanReviewRepository
from core.db.models import HumanReviewRating

trace_id = f"test-human-review-{uuid.uuid4().hex[:8]}"

with get_session() as session:
    reviewed_paper = ReviewRepository(session).create_reviewed_paper(trace_id, "test.pdf")
    repo = HumanReviewRepository(session)

    assert repo.get_by_reviewed_paper_id(reviewed_paper.id) is None
    print("OK: no submission before the first upsert")

    first = repo.upsert(
        reviewed_paper.id,
        summary="first draft",
        strengths=["clear writing"],
        rating=HumanReviewRating.borderline_accept,
        confidence=3,
    )
    assert first.summary == "first draft"
    assert first.rating == HumanReviewRating.borderline_accept
    print(f"OK: first upsert created row id={first.id}")

    second = repo.upsert(
        reviewed_paper.id,
        summary="revised submission",
        strengths=["clear writing", "strong experiments"],
        rating=HumanReviewRating.accept,
        confidence=5,
    )
    assert second.id == first.id, "resubmission must UPDATE the existing row, not insert a new one"
    assert second.summary == "revised submission"
    assert second.rating == HumanReviewRating.accept
    print(f"OK: second upsert updated the SAME row (id={second.id}), not a duplicate")

    fetched = repo.get_by_reviewed_paper_id(reviewed_paper.id)
    assert fetched.id == first.id
    assert fetched.summary == "revised submission"
    assert fetched.strengths == ["clear writing", "strong experiments"]
    assert fetched.confidence == 5
    print("OK: get_by_reviewed_paper_id returns the latest values")

print("All human_review_repository checks passed.")
