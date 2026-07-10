"""
Thin FastAPI layer whose only job is to run server/pipeline.py's real stages
on an uploaded PDF and push each stage's result to the browser live, via
Server-Sent Events, so ai_paper_reviewer_ui.html can render actual pipeline
state instead of its original hardcoded demo data.

core/db's MySQL tables (reviewed_papers, review_assessments, human_approvals,
reflection_flags) ARE the real persistence layer now -- server/pipeline.py
writes each judgment agent's output and reflection's flags as they complete.
The review run genuinely pauses at a human-approval interrupt (LangGraph's
interrupt(), see core/graph/nodes.py) before issuing a final recommendation;
POST /api/approval/{run_id} below resumes that parked run for real via
Command(resume=...) AND persists the decision to MySQL. Run metadata for the
RAG-query endpoint still lives in-memory in server/pipeline.py (that part is
a demo/inspection convenience, not lifecycle data worth persisting).

Run with:  python -m uvicorn server.main:app --reload --port 8000
Then open: http://localhost:8000/
"""
from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core.db.models import ConflictOfInterest, HumanReviewRating, ReviewerExpertise, Venue
from core.db.repositories.human_review_repository import HumanReviewRepository
from core.db.repositories.review_repository import ReviewRepository
from core.db.session import get_session
from server.pipeline import (
    check_system_health,
    load_eval_metrics,
    query_paper_index,
    resume_with_approval,
    run_pipeline,
    run_rebuttal_rereview,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = REPO_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_CROPS_DIR = REPO_ROOT / "data" / "figure_crops"
FIGURE_CROPS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AI Paper Reviewer -- pipeline demo server")

# Serves the PNG crops core/parsing/figure_cropper.py writes, so the live UI
# can render real figure thumbnails (not just captions/descriptions) as soon
# as the "vision" stage event reports their image_url.
app.mount("/figure_crops", StaticFiles(directory=str(FIGURE_CROPS_DIR)), name="figure_crops")

# run_id -> original uploaded filename (the saved PDF itself is just
# "{run_id}.pdf" -- this is only needed so the reviewed_papers row records
# what the user actually called their file).
_UPLOAD_FILENAMES: dict[str, str] = {}

# --- Resource limits (OWASP LLM10, Unbounded Consumption -- see
# docs/OWASP_LLM_SECURITY.md). A review run monopolizes the local GPU for
# many minutes, so cap how much can be requested at once.
MAX_UPLOAD_BYTES = 30 * 1024 * 1024  # research-paper PDFs are single-digit MB
MAX_CONCURRENT_REVIEWS = 2           # full-graph runs allowed at the same time

_active_reviews = 0
_reviews_lock = threading.Lock()


def _acquire_review_slot() -> bool:
    global _active_reviews
    with _reviews_lock:
        if _active_reviews >= MAX_CONCURRENT_REVIEWS:
            return False
        _active_reviews += 1
        return True


def _release_review_slot() -> None:
    global _active_reviews
    with _reviews_lock:
        _active_reviews = max(0, _active_reviews - 1)


@app.get("/")
def index():
    return FileResponse(REPO_ROOT / "ai_paper_reviewer_ui.html")


@app.get("/dashboard")
def dashboard():
    return FileResponse(REPO_ROOT / "ai_paper_reviewer_saas_dashboard.html")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    run_id = uuid.uuid4().hex[:12]
    pdf_path = UPLOAD_DIR / f"{run_id}.pdf"
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"PDF too large -- limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    pdf_path.write_bytes(contents)
    _UPLOAD_FILENAMES[run_id] = file.filename

    return {"run_id": run_id, "filename": file.filename, "size_bytes": len(contents)}


@app.get("/api/stream/{run_id}")
def stream(run_id: str):
    pdf_path = UPLOAD_DIR / f"{run_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(404, f"No uploaded PDF found for run_id={run_id!r}. Upload first via /api/upload.")

    if not _acquire_review_slot():
        raise HTTPException(429, f"Too many reviews running (limit {MAX_CONCURRENT_REVIEWS}). Try again once one finishes.")

    def event_source():
        # The slot is held for the life of the stream; FastAPI closes the
        # generator on client disconnect, so the finally always releases it.
        try:
            for event in run_pipeline(run_id, str(pdf_path), uploaded_filename=_UPLOAD_FILENAMES.get(run_id)):
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            _release_review_slot()

    return StreamingResponse(event_source(), media_type="text/event-stream")


class RebuttalRequest(BaseModel):
    # the authors' written response to the original review; length-capped so
    # a single request can't flood the context window (OWASP LLM10)
    rebuttal_text: str = Field(..., min_length=1, max_length=20_000)


@app.post("/api/rebuttal/{run_id}")
def rebuttal(run_id: str, body: RebuttalRequest):
    """Rebuttal-aware re-review: re-run the reviewed paper with the authors'
    rebuttal folded in, returning the revised recommendation, how it moved vs.
    the original, and a `rebuttal_run_id` to approve the revised verdict on via
    POST /api/approval/{rebuttal_run_id}."""
    if not _acquire_review_slot():
        raise HTTPException(429, f"Too many reviews running (limit {MAX_CONCURRENT_REVIEWS}). Try again once one finishes.")
    try:
        result = run_rebuttal_rereview(run_id, body.rebuttal_text)
    finally:
        _release_review_slot()
    if "error" in result:
        raise HTTPException(409, result["error"])
    return result


@app.get("/api/query/{run_id}")
def query(run_id: str, q: str, k: int = 5):
    return query_paper_index(run_id, q, k=k)


class ApprovalRequest(BaseModel):
    decision: str  # "approve"/"approved", "reject"/"rejected", "revise"/"revised" -- synonyms normalized graph-side
    approver: Optional[str] = None
    comment: Optional[str] = None
    override_recommendation: Optional[
        Literal["reject", "weak_reject", "borderline", "weak_accept", "accept"]
    ] = None                           # only meaningful with a "revise"/"revised" decision


@app.post("/api/approval/{run_id}")
def approval(run_id: str, body: ApprovalRequest):
    """Resumes the review run genuinely parked at the human-approval
    interrupt (see core.graph.nodes.human_approval) with this decision, and
    persists it to MySQL. A 404 here means the run never reached the
    approval gate or was already decided -- see resume_with_approval's own
    docstring for the exact guard."""
    result = resume_with_approval(run_id, body.model_dump(exclude_none=True))
    if not result["ok"]:
        raise HTTPException(404, result["error"])
    return result


class HumanReviewRequest(BaseModel):
    """A person's own structured review of the paper, following
    Generic_Review_Template_Agentic_AI.docx -- entirely independent of the
    AI's FinalReview/HumanApproval flow above: no shared schema, no shared
    enum, no interaction with the graph. `rating` is this template's 6-point
    scale (R/WR/BR/BA/WA/A), not FinalReview's 5-point Recommendation."""
    paper_id_display: Optional[str] = None
    paper_title: Optional[str] = None
    venue: Optional[Literal["MICCAI", "BMVC", "NeurIPS", "IJCB", "CVPR", "Other"]] = None
    venue_other: Optional[str] = None
    reviewer_name: Optional[str] = None
    conflict_of_interest: Optional[Literal["none", "declared"]] = None
    reviewer_expertise: Optional[Literal["expert", "knowledgeable", "passing_familiarity"]] = None
    summary: Optional[str] = None
    strengths: list[str] = []
    weaknesses_major: list[str] = []
    weaknesses_minor: list[str] = []
    questions_for_rebuttal: list[str] = []
    final_conclusion: Optional[str] = None
    rating: Optional[Literal["R", "WR", "BR", "BA", "WA", "A"]] = None
    confidence: Optional[int] = Field(default=None, ge=1, le=5)
    rating_justification: Optional[str] = None


_VENUE_BY_VALUE = {v.value: v for v in Venue}
_COI_BY_VALUE = {v.value: v for v in ConflictOfInterest}
_EXPERTISE_BY_VALUE = {v.value: v for v in ReviewerExpertise}
_RATING_BY_VALUE = {v.value: v for v in HumanReviewRating}


def _serialize_human_review(review) -> dict:
    return {
        "paper_id_display": review.paper_id_display,
        "paper_title": review.paper_title,
        "venue": review.venue.value if review.venue else None,
        "venue_other": review.venue_other,
        "reviewer_name": review.reviewer_name,
        "conflict_of_interest": review.conflict_of_interest.value if review.conflict_of_interest else None,
        "reviewer_expertise": review.reviewer_expertise.value if review.reviewer_expertise else None,
        "summary": review.summary,
        "strengths": review.strengths or [],
        "weaknesses_major": review.weaknesses_major or [],
        "weaknesses_minor": review.weaknesses_minor or [],
        "questions_for_rebuttal": review.questions_for_rebuttal or [],
        "final_conclusion": review.final_conclusion,
        "rating": review.rating.value if review.rating else None,
        "confidence": review.confidence,
        "rating_justification": review.rating_justification,
        "updated_at": review.updated_at.isoformat() if review.updated_at else None,
    }


@app.post("/api/human-review/{run_id}")
def submit_human_review(run_id: str, body: HumanReviewRequest):
    """Submit or update this run's independent, human-written structured
    review. Re-POSTing overwrites the same row (one per reviewed paper,
    see HumanReviewRepository.upsert) rather than creating a duplicate."""
    try:
        with get_session() as session:
            reviewed_paper = ReviewRepository(session).get_by_trace_id(run_id)
            if reviewed_paper is None:
                raise HTTPException(404, f"No reviewed paper found for run_id={run_id!r}. Upload the paper first.")
            fields = body.model_dump(exclude={"venue", "conflict_of_interest", "reviewer_expertise", "rating"})
            fields["venue"] = _VENUE_BY_VALUE.get(body.venue) if body.venue else None
            fields["conflict_of_interest"] = _COI_BY_VALUE.get(body.conflict_of_interest) if body.conflict_of_interest else None
            fields["reviewer_expertise"] = _EXPERTISE_BY_VALUE.get(body.reviewer_expertise) if body.reviewer_expertise else None
            fields["rating"] = _RATING_BY_VALUE.get(body.rating) if body.rating else None
            review = HumanReviewRepository(session).upsert(reviewed_paper.id, **fields)
            return {"ok": True, "human_review": _serialize_human_review(review)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, f"Could not save human review -- MySQL unreachable or not migrated: {exc}")


@app.get("/api/human-review/{run_id}")
def get_human_review(run_id: str):
    """Fetch a previously-saved human-review draft/submission for this run,
    if any -- used to restore form state after a page refresh."""
    try:
        with get_session() as session:
            reviewed_paper = ReviewRepository(session).get_by_trace_id(run_id)
            if reviewed_paper is None:
                raise HTTPException(404, f"No reviewed paper found for run_id={run_id!r}.")
            review = HumanReviewRepository(session).get_by_reviewed_paper_id(reviewed_paper.id)
            if review is None:
                return {"found": False}
            return {"found": True, "human_review": _serialize_human_review(review)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, f"Human review unavailable -- MySQL unreachable or not migrated: {exc}")


@app.get("/api/health")
def health():
    return check_system_health()


@app.get("/api/eval-metrics")
def eval_metrics():
    """Real accuracy/F1/Cohen's-kappa numbers from the PeerRead evaluation
    harness (scripts/run_peerread_evaluation.py), read from whichever
    output_results/*.metrics.json file(s) exist -- a snapshot from the last
    time that harness was run, not a live per-request statistic (see
    load_eval_metrics's own docstring)."""
    return {"runs": load_eval_metrics()}


@app.get("/api/history")
def history(limit: int = 20, offset: int = 0):
    """Past reviews persisted to MySQL by server/pipeline.py -- real rows,
    not synthesized from in-memory run state (that's why this survives a
    server restart while /api/stream/{run_id} does not)."""
    try:
        with get_session() as session:
            rows = ReviewRepository(session).get_history(limit=limit, offset=offset)
            return [
                {
                    "trace_id": r.trace_id,
                    "uploaded_filename": r.uploaded_filename,
                    "parsed_title": r.parsed_title,
                    "status": r.status.value if r.status else None,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                }
                for r in rows
            ]
    except Exception as exc:
        raise HTTPException(503, f"Review history unavailable -- MySQL unreachable or not migrated: {exc}")


@app.get("/api/history/{trace_id}")
def history_detail(trace_id: str):
    """One past review's full assessment trail -- every review_assessments
    row (novelty/methodology/citation/evidence/final, across every revision
    pass) plus every reflection_flags row raised against them."""
    try:
        with get_session() as session:
            repo = ReviewRepository(session)
            reviewed_paper = repo.get_by_trace_id(trace_id)
            if reviewed_paper is None:
                raise HTTPException(404, f"No reviewed paper found for trace_id={trace_id!r}.")
            assessments = repo.get_assessments(reviewed_paper.id)
            return {
                "trace_id": reviewed_paper.trace_id,
                "uploaded_filename": reviewed_paper.uploaded_filename,
                "parsed_title": reviewed_paper.parsed_title,
                "status": reviewed_paper.status.value if reviewed_paper.status else None,
                "created_at": reviewed_paper.created_at.isoformat() if reviewed_paper.created_at else None,
                "completed_at": reviewed_paper.completed_at.isoformat() if reviewed_paper.completed_at else None,
                "assessments": [
                    {
                        "agent_name": a.agent_name,
                        "revision_pass": a.revision_pass,
                        "output_json": a.output_json,
                        "created_at": a.created_at.isoformat() if a.created_at else None,
                    }
                    for a in assessments
                ],
            }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, f"Review history unavailable -- MySQL unreachable or not migrated: {exc}")
