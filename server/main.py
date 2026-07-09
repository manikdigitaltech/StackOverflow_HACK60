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
import os
import uuid
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from core.db.repositories.review_repository import ReviewRepository
from core.db.session import get_session
from server.pipeline import check_system_health, query_paper_index, resume_with_approval, run_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = REPO_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AI Paper Reviewer -- pipeline demo server")

# run_id -> original uploaded filename (the saved PDF itself is just
# "{run_id}.pdf" -- this is only needed so the reviewed_papers row records
# what the user actually called their file).
_UPLOAD_FILENAMES: dict[str, str] = {}


def _csv_env(name: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, "").split(",") if item.strip()]


cors_origins = _csv_env("CORS_ALLOW_ORIGINS")
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )


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
    contents = await file.read()
    pdf_path.write_bytes(contents)
    _UPLOAD_FILENAMES[run_id] = file.filename

    return {"run_id": run_id, "filename": file.filename, "size_bytes": len(contents)}


@app.get("/api/stream/{run_id}")
def stream(run_id: str):
    pdf_path = UPLOAD_DIR / f"{run_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(404, f"No uploaded PDF found for run_id={run_id!r}. Upload first via /api/upload.")

    def event_source():
        for event in run_pipeline(run_id, str(pdf_path), uploaded_filename=_UPLOAD_FILENAMES.get(run_id)):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")


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


@app.get("/api/health")
def health():
    return check_system_health()


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
