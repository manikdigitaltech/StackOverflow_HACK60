"""
Thin FastAPI layer whose only job is to run server/pipeline.py's real stages
on an uploaded PDF and push each stage's result to the browser live, via
Server-Sent Events, so ai_paper_reviewer_ui.html can render actual pipeline
state instead of its original hardcoded demo data.

This is a demo/inspection surface, not the review-lifecycle system --
core/db's MySQL tables (reviewed_papers, review_assessments, human_approvals)
are the real persistence layer for once agents exist and actually write
reviews. Nothing here writes to MySQL; run state lives in-memory in
server/pipeline.py for the life of the process.

Run with:  python -m uvicorn server.main:app --reload --port 8000
Then open: http://localhost:8000/
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from server.pipeline import check_system_health, query_paper_index, resume_with_approval, run_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = REPO_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AI Paper Reviewer -- pipeline demo server")


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

    return {"run_id": run_id, "filename": file.filename, "size_bytes": len(contents)}


@app.get("/api/stream/{run_id}")
def stream(run_id: str):
    pdf_path = UPLOAD_DIR / f"{run_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(404, f"No uploaded PDF found for run_id={run_id!r}. Upload first via /api/upload.")

    def event_source():
        for event in run_pipeline(run_id, str(pdf_path)):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")


class ApprovalRequest(BaseModel):
    decision: str                      # "approve"/"approved", "reject"/"rejected", "revise"/"revised" -- synonyms normalized graph-side
    approver: Optional[str] = None
    comment: Optional[str] = None
    override_recommendation: Optional[
        Literal["reject", "weak_reject", "borderline", "weak_accept", "accept"]
    ] = None                           # only meaningful with a "revised" decision


@app.post("/api/approve/{run_id}")
def approve(run_id: str, body: ApprovalRequest):
    """Resume a review run parked at the human-approval interrupt with the
    human's decision; returns the approval record and the (possibly
    override-rewritten) final review."""
    result = resume_with_approval(run_id, body.model_dump(exclude_none=True))
    if "error" in result:
        raise HTTPException(409, result["error"])
    return result


@app.get("/api/query/{run_id}")
def query(run_id: str, q: str, k: int = 5):
    return query_paper_index(run_id, q, k=k)


@app.get("/api/health")
def health():
    return check_system_health()
