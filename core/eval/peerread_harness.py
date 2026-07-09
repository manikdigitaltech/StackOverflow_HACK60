"""
PeerRead ICLR-2017 test-split evaluation harness -- the graded core of this
project (see docs/CONTEXT.md). Runs the full 9-agent LangGraph review
against each test-split paper's real PDF, maps `final_recommendation` to a
binary accept/reject prediction, and scores it against PeerRead's own
`accepted` ground truth with accuracy / F1 / Cohen's kappa.

The `test` split used here is the same one excluded from both the
literature-RAG and novelty corpora during ingestion (see
docs/PEERREAD_CORPUS_MODULE.md) -- no agent in this run has ever seen these
38 papers as background "prior art."
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from core.parsing.docling_parser import DoclingParser

logger = logging.getLogger(__name__)

# final_recommendation -> binary predicted accept/reject. "borderline" maps
# to reject: a conservative default (matches how a program committee treats
# a true toss-up) that keeps the mapping well-defined onto PeerRead's binary
# ground truth without inventing a third class it doesn't have.
RECOMMENDATION_TO_ACCEPT = {
    "accept": True,
    "weak_accept": True,
    "borderline": False,
    "weak_reject": False,
    "reject": False,
}


@dataclass
class PaperResult:
    paper_id: str
    title: str
    ground_truth_accepted: bool
    predicted_accept: Optional[bool] = None
    final_recommendation: Optional[str] = None
    confidence: Optional[str] = None
    elapsed_s: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "ground_truth_accepted": self.ground_truth_accepted,
            "predicted_accept": self.predicted_accept,
            "final_recommendation": self.final_recommendation,
            "confidence": self.confidence,
            "elapsed_s": self.elapsed_s,
            "error": self.error,
        }


def load_test_set(peerread_dir: Path, venue: str) -> list[dict]:
    """Loads the held-out `test` split: ground truth from `reviews/*.json`
    joined with the matching `pdfs/*.pdf` (both required per paper -- the
    graph runs against a real Docling-parsed PDF, not the review JSON)."""
    reviews_dir = peerread_dir / venue / "test" / "reviews"
    pdfs_dir = peerread_dir / venue / "test" / "pdfs"
    papers = []
    for review_path in sorted(reviews_dir.glob("*.json")):
        data = json.loads(review_path.read_text(encoding="utf-8"))
        pdf_path = pdfs_dir / f"{data['id']}.pdf"
        if not pdf_path.exists():
            logger.warning("No PDF for paper id=%s -- skipping", data["id"])
            continue
        papers.append(
            {
                "paper_id": f"{venue}:{data['id']}",
                "title": data.get("title", ""),
                "accepted": bool(data["accepted"]),
                "pdf_path": pdf_path,
            }
        )
    return papers


def run_single_paper(graph: Any, parser: DoclingParser, paper: dict) -> PaperResult:
    """Parses one test paper's real PDF and runs it through the full
    compiled review graph, end to end -- the same graph server/pipeline.py
    drives for a live upload, just invoked synchronously here."""
    result = PaperResult(
        paper_id=paper["paper_id"], title=paper["title"], ground_truth_accepted=paper["accepted"]
    )
    t0 = time.time()
    try:
        parsed = parser.parse(str(paper["pdf_path"]))
        final_state = graph.invoke(
            {"parsed_paper": parsed},
            config={"configurable": {"thread_id": f"eval-{paper['paper_id']}"}},
        )
        final_review = final_state["final_review"]
        result.final_recommendation = final_review.final_recommendation
        result.confidence = final_review.confidence
        result.predicted_accept = RECOMMENDATION_TO_ACCEPT[final_review.final_recommendation]
    except Exception as exc:  # a single paper's failure shouldn't abort the whole harness run
        logger.exception("Paper %s failed", paper["paper_id"])
        result.error = str(exc)
    result.elapsed_s = round(time.time() - t0, 2)
    return result


def compute_metrics(results: list[PaperResult]) -> dict:
    """Accuracy/F1/Cohen's kappa over papers that produced a usable
    prediction. Papers that errored (parse failure, LLM timeout, etc.) are
    counted and reported separately rather than silently dropped from the
    denominator without a trace."""
    usable = [r for r in results if r.predicted_accept is not None]
    n_errored = len(results) - len(usable)
    if not usable:
        return {"n_total": len(results), "n_usable": 0, "n_errored": n_errored}

    y_true = [r.ground_truth_accepted for r in usable]
    y_pred = [r.predicted_accept for r in usable]
    return {
        "n_total": len(results),
        "n_usable": len(usable),
        "n_errored": n_errored,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "cohen_kappa": round(cohen_kappa_score(y_true, y_pred), 4),
        "ground_truth_accept_rate": round(sum(y_true) / len(y_true), 4),
        "predicted_accept_rate": round(sum(y_pred) / len(y_pred), 4),
    }
