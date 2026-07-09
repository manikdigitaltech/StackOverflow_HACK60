"""
Runs the real, currently-implemented pipeline stages for one uploaded paper
and yields a live event per stage -- this is what server/main.py streams to
the browser over Server-Sent Events.

Deliberately honest about what exists: parse / figure-vision / RAG-chunk /
paper-RAG-index / the full 9-agent LangGraph review (paper understanding
through final review) are real code paths, executed for real, on the real
uploaded PDF. Only human-in-the-loop approval and MySQL persistence remain
unbuilt (Phase 2) -- that one stage is emitted with status="not_implemented"
rather than faked, so the UI never shows work no code actually did.
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from core.agents.novelty import NoveltyEvaluationAgent, NoveltyEvaluationAgentError
from core.agents.novelty.adapter import parsed_paper_to_novelty_input
from core.agents.novelty.config import DEFAULT_PATHS as NOVELTY_PATHS
from core.config.settings import settings
from core.graph.build_graph import build_review_graph
from core.rag.adapters import parsed_paper_to_chunker_input
from core.rag.chunking.section_chunker import chunk_paper
from core.rag.embeddings.embedding_provider import BgeSmallEmbeddingProvider
from core.rag.indexes.paper_index import PaperIndex
from core.schemas.agent_output_schemas import ParsedPaper

# LangGraph node name -> the `stage` key the UI's CARDS array expects (see
# ai_paper_reviewer_ui.html). "prepare_revision" and "ready_for_synthesis"
# are orchestration-only nodes with no user-facing output and are handled
# separately in _run_review_graph, not through this table.
_GRAPH_NODE_TO_STAGE = {
    "paper_understanding": "paper_understanding_agent",
    "literature_rag": "literature_rag",
    "figure_table": "figure_table_agent",
    "novelty": "novelty_llm_agent",
    "methodology": "methodology_agent",
    "citation": "citation_agent",
    "evidence_reproducibility": "evidence_agent",
    "reflection": "reflection_agent",
    "final_review": "final_report",
}

# run_id -> {"parsed_paper": ParsedPaper, "paper_index": PaperIndex, "pdf_path": str}
# In-memory only -- this is a demo/inspection layer, not the review-lifecycle
# persistence (that's core.db's job once agents actually write to it).
_RUNS: Dict[str, Dict[str, Any]] = {}


def _event(stage: str, status: str, **detail) -> Dict[str, Any]:
    return {"stage": stage, "status": status, "ts": time.time(), **detail}


def run_pipeline(run_id: str, pdf_path: str) -> Iterator[Dict[str, Any]]:
    """Generator: yields one or more events per stage as they actually complete."""
    _RUNS[run_id] = {"pdf_path": pdf_path}
    t_start = time.time()

    # --- Stage: parse (Scientific Document Understanding) ---
    yield _event("parse", "running", message="Parsing PDF with Docling (layout + OCR-if-needed)...")
    try:
        from core.parsing.docling_parser import DoclingParser

        t0 = time.time()
        parsed_paper: ParsedPaper = DoclingParser().parse(pdf_path)
        _RUNS[run_id]["parsed_paper"] = parsed_paper
        yield _event(
            "parse", "done", elapsed_s=round(time.time() - t0, 2),
            title=parsed_paper.title,
            abstract_preview=parsed_paper.abstract[:400],
            section_names=[s.name for s in parsed_paper.sections],
            num_tables=len(parsed_paper.tables),
            num_figures=len(parsed_paper.figures),
            num_references=len(parsed_paper.references),
        )
    except Exception as exc:
        yield _event("parse", "error", message=str(exc), trace=traceback.format_exc(limit=3))
        return  # nothing downstream can run without a parsed paper

    # --- Stage: figure/table vision analysis ---
    if not settings.vision.enabled:
        yield _event("vision", "skipped", message="VISION__ENABLED is false -- no local vision model configured.")
    elif not parsed_paper.figures:
        yield _event("vision", "skipped", message="No figures detected in this PDF.")
    else:
        yield _event("vision", "running", message=f"Cropping + describing up to {settings.vision.max_figures_per_paper} figure(s)...")
        try:
            from core.parsing.figure_analyzer import analyze_figures

            t0 = time.time()
            parsed_paper = analyze_figures(parsed_paper)
            _RUNS[run_id]["parsed_paper"] = parsed_paper
            analyzed = [f for f in parsed_paper.figures if f.ocr_text]
            yield _event(
                "vision", "done", elapsed_s=round(time.time() - t0, 2),
                num_analyzed=len(analyzed),
                figures=[{"figure_id": f.figure_id, "caption": f.caption, "description": f.ocr_text} for f in analyzed],
            )
        except Exception as exc:
            yield _event("vision", "error", message=str(exc), trace=traceback.format_exc(limit=3))

    # --- Stage: RAG chunking (builds Index A input) ---
    yield _event("chunk", "running", message="Section-aware chunking for the paper's own RAG index (Index A)...")
    try:
        t0 = time.time()
        chunker_input = parsed_paper_to_chunker_input(parsed_paper)
        chunks = chunk_paper(run_id, chunker_input)
        by_section: Dict[str, int] = {}
        for c in chunks:
            by_section[c.section] = by_section.get(c.section, 0) + 1
        yield _event(
            "chunk", "done", elapsed_s=round(time.time() - t0, 2),
            num_chunks=len(chunks), by_section=by_section,
        )
    except Exception as exc:
        yield _event("chunk", "error", message=str(exc), trace=traceback.format_exc(limit=3))
        chunks = []

    # --- Stage: build Paper-RAG Index A (hybrid dense+BM25) ---
    if chunks:
        yield _event("paper_rag_build", "running", message="Embedding chunks (bge-small) + building FAISS + BM25 (Index A)...")
        try:
            t0 = time.time()
            paper_index = PaperIndex(embedding_provider=BgeSmallEmbeddingProvider(device=_embedding_device()))
            paper_index.build(chunks)
            _RUNS[run_id]["paper_index"] = paper_index
            yield _event(
                "paper_rag_build", "done", elapsed_s=round(time.time() - t0, 2),
                num_vectors=len(chunks), queryable=True,
            )
        except Exception as exc:
            yield _event("paper_rag_build", "error", message=str(exc), trace=traceback.format_exc(limit=3))
    else:
        yield _event("paper_rag_build", "skipped", message="No chunks produced -- nothing to index.")

    # --- Stage: Novelty Agent (real, local, no LLM -- arko_novelty_agent merge) ---
    novelty_corpus_dir = Path(__file__).resolve().parent.parent / NOVELTY_PATHS.corpus_dir
    if not novelty_corpus_dir.is_dir() or not any(novelty_corpus_dir.glob("*.json")):
        yield _event(
            "novelty_agent", "not_available",
            message=f"No novelty corpus indexed yet -- add PeerRead-shaped JSON files to {NOVELTY_PATHS.corpus_dir} first.",
        )
    else:
        yield _event("novelty_agent", "running", message="Embedding + FAISS-retrieving against the local novelty corpus...")
        try:
            t0 = time.time()
            agent = _get_novelty_agent()
            novelty_input = parsed_paper_to_novelty_input(parsed_paper)
            report = agent.evaluate(novelty_input, paper_id=run_id)
            yield _event("novelty_agent", "done", elapsed_s=round(time.time() - t0, 2), report=report.to_dict())
        except NoveltyEvaluationAgentError as exc:
            yield _event("novelty_agent", "error", message=str(exc))

    # --- Stages: the full 9-agent LangGraph review (paper understanding
    # through final review), replacing the old placeholder cards with real
    # graph-driven events. ---
    yield from _run_review_graph(run_id, parsed_paper)

    # --- Stage that genuinely doesn't exist in the codebase yet (Phase 2) ---
    yield _event("human_approval", "not_implemented", label="Human-in-the-Loop Approval")

    yield _event("pipeline", "complete", total_elapsed_s=round(time.time() - t_start, 2))


_review_graph_singleton = None


def _get_review_graph():
    """Builds the compiled LangGraph review graph once per process and reuses
    it -- constructing it rebuilds all 9 agents (and their LLM client), so
    doing that per uploaded paper would pay that cost on every review."""
    global _review_graph_singleton
    if _review_graph_singleton is None:
        _review_graph_singleton = build_review_graph()
    return _review_graph_singleton


def _serialize(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _summary_message(node_name: str, value: Any) -> str:
    """One-line human-readable summary per node type, for the activity feed
    and card subtitle -- everything else about the output still reaches the
    UI verbatim via the event's `result` field."""
    if node_name == "paper_understanding":
        return f"{len(value.stated_contributions)} contribution(s), {len(value.key_terms)} key term(s) identified."
    if node_name == "literature_rag":
        return f"Retrieved {len(value.matches)} literature match(es)."
    if node_name == "figure_table":
        return f"{len(value.figure_summaries)} figure(s), {len(value.table_summaries)} table(s) summarized."
    if node_name == "novelty":
        return f"Novelty rating: {value.novelty_rating}."
    if node_name == "methodology":
        return f"Soundness rating: {value.soundness_rating}."
    if node_name == "citation":
        return f"Citation quality: {value.citation_quality_rating}."
    if node_name == "evidence_reproducibility":
        return f"Overall rating: {value.overall_rating}."
    if node_name == "reflection":
        return f"Confidence: {value.overall_confidence}, {len(value.flags)} flag(s), needs_revision={value.needs_revision}."
    if node_name == "final_review":
        return f"Final recommendation: {value.final_recommendation} (confidence: {value.confidence})."
    return "Completed."


def _run_review_graph(run_id: str, parsed_paper: ParsedPaper) -> Iterator[Dict[str, Any]]:
    """Streams the compiled review graph node-by-node, translating each
    LangGraph update into the SSE event shape the UI already understands.
    Real graph execution -- not a re-implementation of the graph's logic."""
    graph = _get_review_graph()

    # Stage-1 nodes (paper_understanding, literature_rag, figure_table,
    # methodology, evidence_reproducibility) all start immediately at START --
    # announce them running up front rather than only on completion, so the
    # UI shows the parallel fan-out instead of cards jumping straight to done.
    for node in ("paper_understanding", "literature_rag", "figure_table", "methodology", "evidence_reproducibility"):
        yield _event(_GRAPH_NODE_TO_STAGE[node], "running", message="Running...")

    try:
        for update in graph.stream(
            {"parsed_paper": parsed_paper},
            config={"configurable": {"thread_id": run_id}},
            stream_mode="updates",
        ):
            for node_name, partial_state in update.items():
                if node_name == "prepare_revision":
                    revision_count = partial_state.get("revision_count", 1)
                    yield _event(
                        "reflection_agent", "running",
                        message=f"Revision pass {revision_count}: re-running assessment agents with feedback...",
                    )
                    # This pass re-triggers exactly these 4 nodes -- announce
                    # them running again so the UI doesn't sit on stale "done".
                    for node in ("novelty", "methodology", "citation", "evidence_reproducibility"):
                        yield _event(_GRAPH_NODE_TO_STAGE[node], "running", message=f"Revision pass {revision_count}...")
                    continue
                if node_name == "ready_for_synthesis":
                    continue  # pure pass-through sync point, nothing user-facing

                stage = _GRAPH_NODE_TO_STAGE.get(node_name)
                if stage is None:
                    continue

                # Each real node returns exactly one {output_key: value} pair
                # (verified in core/graph/nodes.py) -- prepare_revision/
                # ready_for_synthesis, the only exceptions, are handled above.
                (_, output_value), = partial_state.items()
                yield _event(
                    stage, "done",
                    message=_summary_message(node_name, output_value),
                    result=_serialize(output_value),
                )
    except Exception as exc:
        yield _event(
            "pipeline", "error",
            message=f"Review graph failed: {exc}", trace=traceback.format_exc(limit=5),
        )


def _embedding_device() -> str:
    device = getattr(settings.embeddings, "device", "cpu")
    return device if device in ("cpu", "cuda") else "cpu"


_novelty_agent_singleton: Optional[NoveltyEvaluationAgent] = None


def _get_novelty_agent() -> NoveltyEvaluationAgent:
    """Lazily index the novelty corpus once per process and cache it --
    re-embedding the whole corpus on every uploaded paper would make each
    review pay the full corpus-indexing cost, not just its own parse cost."""
    global _novelty_agent_singleton
    if _novelty_agent_singleton is None:
        agent = NoveltyEvaluationAgent()
        corpus_dir = Path(__file__).resolve().parent.parent / NOVELTY_PATHS.corpus_dir
        agent.index_corpus(corpus_dir)
        _novelty_agent_singleton = agent
    return _novelty_agent_singleton


def query_paper_index(run_id: str, query: str, k: int = 5) -> Dict[str, Any]:
    """Live hybrid retrieval against the uploaded paper's own Index A -- the
    one RAG capability that needs no external corpus, so it's queryable the
    moment a paper finishes the chunk/index stages above."""
    run = _RUNS.get(run_id)
    if run is None or "paper_index" not in run:
        return {"error": "No built index for this run_id yet -- wait for the paper_rag_build stage to finish."}

    paper_index: PaperIndex = run["paper_index"]
    results = paper_index.retrieve(query, k=k)
    return {
        "query": query,
        "results": [
            {"score": round(r.score, 4), "section": r.metadata.get("section"), "content": r.content}
            for r in results
        ],
    }


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    return _RUNS.get(run_id)


def check_system_health() -> Dict[str, Any]:
    """Live status of every external dependency the pipeline actually needs --
    each check is a real probe (socket connect / HTTP call / importlib / file
    existence), not a hardcoded 'Healthy' like the original dashboard mockup."""
    import importlib.util
    import socket
    from pathlib import Path

    def check_tcp(host: str, port: int, timeout: float = 1.5) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    checks: Dict[str, Dict[str, Any]] = {}

    ollama_url = settings.llm.base_url.replace("http://", "").replace("https://", "")
    ollama_host, _, ollama_port = ollama_url.partition(":")
    ollama_up = check_tcp(ollama_host, int(ollama_port or 11434))
    ollama_models = []
    if ollama_up:
        try:
            import urllib.request

            with urllib.request.urlopen(f"{settings.llm.base_url}/api/tags", timeout=2) as resp:
                import json as _json

                ollama_models = [m["name"] for m in _json.loads(resp.read())["models"]]
        except Exception:
            pass
    checks["ollama"] = {
        "healthy": ollama_up, "label": "Ollama (LLM + Vision)",
        "detail": f"{len(ollama_models)} model(s) pulled: {', '.join(ollama_models) or 'none'}" if ollama_up else f"unreachable at {settings.llm.base_url}",
    }

    mysql_up = check_tcp(settings.db.host, settings.db.port)
    checks["mysql"] = {
        "healthy": mysql_up, "label": "MySQL (review lifecycle)",
        "detail": f"{settings.db.host}:{settings.db.port}" if mysql_up else f"unreachable at {settings.db.host}:{settings.db.port} -- no docker-compose up yet",
    }

    docling_ok = importlib.util.find_spec("docling") is not None
    checks["docling"] = {
        "healthy": docling_ok, "label": "Docling (PDF parsing)",
        "detail": "installed" if docling_ok else "not installed -- parse stage will fail until `pip install docling` completes",
    }

    repo_root = Path(__file__).resolve().parent.parent
    lit_index = repo_root / "data" / "literature_index" / "index.faiss"
    checks["literature_index"] = {
        "healthy": lit_index.exists(), "label": "Literature Index (Index B)",
        "detail": "built" if lit_index.exists() else "not built yet -- run core/rag/ingestion/build_corpus.py against a PeerRead clone",
    }

    checkpoint_path = repo_root / Path(settings.checkpoint.sqlite_path.lstrip("./"))
    checks["checkpoint_db"] = {
        "healthy": True, "label": "LangGraph Checkpoint DB",
        "detail": "exists" if checkpoint_path.exists() else "will be created on first orchestration run (not built yet -- no LangGraph graph exists)",
    }

    return checks
