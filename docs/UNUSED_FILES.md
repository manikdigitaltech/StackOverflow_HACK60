# Unused / Non-Runtime Files Audit

This file documents files that are not used by the current primary runtime:

```bash
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000
```

The current live app is the FastAPI + SSE server in `server/`, backed by the
core parsing, RAG, agent, graph, and LLM modules. Items below are grouped by
confidence so cleanup can be done without accidentally deleting planned work.

## Clearly Unused Today

These files are empty and are not imported by the FastAPI server, core graph,
tests, or manual scripts.

| File | Why it appears unused |
| --- | --- |
| `app/main.py` | Empty Streamlit app shell. The real UI is served from `server/main.py`. |
| `app/pages/1_upload_paper.py` | Empty Streamlit page stub. |
| `app/pages/2_review_dashboard.py` | Empty Streamlit page stub. |
| `app/pages/3_human_approval.py` | Empty Streamlit page stub. |
| `app/pages/4_review_history.py` | Empty Streamlit page stub. |
| `app/ui_components/review_renderer.py` | Empty Streamlit/component stub. |
| `app/ui_components/approval_widget.py` | Empty Streamlit/component stub. |
| `app/ui_components/progress_tracker.py` | Empty Streamlit/component stub. |

Context: `docs/UI_WORK.md` says the original Streamlit page stubs were never
built out and the project moved to the FastAPI + `ai_paper_reviewer_ui.html`
dashboard for real-time updates.

## Legacy / Stale Artifacts

| File | Why it appears unused |
| --- | --- |
| `data/faiss_index/literature.index` | No runtime code reads this path. The active literature index is `data/literature_index/index.faiss`; `core/agents/novelty/config.py` describes `data/faiss_index` as a retired legacy index. |
| `Generic_Review_Template_Agentic_AI.docx` | No code or docs reference this document. It may be an old planning/template artifact. |

## Not Used By The Live Server, But Keep If Planned

These are not part of the current live FastAPI demo path, but they represent
planned, optional, or manual workflows. Do not delete them unless that feature
is intentionally being dropped.

| Path | Current status |
| --- | --- |
| `core/db/repositories/*.py` | Repository layer for MySQL persistence. The live server currently keeps review/approval state in memory/checkpoints and does not write to these repositories. `scripts/test_db_roundtrip.py` uses part of this layer manually. |
| `core/db/session.py` | MySQL session factory. Used by DB scripts/repositories, not by the live review pipeline. |
| `core/db/models.py` | SQLAlchemy models for persistence. Used by migrations and repository code, not by the live server yet. |
| `core/db/migrations/*` and `alembic.ini` | Needed only when enabling MySQL persistence. |
| `core/eval/peerread_harness.py` | Evaluation harness, used by `scripts/run_peerread_evaluation.py`; not part of the live server. |
| `scripts/*.py` | Manual verification and CLI scripts. They are entrypoints, not runtime imports. Keep them for debugging, evaluation, and demos. |
| `docs/*.md` | Documentation only. Not runtime code. |
| `data/raw_papers/*.pdf` | Sample/manual test PDFs. Not required by the upload-based server flow. |

## Package Marker Files

Several `__init__.py` files are empty. They may look unused in text search, but
they are normal Python package markers and should generally be kept:

```text
core/__init__.py
core/agents/__init__.py
core/config/__init__.py
core/db/__init__.py
core/db/repositories/__init__.py
core/eval/__init__.py
core/graph/__init__.py
core/llm/__init__.py
core/rag/__init__.py
core/rag/chunking/__init__.py
core/rag/embeddings/__init__.py
core/rag/ingestion/__init__.py
core/rag/indexes/__init__.py
core/rag/live_sources/__init__.py
core/rag/retrieval/__init__.py
server/__init__.py
tests/__init__.py
tests/unit/__init__.py
```

`core/llm/chains/__init__.py` is also empty and has no sibling chain modules
today. It can be removed if no future LangChain chain package is planned.

## Confirmed Used Runtime Files

These files should not be marked unused:

| File | Used by |
| --- | --- |
| `server/main.py` | Main FastAPI app and HTTP routes. |
| `server/pipeline.py` | Upload review pipeline, SSE events, health checks, paper index queries. |
| `ai_paper_reviewer_ui.html` | Served by `GET /`. |
| `ai_paper_reviewer_saas_dashboard.html` | Served by `GET /dashboard`. |
| `core/agents/novelty_agent.py` | Used by the LangGraph review nodes. |
| `core/agents/novelty/*` | Used by the embedding-based novelty stage in `server/pipeline.py` and unit tests. |
| `data/literature_index/index.faiss` | Active persistent literature RAG index. |
| `data/literature_index/records.jsonl` | Active metadata records for the literature RAG index. |
| `data/novelty_corpus/*.json` | Corpus for the embedding-based novelty agent. |

## How This Was Checked

The audit used repository-wide text search and empty-file checks:

```bash
rg "from app|import app|streamlit|app/main|app\\.main|app/pages|core\\.db|data/faiss_index" -n .
find . -type f -empty -print
rg "data/faiss_index|FAISS__INDEX_PATH|settings\\.faiss|faiss.index_path" -n .
rg "peerread_harness|core\\.eval|ReviewRepository|PaperRepository|ChunkRepository|ApprovalRepository|ReflectionRepository" -n .
```

This is a static audit, not a proof that a file can be safely deleted. Treat
the "clearly unused today" group as deletion candidates and the optional/manual
groups as feature-boundary notes.
