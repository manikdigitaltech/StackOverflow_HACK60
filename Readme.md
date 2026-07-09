# Autonomous AI Paper Reviewer & Scientific Evaluation Agent

A multi-agent system that ingests a research paper PDF and produces a
structured, evidence-backed peer review — parsing, RAG-grounded literature
comparison, novelty/methodology/citation/evidence assessment, self-reflection,
and a final structured recommendation. Fully local: Ollama for LLM inference,
FAISS for retrieval, no cloud API calls in the core pipeline.

***
Please create new branches with your name
***

## Prerequisites

- **Python 3.11+**
- **[Ollama](https://ollama.com)** installed and running locally, with at least
  one text model pulled:
  ```bash
  ollama pull qwen2.5:7b
  ```
  (Optional, for figure/table vision analysis — off by default:
  `ollama pull qwen2.5vl:7b`)
- **MySQL** — only needed for the human-in-the-loop / review-persistence layer
  (not required to run the live pipeline demo below). No `docker-compose.yml`
  is checked in; stand one up yourself if you're working on that layer.

## Setup

```bash
pip install -r requirements.txt
cp env.example .env   # adjust DB/LLM/vision settings as needed
```

## Running the live pipeline UI

The primary way to see the system working end to end: a FastAPI server that
streams every real pipeline stage live (parse → figure/table vision →
RAG indexing → literature retrieval → agents) to a browser dashboard over
Server-Sent Events.

```bash
python -m uvicorn server.main:app --reload --port 8000
```

Open `http://localhost:8000/`, upload a PDF, and watch it run. Stages that
don't exist yet in the codebase are shown honestly as "not yet implemented,"
never faked.

## Running the orchestration graph directly

The LangGraph review pipeline (`core/graph/`) wires the 9 review agents into
one bounded run (parallel fan-out, a self-reflection step with one bounded
revision pass, final synthesis). To exercise it directly against a real
parsed paper and local Ollama:

```python
from core.graph.build_graph import build_review_graph
from core.parsing.docling_parser import DoclingParser

parsed_paper = DoclingParser().parse("./data/raw_papers/your_paper.pdf")
graph = build_review_graph()
result = graph.invoke(
    {"parsed_paper": parsed_paper},
    config={"configurable": {"thread_id": "some-run-id"}},
)
print(result["final_review"])
```

A fast, fully-mocked structural test of the graph's topology (no LLM calls,
runs in milliseconds) lives at `scripts/test_graph_topology.py`:

```bash
python -m scripts.test_graph_topology
```

## Running tests

```bash
pytest tests/unit -v
```

Individual manual verification scripts for specific pieces (parsing, RAG,
each agent, the novelty pipeline, etc.) live under `scripts/test_*.py` — run
any of them directly, e.g.:

```bash
python -m scripts.test_parsing "./data/raw_papers/your_paper.pdf"
python -m scripts.test_novelty_agent
```

## Project structure

```
core/
  parsing/       Docling PDF parsing, section segmentation, figure/table extraction
  rag/           Two-index retrieval: Index A (per-paper hybrid dense+BM25),
                 Index B (persistent literature corpus)
  agents/        The 9 review agents (paper understanding, literature RAG,
                 novelty, methodology, citation, evidence/reproducibility,
                 figure/table, reflection, final review) + a separate
                 embedding-only novelty scorer (core/agents/novelty/)
  graph/         LangGraph orchestration wiring the agents into one review run
  llm/           Ollama client factory, prompt manager, structured-output helper
  db/            SQLAlchemy models + Alembic migration for review persistence
  config/        Settings (env-overridable) and prompt templates
server/          FastAPI backend for the live pipeline dashboard (SSE)
scripts/         Manual verification scripts and CLI entry points
tests/unit/      Automated pytest suite
data/            Sample PDFs, literature/novelty corpora (seed data only —
                 real PeerRead data needs to be cloned/built separately)
```

## Current status

- **Working, verified**: PDF parsing, both RAG indexes, all 9 review agents
  (individually and via the orchestration graph), the live SSE dashboard.
- **Not yet built**: human-in-the-loop approval + review persistence, the
  PeerRead evaluation harness (accuracy/F1/Cohen's κ against ground truth —
  the graded core of the problem statement), a real PeerRead-built literature
  corpus (only seed data exists today).
- Figure/table vision analysis is code-complete but off by default (no local
  vision model pulled).
