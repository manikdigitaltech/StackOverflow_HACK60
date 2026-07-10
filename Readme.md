# Autonomous AI Paper Reviewer & Scientific Evaluation Agent

A multi-agent system that ingests a research paper PDF and produces a
structured, evidence-backed peer review - parsing, RAG-grounded literature
comparison, novelty/methodology/citation/evidence assessment, self-reflection,
and a final structured recommendation. Fully local: Ollama for LLM inference,
FAISS for retrieval, no cloud API calls in the core pipeline.

***
Please create new branches with your name
***

## Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.com)** installed and running locally, with at least
  one text model pulled. Check `LLM__PROVIDER` in your `.env` and pull the
  matching tag from `core/llm/llm_provider.py`'s `_OLLAMA_MODEL_MAP` (e.g.
  `qwen2.5-7b` → `qwen2.5:7b-instruct-q8_0`):
  ```bash
  ollama pull qwen2.5:7b-instruct-q8_0
  ```
  Optional, for figure/table vision analysis (`VISION__ENABLED=true`):
  `ollama pull qwen2.5vl:7b`
- **GPU strongly recommended, not required.** If a CUDA GPU is available,
  make sure `torch` is installed with a build matching your driver's CUDA
  version (`torch.cuda.is_available()` should return `True`) and set
  `EMBEDDINGS__DEVICE=cuda` in `.env` - otherwise every embedding call
  (RAG indexing, the novelty scorer) silently runs on CPU even with a GPU
  present, since nothing auto-detects this.
- **MySQL**, for the human-in-the-loop / review-persistence layer. Real and
  running matters here - reviews are actually written to it now, not just
  schema-ready. No `docker-compose.yml` is checked in; the settings default
  to `localhost:3307` / db `paper_reviewer` / user `reviewer_app` (see
  `env.example`) - point those at a real instance, or stand one up yourself
  (`mysqld --datadir=... --port=3307 ...`, then run the Alembic migration in
  `core/db/migrations/`). The live dashboard's System Health panel
  (`GET /api/health`) tells you honestly whether it's reachable.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp env.example .env   # adjust DB/LLM/vision/embeddings settings as needed
```

## Running the live pipeline UI

The primary way to see the system working end to end: a FastAPI server that
streams every real pipeline stage live (parse → vision → RAG indexing →
all 10 review agents → final review) to a browser dashboard over
Server-Sent Events, and persists every run to MySQL as it goes.

```bash
python -m uvicorn server.main:app --reload --port 8000
```

Open `http://localhost:8000/`, upload a PDF, and watch it run. The review
run genuinely pauses mid-graph at the human-approval gate (a real LangGraph
`interrupt()`, not a post-hoc "review the finished result" step) until you
approve, reject, or revise it from the dashboard - see `docs/CONTEXT.md` §7
item 2. Past runs are browsable in the dashboard's History tab; system
dependency health (Ollama, MySQL, Docling, literature index, checkpoint DB)
is a live panel, not a hardcoded "Healthy."

## Running the orchestration graph directly

The LangGraph review pipeline (`core/graph/`) wires 10 review agents into
one bounded run (parallel fan-out, an Adversarial Critic attacking
Methodology/Citation/Evidence's verdicts, a self-reflection step with one
bounded revision pass, final synthesis). To exercise it directly against a
real parsed paper and local Ollama:

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
each agent, the novelty pipeline, etc.) live under `scripts/test_*.py` - run
any of them directly, e.g.:

```bash
python -m scripts.test_parsing "./data/raw_papers/your_paper.pdf"
python -m scripts.test_novelty_agent
```

## Running the graded evaluation (PeerRead accuracy/F1/κ)

The problem statement's graded core: runs the full graph against every
paper in PeerRead ICLR-2017's held-out `test` split (never touched by any
corpus/index - see `docs/PEERREAD_CORPUS_MODULE.md`), maps the recommendation
to accept/reject, and scores against real ground truth.

```bash
python -m scripts.run_peerread_evaluation --output output_results/peerread_eval.jsonl
```

Real PeerRead data needs to be present first (`data/peerread_raw/iclr_2017/`
— see `docs/PEERREAD_CORPUS_MODULE.md` for how it was fetched). Current
numbers are in `docs/CONTEXT.md` §7 item 3 - re-run and don't trust numbers
in docs as permanently current. Optional, slower, complementary quality
signals (does an agent's *reasoning* hold up, not just its final call) are
in `docs/QUALITY_GATES.md`.

## Project structure

```
core/
  parsing/       Docling PDF parsing, section segmentation, figure/table extraction,
                 prompt-injection guardrails on every extracted text field
  rag/           Two-index retrieval: Index A (per-paper hybrid dense+BM25),
                 Index B (persistent literature corpus, real ICLR-2017 data)
  agents/        10 review agents (paper understanding, literature RAG,
                 novelty, methodology, citation, evidence/reproducibility,
                 figure/table, adversarial critic, reflection, final review)
                 + a separate embedding-only novelty scorer (core/agents/novelty/)
  graph/         LangGraph orchestration wiring the agents into one review run
  llm/           Ollama client factory, prompt manager, structured-output helper
  db/            SQLAlchemy models + repositories + Alembic migration - real,
                 live MySQL persistence, not just schema
  eval/          PeerRead accuracy/F1/κ harness + optional DeepEval/RAGAS quality gates
  utils/         Prompt-injection guardrails, grounding checks, token budgeting
  config/        Settings (env-overridable) and prompt templates
server/          FastAPI backend for the live pipeline dashboard (SSE), persistence,
                 approval/history/health endpoints
scripts/         Manual verification scripts, CLI entry points, the eval harness
tests/unit/      Automated pytest suite
data/            Sample PDFs + real ICLR-2017 PeerRead data (gitignored, see
                 docs/PEERREAD_CORPUS_MODULE.md to regenerate)
```

## Current status

See `docs/CONTEXT.md` for the full, actively-maintained picture (it
explicitly warns not to trust any snapshot as permanently current - check
git state yourself) - condensed here:

- **Working, verified, real data**: PDF parsing (with guardrails), both RAG
  indexes (the literature index holds real ICLR-2017 papers, not seed data,
  optionally augmented live by arXiv/Semantic Scholar), all 10 review agents
  individually and via the orchestration graph, the live SSE dashboard
  (every agent + Final Review + Human Approval + History + System Health),
  MySQL persistence, human-in-the-loop approval (a real LangGraph
  `interrupt()`/`Command(resume=...)` mid-graph pause, not a post-hoc
  record), and the graded PeerRead evaluation harness (real accuracy/F1/κ
  numbers, not just built-but-unrun).
- **Known open problem, being actively worked**: the eval harness's Cohen's
  κ is currently low (~0.10, "slight" agreement) - see `docs/CONTEXT.md` §7
  item 3 for the honest read and what's already been tried. The Adversarial
  Critic agent was added directly in response to this; re-running the
  harness to see whether it moved is still open.
- Figure/table vision analysis is code-complete but off by default (no local
  vision model pulled).
