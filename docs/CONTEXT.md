# Autonomous AI Paper Reviewer

*Read this first in a new session, on a new machine, before touching
anything. It's self-contained: assumes only that this repo has been
cloned/copied to the new computer, nothing about the prior machine or chat
history. Deep-dive docs referenced throughout live at the repo root
alongside this file — read them for implementation detail; this file is the
map, not the territory.*

## 1. What this project is

A hackathon team (branch owners: manik, kanishka, arko, saumya, vivek,
tamanna) building an **Autonomous AI Paper Reviewer & Scientific Evaluation
Agent** per a hackathon problem statement. Repo:
`manikdigitaltech/StackOverflow_HACK60`.

**Required architecture** (per the problem statement): parse a submitted PDF
→ multi-agent controller (Novelty/Methodology/Evidence/Citation agents) →
RAG-grounded literature comparison → self-reflection/verifier step →
human-in-the-loop approval → structured final report (summary, strengths,
weaknesses, questions for authors, 5-way rating, confidence, justification).
**Graded core**: evaluate against PeerRead's ICLR-2017 subset, using its
built-in 80/10/10 split (never invent your own split), reporting accuracy /
F1 / Cohen's κ on the test set only. Constraints: fully local (Ollama), 24GB
GPU budget. Streamlit was the *suggested* UI tool but the problem statement
explicitly permits substitution — this project built a custom FastAPI+SSE
dashboard instead (rationale in `UI_WORK.md`).

## 2. Environment setup on the new machine

```bash
pip install -r requirements.txt
cp env.example .env          # adjust as needed
ollama pull qwen2.5:7b       # text model, required
# optional, for figure/table vision analysis (off by default):
ollama pull qwen2.5vl:7b
```

**Platform gotchas hit on the original (Windows) machine — may not apply on
yours, but worth knowing:**
- Use `py`, not `python`, if on Windows without `python` on PATH.
- Git Bash on Windows mangles colons in paths like
  `git show branch:path/to/file` — the shell auto-converts `/` to `\` and
  `:` to `;`, breaking the command. Fix: prefix with
  `MSYS_NO_PATHCONV=1 git show "branch:path"`, or just use PowerShell instead.
- `/tmp/...` paths don't resolve correctly in Git Bash on Windows either —
  use an explicit Windows-style scratch path instead.

**Verify the setup works:**
```bash
pytest tests/unit -v                          # 43 tests, should all pass
python -m scripts.test_graph_topology          # graph structure, mocked, fast
python -m uvicorn server.main:app --reload --port 8000   # then open localhost:8000
```

## 3. Current state — what's real and verified

Detailed docs for each area are at the repo root: `RAG_ARCHITECTURE.md`,
`PARSING_ARCHITECTURE.md`, `NOVELTY_AGENT.md`, `AGENTS_ARCHITECTURE.md`,
`VLM_FIGURE_TABLE_ANALYSIS.md`, `LANGGRAPH_ORCHESTRATION.md`,
`MULTI_AGENT_SYSTEM.md`, `UI_WORK.md`. Condensed summary:

| Area                                                        | Status                                                                                                      |
| ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| PDF parsing (Docling)                                       | ✅ Real, tested on real PDFs                                                                                 |
| RAG — paper's own index (hybrid dense+BM25)                 | ✅ Real, tested, live-queryable                                                                              |
| RAG — persistent literature corpus                          | ✅ Code real & tested; **no PeerRead data indexed yet**                                                      |
| Figure/table vision analysis                                | ✅ Code real & tested (mocked); **no vision model pulled**                                                   |
| 9 review agents (incl. 2 novelty implementations)           | ✅ All real, tested individually and together                                                                |
| LangGraph orchestration                                     | ✅ Built + verified (mocked topology test AND a real ~20min Ollama run) — **not yet wired into the live UI** |
| Live SSE dashboard                                          | ✅ Real, working — shows placeholder cards for agents not yet wired into it                                  |
| Human-in-the-loop approval                                  | ❌ Not started                                                                                               |
| Review persistence (MySQL)                                  | ❌ Schema exists, nothing writes to it; MySQL isn't even running                                             |
| **Evaluation harness (PeerRead test split, accuracy/F1/κ)** | ❌ **Not started — this is the graded core**                                                                 |

## 4. Branch history — what was merged, what wasn't, and why

All of the following branches share **no commit history with each other**
except a common bare initial commit (`9447376`/`79f586a`) — except
`tamanna-kanwar-local-llm-fine-tuning`, which has completely separate history.

| Branch                                            | Verdict                    | Why                                                                                                                                                                                                                                |
| ------------------------------------------------- | -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `manik`                                           | **Trunk**                  | The only branch spanning the whole architecture (parsing, DB schema, LLM plumbing)                                                                                                                                                 |
| `kanishka-RAG-phase1`                             | **Merged**                 | Superior 2-index RAG design (hybrid dense+BM25 + persistent specter2 corpus), fully tested — replaced manik's original single-index retriever entirely                                                                             |
| `arko_novelty_agent`                              | **Merged**                 | Complete, 19/19-tested embedding-based novelty scorer — moved to `core/agents/novelty/`                                                                                                                                            |
| `saumya-VLM`                                      | **Superseded, not merged** | Cloud Gemini API prototype; the *idea* (vision analysis) was rebuilt properly, locally, from scratch as `core/parsing/figure_analyzer.py`                                                                                          |
| `vivek-RAGAS-deepeval`                            | **Shelved, not merged**    | Right idea (DeepEval for agent quality gates, RAGAS for RAG faithfulness) but premature — nothing existed yet for it to evaluate. Good template once the eval harness exists.                                                      |
| `tamanna-kanwar-local-llm-fine-tuning`            | **Shelved, not merged**    | Careful data-hygiene work (explicitly avoids fine-tuning on PeerRead, the mandatory eval set) but unintegrated (targets a toy `distilgpt2`, no path to serve a fine-tuned model through `get_llm()`) and not required by the brief |
| **manik's own follow-up push** (commit `94459cb`) | **Merged**                 | Added the entire missing agent layer: 9 agents + `BaseAgent` + structured-output helper + review-output schemas. The single highest-value push evaluated this session.                                                             |

**One real integration conflict, resolved**: manik's `literature_rag_agent.py`
imported the retired single-index retriever. Rewritten against kanishka's
`LiteratureIndex` — see `AGENTS_ARCHITECTURE.md`.

## 5. Check git state fresh — do not trust any snapshot below

As of the end of the prior session: on branch `manik`, local HEAD was 1
commit behind `origin/manik`. A large amount of work was **uncommitted** in
the working tree (everything in §3/§4 above). The user was guided through
creating a new branch and pushing — **run `git status`, `git branch -a`, and
`git log --oneline -5` now** to see what actually happened since; don't
assume anything about commit/push state from this document.

## 6. Non-obvious gotchas worth knowing before you touch the code

- **LangGraph fan-in**: `add_edge(a, "x")` + `add_edge(b, "x")` called
  separately are *independent OR-triggers*, not an AND-join. A true "wait for
  all of these" join needs `add_edge([a, b], "x")` — all sources in one list,
  one call. Two real bugs from getting this wrong are documented in detail
  in `LANGGRAPH_ORCHESTRATION.md` — read it before modifying `core/graph/build_graph.py`.
- **Ollama `json_mode` default**: `get_llm()` now defaults to
  `json_mode=True` (forces `format="json"`). Any *free-text* prompt (like
  the `connectivity_test` template) must explicitly pass `json_mode=False`,
  or Ollama will reject/mangle the response.
- **PeerRead has two different raw shapes in play**: `parsed_pdfs/*.json`
  (full text, science-parse format — what manik's original loader used) vs.
  `reviews/*.json` (title/abstract/`accepted` label only — what kanishka's
  `build_corpus.py` uses, and what the eval harness will need for ground
  truth). Don't confuse them.
- **`LiteratureMatch.paper_id` is `str`, not `int`** — changed during the RAG
  merge because kanishka's corpus uses venue-prefixed string ids
  (`"iclr_2017:304"`). Confirmed nothing else in the codebase read it as an
  int before changing it.
- **Revision loop needs real feedback wiring, not just a structural loop** —
  `core/agents/revision.py` + the `{revision_feedback}` prompt placeholder in
  4 of the agents' `prompts.yaml` templates is what makes a revision pass
  actually different from an identical re-run. If you add a 5th agent to the
  revision loop, it needs the same treatment.
- **`.gitignore` now covers `__pycache__`/`.pytest_cache`/etc.** — manik's
  original repo had *committed* `.pyc` files (compiled for a different
  Python version than what's likely on your machine); these were cleaned up
  during the merge.

## 7. What's next (dependency order)

1. **Wire the LangGraph orchestration into `server/pipeline.py`** — replace
   the remaining placeholder "not yet implemented" cards in the live
   dashboard with real graph-driven SSE events. Mechanical, not risky.
2. **Human-in-the-loop + persistence** — LangGraph interrupt/resume before
   final output; wire graph nodes to actually write to MySQL (needs MySQL
   running — no local `docker-compose.yml` exists; create one or get it from
   a teammate).
3. **The evaluation harness (the graded core)** — load PeerRead's labeled
   test split (`reviews/*.json`, has the `accepted` field), run the full
   graph per test paper, map `final_recommendation` to accept/reject, compute
   accuracy/F1/Cohen's κ against ground truth. Needs a real PeerRead clone.
4. **Data**: clone PeerRead, run `core/rag/ingestion/build_corpus.py`
   (builds the real literature index) and consider building a real
   `data/novelty_corpus/` (currently only 2 toy seed papers). Optionally pull
   a vision model if live figure analysis should actually run.

## 8. Where everything lives (quick map)

```
core/parsing/     Docling parsing, section segmentation, figure cropping, context building
core/rag/         Two-index retrieval (paper's own + persistent literature corpus)
core/agents/      The 9 review agents + shared base/structured-output infra
                  core/agents/novelty/  <- the separate embedding-based novelty scorer
core/graph/       LangGraph orchestration (state.py, nodes.py, build_graph.py)
core/llm/         Ollama client factories (text + vision), prompt manager, structured output
core/db/          SQLAlchemy models + migration (schema ready, unused until Phase 2)
core/config/      Settings (env-overridable) + prompts.yaml
server/           FastAPI backend for the live SSE dashboard
ai_paper_reviewer_ui.html   The dashboard itself
scripts/          Manual verification scripts + CLI entry points
tests/unit/       Automated pytest suite (43 tests)
data/             Sample PDFs + seed corpora (real PeerRead data not included)
```
