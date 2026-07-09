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
`MULTI_AGENT_SYSTEM.md`, `UI_WORK.md`, `PEERREAD_CORPUS_MODULE.md`. Condensed summary:

| Area                                                        | Status                                                                                                      |
| ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| PDF parsing (Docling)                                       | ✅ Real, tested on real PDFs                                                                                 |
| RAG — paper's own index (hybrid dense+BM25)                 | ✅ Real, tested, live-queryable                                                                              |
| RAG — persistent literature corpus                          | ✅ Real, real ICLR-2017 data indexed (389 papers, train+dev) — see `PEERREAD_CORPUS_MODULE.md`               |
| Figure/table vision analysis                                | ✅ Real — vision model pulled (`qwen2.5vl:7b`) and `VISION__ENABLED=true`, verified end-to-end on GPU        |
| 9 review agents (incl. 2 novelty implementations)           | ✅ All real, tested individually and together                                                                |
| LangGraph orchestration                                     | ✅ Built + verified (mocked topology test, a real ~20min Ollama run, AND now live through the dashboard) — see `LANGGRAPH_ORCHESTRATION.md` / `UI_WORK.md` |
| Live SSE dashboard                                          | ✅ Real, working — all 9 agents + Final Review/Human Approval views render real graph output, no placeholder cards left except `human_approval` (Phase 2) |
| Human-in-the-loop approval                                  | ❌ Not started (view shows the real final review; Approve/Reject clicks are honestly inert, no persistence)  |
| Review persistence (MySQL)                                  | ❌ Schema exists, nothing writes to it; MySQL isn't even running                                             |
| **Evaluation harness (PeerRead test split, accuracy/F1/κ)** | ✅ Built (`core/eval/peerread_harness.py`, `scripts/run_peerread_evaluation.py`) — see run results below     |

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

1. ~~**Wire the LangGraph orchestration into `server/pipeline.py`**~~ — **Done.**
   `server/pipeline.py` streams the compiled graph node-by-node
   (`stream_mode="updates"`) into real SSE events; the dashboard shows all 9
   agents plus a live-rendered Final Review / Human Approval view instead of
   placeholder cards. See `UI_WORK.md`.
2. **Human-in-the-loop + persistence** — the one remaining big piece.
   LangGraph interrupt/resume before final output; wire graph nodes to
   actually write to MySQL (needs MySQL running — no local
   `docker-compose.yml` exists; create one or get it from a teammate). The
   Human Approval view already renders the real final review; its
   Approve/Request Changes/Reject buttons are wired but honestly inert (no
   backend to persist to yet).
3. ~~**The evaluation harness (the graded core)**~~ — **Built.**
   `core/eval/peerread_harness.py` + `scripts/run_peerread_evaluation.py`
   load PeerRead's labeled `test` split (`data/peerread_raw/iclr_2017/test/`,
   38 papers, held out of both corpora — see `PEERREAD_CORPUS_MODULE.md`),
   parse each paper's real PDF with Docling, run it through the full graph,
   map `final_recommendation` to accept/reject (`accept`/`weak_accept`→
   accept; `borderline`/`weak_reject`/`reject`→reject), and score against
   ground truth with accuracy/F1/Cohen's κ via scikit-learn. Results are
   written to `output_results/peerread_eval.{jsonl,metrics.json}`
   (gitignored — regenerate with the script above).

   **Two real bugs the first full run surfaced, both fixed in
   `core/config/prompts.yaml`:** (a) `methodology_agent` and
   `evidence_reproducibility_agent`'s JSON-shape examples showed a
   `poor/fair/good/excellent`-vocabulary value directly next to an
   `adequate/weak/missing`-vocabulary field in the same example object —
   the model kept bleeding "good" into the wrong field (32/38 runs failed
   schema validation before the fix; 35/38 succeeded after, plus bumping
   `invoke_for_json`'s retry cap 2→3). (b) `final_review_agent`'s shape
   example showed `"final_recommendation": "borderline"` as a literal
   example value — the model anchored on it and returned "borderline" for
   **all 35** genuinely different papers (`f1`/`cohen_kappa` both exactly
   0.0, `predicted_accept_rate` exactly 0.0 despite 34% of papers actually
   being accepted — a dead giveaway of prompt-anchoring, not real review
   variance). Fixed by replacing the literal example with a placeholder
   plus an explicit decision rubric tied to the upstream agents' actual
   ratings; verified on a 6-paper smoke test that recommendations now
   genuinely vary (`weak_reject`/`weak_accept` mixed, no repeats). A full
   clean 38-paper re-run with both fixes was in progress as this was
   written — run it yourself and check `output_results/peerread_eval.metrics.json`
   for current numbers; don't trust any number pasted here as still current,
   per this file's own §5 rule.
4. **Data** — done for `iclr_2017`: `data/peerread_raw/` cloned (reviews +
   the 38 `test`-split PDFs; train/dev PDFs skipped, not needed),
   `core/rag/ingestion/build_corpus.py` run (389-paper literature index) and
   `data/novelty_corpus/` populated with the same 389 real papers (see
   `PEERREAD_CORPUS_MODULE.md`). Still open: the `acl_2017` venue (listed
   in `IngestionSettings.peerread_venues` but never fetched — the graded core
   only needs ICLR-2017, so this was deliberately skipped).

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
