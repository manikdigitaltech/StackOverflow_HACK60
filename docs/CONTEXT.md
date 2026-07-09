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
`MULTI_AGENT_SYSTEM.md`, `UI_WORK.md`, `PEERREAD_CORPUS_MODULE.md`,
`QUALITY_GATES.md`. Condensed summary:

| Area                                                        | Status                                                                                                      |
| ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| PDF parsing (Docling)                                       | ✅ Real, tested on real PDFs — now also runs prompt-injection guardrails on every text field (see `QUALITY_GATES.md`) |
| RAG — paper's own index (hybrid dense+BM25)                 | ✅ Real, tested, live-queryable                                                                              |
| RAG — persistent literature corpus                          | ✅ Real, real ICLR-2017 data indexed (389 papers, train+dev) — see `PEERREAD_CORPUS_MODULE.md`               |
| Figure/table vision analysis                                | ✅ Real — vision model pulled (`qwen2.5vl:7b`) and `VISION__ENABLED=true`, verified end-to-end on GPU        |
| 10 review agents (9 original + Adversarial Critic, incl. 2 novelty implementations) | ✅ All real, tested individually and together — Adversarial Critic attacks Methodology/Citation/Evidence's verdicts, feeds Reflection |
| LangGraph orchestration                                     | ✅ Built + verified (mocked topology test, a real ~20min Ollama run, AND now live through the dashboard) — see `LANGGRAPH_ORCHESTRATION.md` / `UI_WORK.md` |
| Live SSE dashboard                                          | ✅ Real, working — every agent + Final Review/Human Approval/History/System-Health views render real data, no placeholder cards left except `human_approval`'s true interrupt/resume (Phase 2) |
| Human-in-the-loop approval                                  | ⚠️ Partial — the view shows the real final review and `POST /api/approval/{run_id}` genuinely persists a decision; the graph still doesn't pause mid-run to wait for one (see §7 item 2) |
| Review persistence (MySQL)                                  | ✅ Real — `reviewed_papers`/`review_assessments`/`reflection_flags`/`human_approvals` all written for real, best-effort (a DB hiccup never breaks a live review); MySQL running locally on port 3307 |
| **Evaluation harness (PeerRead test split, accuracy/F1/κ)** | ✅ Built and run for real (`core/eval/peerread_harness.py`, `scripts/run_peerread_evaluation.py`) — see run results in §7 item 3 |
| Agent-output quality gates (DeepEval/RAGAS)                 | ✅ Built, both fixed from a non-working shelved branch and verified live — see `QUALITY_GATES.md`. Optional/offline, not wired into the default eval run yet |

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
| `vivek-RAGAS-deepeval`                            | **Integrated (fixed, not the original code)** | Right idea, and once the eval harness existed it was worth reviving — but none of its 3 files ran as originally written. Ported + fixed into `core/eval/deepeval_quality.py` / `ragas_quality.py` / `core/utils/guardrails.py`, see `QUALITY_GATES.md` for the exact bugs (bad local-model routing in both DeepEval and RAGAS, a wrong field reference, an environment-incompatibility import blocker in `ragas` itself, and a deprecated regex-flag pattern in guardrails). |
| `tamanna-kanwar-local-llm-fine-tuning`            | **Real leakage found; a fix + model retarget was in progress as this was last edited** | This branch's own continuation, `finetune/` (a separate nested git repo, gitignored from this project), was reviewed and found to have **confirmed PeerRead test-split leakage**: 34 of the 38 graded test papers appear in its training data (a different HF dataset happened to scrape the same underlying papers; the data-hygiene intent was real but the validator only checked a `source` string, never actual paper identity). It also still targets a toy `distilgpt2` rather than the model this project actually deploys. A fix (purge the leaked rows, make the validator check real paper identity, retarget to `Qwen/Qwen2.5-7B-Instruct` per `.env`'s `LLM__PROVIDER`) was dispatched — check `finetune/`'s own git log and this file's next edit for whether it landed. |
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
2. **Human-in-the-loop persistence** — ~~the one remaining big piece~~ **mostly
   done.** MySQL is running locally (port 3307 — a teammate had already
   stood up an instance matching this project's own documented port
   convention; no Docker was available on this machine, and no
   `docker-compose.yml` exists). `server/pipeline.py` writes
   `reviewed_papers`/`review_assessments`/`reflection_flags` for real as
   each run streams, best-effort (a DB hiccup is logged and swallowed,
   never allowed to break a live review). `POST /api/approval/{run_id}`
   genuinely persists a decision to `human_approvals`, and `GET
   /api/history` / `GET /api/history/{trace_id}` surface real past runs in
   the dashboard. **What's still not built**: a true LangGraph
   interrupt/resume — the graph still runs start-to-finish in one pass; a
   human reviews and records a decision on the *completed* result
   afterward, rather than the graph actually pausing mid-run to wait for
   one. This was a deliberate scope choice (see the DB-setup/HITL-scope
   decisions made this session) — revisit only if true mid-run pause/resume
   becomes a real requirement, since it needs a durable checkpointer
   (`SqliteSaver`, not the current in-memory one) to survive across
   separate HTTP requests.
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
   genuinely vary (`weak_reject`/`weak_accept` mixed, no repeats).

   **Two rounds of rubric fixes so far, numbers below are from the SECOND
   round (re-run and replace these, don't trust them as permanently
   current, per this file's own §5 rule):**

   Round 1 fix (described above) resolved the degenerate "always
   borderline" collapse but overcorrected: it required all 4 rating
   signals (novelty/soundness/citation/evidence) to unanimously agree
   before allowing a confident accept/reject lean, which real mixed papers
   almost never do — so nearly everything fell through to "borderline"
   again, just via a different mechanism, and separately (round 1's actual
   numbers: accuracy 0.5143, f1 0.4138, κ 0.0067, `weak_reject`:19/
   `weak_accept`:16) still never produced a confident `accept`/`reject`.

   **Round 2 fix**: redesigned the rubric around how real peer review
   actually weighs signals — novelty/soundness/evidence are the primary
   "is this good science" axis (majority-of-3, which almost always
   resolves to a clear direction); citation quality is a secondary
   strength-modifier, not a veto (a paper can have real citation gaps and
   still clearly deserve accept if the science is strong).

   | Metric | Value |
   |---|---|
   | n_usable / n_total | 33 / 38 (5 residual schema-validation misses — same low-frequency LLM noise, not a systematic bug) |
   | accuracy | 0.4848 |
   | f1 | 0.5854 |
   | cohen_kappa | 0.1024 |
   | ground_truth_accept_rate | 0.3939 |
   | predicted_accept_rate | 0.8485 |
   | `final_recommendation` distribution | `weak_accept`: 28, `weak_reject`: 5 (still no confident `accept`/`reject`, and no `borderline`) |

   **Honest read:** genuine progress on the more rigorous metric — κ nearly
   tripled (0.007→0.10, "slight" agreement on the standard scale, up from
   "none") and F1 improved substantially. But raw accuracy actually *dropped*
   (51%→48%), because the fix traded one bias for another: it now
   over-predicts accept (85% predicted vs. 39% actual base rate), most
   likely because the upstream novelty/methodology/evidence agents
   themselves lean generously positive by default (rarely landing on
   "poor") and a majority-of-3 vote amplifies that leniency. This is the
   concrete evidence motivating the **Adversarial Critic** addition (see §3
   table and `AGENTS_ARCHITECTURE.md`) — forcing Methodology/Citation/
   Evidence to defend specific claims against real pushback, rather than
   settle on a comfortable middling rating, is a directly targeted response
   to this exact failure mode. Re-run the harness after the Adversarial
   Critic lands to see whether κ/accuracy actually move, rather than
   assuming it will help.
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
                  (runs core/utils/guardrails.py's prompt-injection sanitizer on every text field)
core/rag/         Two-index retrieval (paper's own + persistent literature corpus)
core/agents/      The 10 review agents (9 original + Adversarial Critic) + shared base/structured-output infra
                  core/agents/novelty/  <- the separate embedding-based novelty scorer
core/graph/       LangGraph orchestration (state.py, nodes.py, build_graph.py)
core/llm/         Ollama client factories (text + vision), prompt manager, structured output
core/db/          SQLAlchemy models + migration + repositories -- real, live, MySQL running on :3307
core/eval/        PeerRead accuracy/F1/κ harness (peerread_harness.py) + optional DeepEval/RAGAS
                  quality gates (deepeval_quality.py, ragas_quality.py) -- see QUALITY_GATES.md
core/utils/       guardrails.py (prompt-injection defenses), grounding.py, token_budget.py
core/config/      Settings (env-overridable) + prompts.yaml
server/           FastAPI backend for the live SSE dashboard -- pipeline.py streams the graph AND
                  persists to MySQL; main.py adds /api/approval, /api/history, /api/health
ai_paper_reviewer_ui.html   The dashboard itself
scripts/          Manual verification scripts + CLI entry points (incl. run_peerread_evaluation.py)
tests/unit/       Automated pytest suite (53+ tests)
data/             Sample PDFs + real ICLR-2017 corpora (data/peerread_raw/, gitignored -- see
                  PEERREAD_CORPUS_MODULE.md for how to regenerate)
finetune/         A separate, gitignored, nested git repo -- NOT part of this project's own
                  history. A fine-tuning side-effort; see §4's branch history table for its
                  status (real PeerRead test-split leakage was found and is being fixed).
```
