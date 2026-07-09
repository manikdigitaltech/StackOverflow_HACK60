# TODO — Problem Statement Coverage & Remaining Work

Audit of the project against `docs/PROBLEM_STATEMENT.md`. Section refs (§) point
back into that file. Status verified against the actual codebase, not just docs.

---

## ✅ Fully covered (built, tested, and measured)

| Requirement (§) | Where |
|---|---|
| Scientific Document Understanding — PDF parse, layout, OCR, figure/table/reference extraction (§1, §6.1) | `core/parsing/` (Docling + section segmenter + reference extractor + prompt-injection guardrails) |
| Multi-Agent Controller via LangGraph (§6.2) | `core/graph/build_graph.py` — 10 agents, parallel fan-out, bounded revision loop, mandatory human-approval interrupt |
| Novelty / Method / Evidence / Citation agents (§6.3) | `core/agents/` (10 agents, incl. an Adversarial Critic added after the first eval run — see `AGENTS_ARCHITECTURE.md`) |
| RAG grounding with FAISS + LangChain (§6.4) | `core/rag/` — Index A (per-paper hybrid dense+BM25) + Index B (persistent literature corpus, real ICLR-2017 data, 389 papers) |
| Semantic Scholar + arXiv retrieval (§9) | `core/rag/live_sources/` via `core/agents/literature_rag_agent.py` — live-source merge into `LiteratureContext`, on by default, degrades gracefully offline (`RAG_SETTINGS.live_sources.enable_arxiv`/`enable_semantic_scholar` to disable) |
| Self-Reflection / Verifier agent (§1, §6.6) | `core/agents/reflection_agent.py` (drives bounded revision loop, now also folds in the Adversarial Critic's critique) |
| **Human-in-the-Loop approval** (§1, §2, §6.7, §11 — the mandatory constraint) | `core/graph/nodes.py::human_approval` — a real LangGraph `interrupt()` genuinely pauses the graph after `final_review`; `POST /api/approval/{run_id}` resumes it via `Command(resume=...)` (approve / reject / revise-with-override) AND persists the decision to MySQL `human_approvals`. Live Approve/Request Changes/Reject buttons in the UI. Verified by `scripts/test_human_approval.py` (4 scenarios) and a live end-to-end server run |
| Local LLM within 24 GB budget, Qwen2.5 7B (§3, §5, §11) | `core/llm/` (Ollama, no cloud calls) |
| VLMs for figures/tables (§4) | `core/parsing/figure_analyzer.py` — code-complete, verified on GPU; off by default (no vision model pulled by default) |
| Structured final report — Summary/Strengths/Weaknesses/Questions/5-way Rating/Confidence/Justification (§7) | `FinalReview` in `core/schemas/agent_output_schemas.py` |
| Optional bonuses — Novelty/Citation/Reproducibility/Evidence mapping/Missing baselines (§7 bonus) | dedicated agents in `core/agents/` |
| Explainable AI / evidence grounding (§4, §10) | per-item grounded verdicts + `core/agents/novelty/decision_trace_builder.py` |
| **Real PeerRead literature corpus (data)** (§9, mandatory dataset) | `core/rag/ingestion/build_corpus.py`, run for real against `data/peerread_raw/iclr_2017/` — 389 papers (train+dev) indexed into both Index B and the novelty corpus, `test` split held out for grading. See `PEERREAD_CORPUS_MODULE.md` |
| **Evaluation harness — accuracy / F1 / Cohen's κ on PeerRead test split** (§8 bonus, §10 metrics) | `core/eval/peerread_harness.py` + `scripts/run_peerread_evaluation.py` — loads the real held-out `test` split (38 papers), runs the full graph per paper, maps `final_recommendation` → accept/reject, scores with scikit-learn. Real numbers (not placeholders) in `docs/CONTEXT.md` §7 item 3 — currently accuracy 0.4848 / f1 0.5854 / κ 0.1024; honest read of what's still off is in that same section |
| Hallucination Rate / Evidence Grounding quality gates (§10) | `core/eval/deepeval_quality.py` (G-Eval + HallucinationMetric) / `core/eval/ragas_quality.py` (Faithfulness/AnswerRelevancy/ContextPrecision/ContextRecall) — ported and fixed from vivek's shelved branch, verified live. Optional/offline, not wired into the default eval run. See `QUALITY_GATES.md` |
| Prompt-injection guardrails | `core/utils/guardrails.py` — sanitizes every extracted text field (title/abstract/sections/tables/figures/references) before it reaches any agent prompt |
| Review persistence (MySQL) | `core/db/` — `reviewed_papers`/`review_assessments`/`reflection_flags`/`human_approvals` all written for real as each run streams, best-effort |

---

## Still open

- [ ] **Re-run the PeerRead evaluation harness now that the Adversarial
  Critic is merged** — the current κ/accuracy numbers in `docs/CONTEXT.md`
  §7 item 3 predate it; the Critic was added directly in response to the
  harness over-predicting accept (85% predicted vs. 39% actual base rate),
  but whether it actually moves the numbers hasn't been checked yet.
- [x] **Rebuttal-aware re-review** (§8 bonus) — done. A `rebuttal_feedback`
  channel (parallel to the self-reflection `revision_feedback`) threads the
  author rebuttal through the 4 assessment agents
  (`core/agents/revision.py` + `prompts.yaml`); `core/graph/rebuttal.py`'s
  `run_rebuttal_rereview()` re-runs the full graph seeded with the rebuttal —
  so the revised verdict also passes the mandatory human-approval gate —
  and `compare_recommendations()` reports the before/after shift. Exposed as
  `POST /api/rebuttal/{run_id}` and verified by
  `scripts/test_rebuttal_rereview.py` (rebuttal moves the verdict
  weak_reject → weak_accept). See `docs/REBUTTAL_REREVIEW.md`.
- [ ] **Wire HyDE/decompose query-shaping** (`core/rag/retrieval/
  query_helpers.py`) — exists but unwired; a smaller, separate follow-up
  from the live-source merge (query-shaping, not source-merging).
- [ ] **A durable graph checkpointer** (`SqliteSaver` instead of the current
  `InMemorySaver`) — only matters if a parked human-approval run needs to
  survive a server restart, not yet a real requirement.
- [ ] **Full-scale fine-tuning run** — `finetune/`'s leaked-data issue is
  fixed and a scoped smoke-test training run succeeded, but a full
  production run over the ~11k-row dataset is unscoped (needs explicit
  approval given shared GPU time).
- [ ] Pull a vision model + set `VISION__ENABLED=true` in a deployed
  environment (code-complete, just off by default).

---

**Bottom line:** every requirement in the problem statement's architecture
diagram — including the mandatory human-in-the-loop approval gate, real
data in the literature corpus, and the graded evaluation harness — is
built, tested, and measured. Rebuttal-aware re-review (§8 bonus) is now done
too. What's left is bonus-tier polish (query-shaping) and re-validating
whether the Adversarial Critic improved the eval numbers it was built to
address.
