# TODO — Problem Statement Coverage & Remaining Work

Audit of the project against `docs/PROBLEM_STATEMENT.md`. Section refs (§) point
back into that file. Status verified against the actual codebase, not just docs.

---

## ✅ Fully covered (built + tested)

| Requirement (§) | Where |
|---|---|
| Scientific Document Understanding — PDF parse, layout, OCR, figure/table/reference extraction (§1, §6.1) | `core/parsing/` (Docling + section segmenter + reference extractor) |
| Multi-Agent Controller via LangGraph (§6.2) | `core/graph/build_graph.py` |
| Novelty / Method / Evidence / Citation agents (§6.3) | `core/agents/` (9 agents) |
| RAG grounding with FAISS + LangChain (§6.4) | `core/rag/` (Index A hybrid + Index B literature) |
| Semantic Scholar + arXiv retrieval (§9) | `core/rag/live_sources/` via `core/agents/literature_rag_agent.py` — live-source merge into `LiteratureContext`, on by default, degrades gracefully offline (`RAG_SETTINGS.live_sources.enable_arxiv`/`enable_semantic_scholar` to disable) |
| Self-Reflection / Verifier agent (§1, §6.6) | `core/agents/reflection_agent.py` (drives bounded revision loop) |
| Local LLM within 24 GB budget, Qwen2.5 7B (§3, §5, §11) | `core/llm/` (Ollama, no cloud calls) |
| VLMs for figures/tables (§4) | `core/parsing/figure_analyzer.py` — ⚠️ no vision model pulled |
| Structured final report — Summary/Strengths/Weaknesses/Questions/5-way Rating/Confidence/Justification (§7) | `FinalReview` in `core/schemas/agent_output_schemas.py` |
| Optional bonuses — Novelty/Citation/Reproducibility/Evidence mapping/Missing baselines (§7 bonus) | dedicated agents in `core/agents/` |
| Explainable AI / evidence grounding (§4, §10) | per-item grounded verdicts + `core/agents/novelty/decision_trace_builder.py` |

---

## ⚠️ Built but not connected (integration gaps)

- [ ] **Wire LangGraph into the live UI** — `server/pipeline.py` still emits
  `status="not_implemented"` for methodology/citation/evidence/reflection/final_review.
  The graph works standalone; the dashboard doesn't drive it yet.
- [x] **Wire live sources into `LiteratureRAGAgent`** — arXiv/Semantic
  Scholar clients now merge into `LiteratureContext` (on by default,
  graceful offline degradation) via `core/rag/retrieval/tools.py`;
  `NoveltyAgent`/`CitationAgent`
  label live vs. curated-corpus hits in their prompts. See
  `docs/AGENTS_ARCHITECTURE.md`. HyDE/decompose (`query_helpers.py`) remain
  unwired — separate, smaller follow-up (query-shaping, not source-merging).
- [ ] **Wire DB persistence** — schema + repositories exist (`core/db/`, incl.
  `approval_repository.py`), but nothing writes to them; MySQL isn't running.

---

## ❌ Not started — graded / high-value / mandatory

- [x] **Human-in-the-Loop approval** — §1, §2, §6.7, §11 (the mandatory
  constraint). Done end to end: `human_approval` graph node with LangGraph
  `interrupt()` after `final_review` (approve / reject / revise-with-override,
  human's override rewrites the recommendation), an `awaiting_approval` SSE
  event + `POST /api/approve/{run_id}` resume endpoint in `server/`, and live
  Approve/Request Changes/Reject buttons in the UI. Verified by
  `scripts/test_human_approval.py` (graph) and an end-to-end mocked server
  flow. Only DB persistence of decisions remains (tracked above under
  "Wire DB persistence").
- [ ] **Build the real PeerRead literature corpus (data)** — §9 (mandatory dataset).
  `core/rag/ingestion/build_corpus.py` exists but was never run against a real
  PeerRead clone → `search_literature()` returns `[]` today, so novelty/citation
  grounding runs empty.
- [ ] **Evaluation harness — accuracy / F1 / Cohen's κ on PeerRead test split** —
  §8 (brownie), §10 (metrics). **Zero code today.** Load PeerRead's built-in
  80/10/10 test split (`reviews/*.json`, has `accepted` label), run the graph
  per paper, map `final_recommendation` → accept/reject, compute metrics on the
  test set only (never invent a split).
- [ ] **Hallucination Rate / Evidence Grounding / Reviewer Agreement metrics** —
  §10. Not measured. (vivek's shelved DeepEval/RAGAS branch is a starting template.)
- [ ] **Rebuttal-aware re-review** — §8 (brownie). Not built. Partial foundation:
  the existing reflection/revision loop could be repurposed.

---

## Priority order

1. **Build the real PeerRead corpus** — unblocks genuine novelty/citation grounding.
   Run `build_corpus.py` against a PeerRead clone.
2. **Evaluation harness** — highest graded value (§8, §10). Test-split loader →
   graph → accept/reject mapping → accuracy/F1/κ.
3. **Bonus polish** — rebuttal re-review, hallucination/grounding metrics
   (DeepEval/RAGAS), wire HyDE/decompose query-shaping, pull a vision model
   + `VISION__ENABLED=true`, DB persistence of approvals/reviews.

---

**Bottom line:** every *architectural* requirement — including the mandatory
human-in-the-loop approval gate — is built and tested. Missing: (a) real data
in the literature corpus, and (b) all measurement — the evaluation harness
that produces the accuracy/F1/κ numbers the challenge is graded on.
