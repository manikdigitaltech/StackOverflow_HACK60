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
| Semantic Scholar + arXiv retrieval (§9) | `core/rag/live_sources/` — ⚠️ built but not wired into any agent |
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
- [ ] **Wire live sources & query helpers** — arXiv/Semantic Scholar clients +
  HyDE/decompose are built and tested, but no agent calls them.
- [ ] **Wire DB persistence** — schema + repositories exist (`core/db/`, incl.
  `approval_repository.py`), but nothing writes to them; MySQL isn't running.

---

## ❌ Not started — graded / high-value / mandatory

- [ ] **Human-in-the-Loop approval** — §1, §2, §6.7, §11. **MANDATORY constraint**,
  not a bonus ("Final decision requires human-in-the-loop approval"). Needs a
  LangGraph `interrupt()` before `final_review` + an approval action in the UI.
  No `interrupt()`, no approval flow exists today.
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

1. **Human-in-the-Loop approval** — the only mandatory (§11) item not done; small.
   LangGraph `interrupt()` before `final_review` + UI approve/reject.
2. **Build the real PeerRead corpus** — unblocks genuine novelty/citation grounding.
   Run `build_corpus.py` against a PeerRead clone.
3. **Evaluation harness** — highest graded value (§8, §10). Test-split loader →
   graph → accept/reject mapping → accuracy/F1/κ.
4. **Wire LangGraph into the live dashboard** — mechanical, low-risk; replaces
   placeholder cards.
5. **Bonus polish** — rebuttal re-review, hallucination/grounding metrics
   (DeepEval/RAGAS), wire live sources, pull a vision model + `VISION__ENABLED=true`.

---

**Bottom line:** every *architectural* requirement is built and tested. Missing:
(a) one mandatory constraint — human-in-the-loop approval, (b) real data in the
literature corpus, and (c) all measurement — the evaluation harness that produces
the accuracy/F1/κ numbers the challenge is graded on.
