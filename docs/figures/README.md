# Figures for the Project PPT

Simple, self-contained SVG diagrams summarizing the `docs/*.md` deep-dives — one per slide.
SVGs can be inserted directly into PowerPoint (`Insert → Pictures`) and stay sharp at any size.
All figures share one palette: blue = pipeline/infra, purple = agents/LLM, amber = human/decision,
green = outputs, red = security.

| # | Figure | Suggested slide | Source doc(s) |
|---|--------|-----------------|---------------|
| 01 | `01_problem_overview.svg` | Problem statement: input → system → output + hard constraints | `PROBLEM_STATEMENT.md` |
| 02 | `02_end_to_end_pipeline.svg` | Our approach: the 8-step flow from upload to approved verdict | `CONTEXT.md`, `PROBLEM_STATEMENT.md` |
| 03 | `03_document_parsing.svg` | Step 1 — scientific document understanding (Docling) | `PARSING_ARCHITECTURE.md` |
| 04 | `04_rag_architecture.svg` | Step 2 — two-index RAG design (Paper-RAG vs Literature-RAG) | `RAG_ARCHITECTURE.md`, `PEERREAD_CORPUS_MODULE.md` |
| 05 | `05_langgraph_multi_agent.svg` | Step 3 — the 10-agent LangGraph orchestration | `LANGGRAPH_ORCHESTRATION.md`, `AGENTS_ARCHITECTURE.md`, `MULTI_AGENT_SYSTEM.md` |
| 06 | `06_novelty_two_ways.svg` | Novelty: deterministic score + grounded LLM reasoning | `NOVELTY_AGENT.md` |
| 07 | `07_vision_figure_analysis.svg` | Figures & tables: caption path vs real VLM vision path | `VLM_FIGURE_TABLE_ANALYSIS.md` |
| 08 | `08_reflection_revision_loop.svg` | Self-reflection & the (real) revision loop | `LANGGRAPH_ORCHESTRATION.md`, `AGENTS_ARCHITECTURE.md` |
| 09 | `09_rebuttal_rereview.svg` | Brownie point: rebuttal-aware re-review | `REBUTTAL_REREVIEW.md` |
| 10 | `10_guardrails_security.svg` | Security: prompt-injection guardrails + OWASP hardening | `GUARDRAILS.md`, `OWASP_LLM_SECURITY.md` |
| 11 | `11_live_dashboard.svg` | Demo surface: FastAPI + SSE live dashboard (why not Streamlit) | `UI_WORK.md` |
| 12 | `12_evaluation_quality.svg` | Evaluation: PeerRead metrics + DeepEval/RAGAS quality gates | `QUALITY_GATES.md`, `PROBLEM_STATEMENT.md` |

Suggested PPT narrative order: 01 → 02 (the story), then 03–08 step by step (the approach),
then 09–12 (extras: rebuttal, security, demo, evaluation).
