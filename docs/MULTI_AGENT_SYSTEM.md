# Multi-Agent System Design — The Big Picture

*A system-level view of how parsing, RAG, and specialized agents combine
into one reviewer, mapped against the problem statement's "Multi-Agent
Controller" requirement. Complements `AGENTS_ARCHITECTURE.md` (what each
agent does) and `LANGGRAPH_ORCHESTRATION.md` (the graph mechanics) — this
doc is about the design philosophy connecting them.*

## The core design bet: specialization over one giant prompt

The problem statement's architecture diagram calls for a "Multi-Agent
Controller" dispatching to specialized agents (Novelty, Methodology,
Evidence, Citation) rather than one LLM call asked to produce an entire
review at once. This system commits to that fully — 10 agents (9 original +
an Adversarial Critic added later, see below), each with a narrow,
well-defined job:

| Agent | Sole responsibility |
|---|---|
| Paper Understanding | Condense the paper into a briefing every other agent reads instead of re-reading the full text |
| Literature RAG | Retrieve prior-art context (no LLM call — pure retrieval) |
| Novelty (×2) | Judge what's genuinely new vs. what overlaps with retrieved work |
| Methodology | Judge experimental soundness |
| Citation | Judge reference-list coverage |
| Evidence & Reproducibility | Judge whether claims are backed by tables, and whether the work is reproducible |
| Figure & Table | Summarize visual/tabular content |
| Adversarial Critic | Actively attack Methodology/Citation/Evidence's verdicts, forcing them to survive real pushback instead of settling on a comfortable middling rating |
| Reflection | Audit Methodology/Citation/Evidence/Novelty *and* the Adversarial Critique for unsupported, inconsistent, or over-conceded claims |
| Final Review | Synthesize everything into the required output shape |

**Why this over one big prompt:** each agent's prompt can enforce a narrow,
checkable contract (see below) that would be much harder to hold a single
sprawling prompt to consistently. A single mega-prompt asked to do all of
this at once tends to produce plausible-sounding prose that's much harder to
audit for exactly *which* claim is grounded in *which* evidence.

## Grounding as a cross-cutting system property

The single most important design decision running through every agent isn't
visible in any one file — it's a discipline enforced consistently across all
of them:

- **Per-item verdicts, not vibes.** Novelty must produce one verdict per
  *stated contribution*; Citation must produce one verdict per *retrieved
  paper*; Methodology must produce one verdict per *fixed methodology
  aspect*. This forces the model to actually address every item rather than
  produce a plausible-sounding summary that quietly skips the hard cases.
- **Cited evidence, not assertion.** Methodology's missing-baselines list
  must be a method the paper's *own text* names; Evidence's claim-checking
  uses tables as the *only* ground truth; Citation's overlap titles must be
  copied verbatim from retrieved text, never paraphrased or invented.
- **A dedicated auditor.** Reflection exists specifically to catch the cases
  where an agent's grounding discipline slipped anyway — it reads all four
  assessments plus the original paper and flags anything speculative or
  inconsistent, with a severity level gating whether a revision pass is worth
  the cost.
- **Deterministic overrides where LLM judgment would only introduce risk.**
  `FinalReview.missing_baselines` is *copied* from Methodology's own list at
  synthesis time rather than re-derived — a case where "let the LLM
  summarize it again" would only risk drift from what was actually found.

## Two novelty signals, on purpose

The system runs both an embedding-based novelty scorer (fast, deterministic,
zero hallucination risk, see `NOVELTY_AGENT.md`) and an LLM-based novelty
agent (slower, but produces *qualitative, grounded reasoning about which
specific contribution is novel and why*) side by side. This wasn't planned
as a redundancy to resolve — they answer genuinely different questions and
a real review benefits from both a hard number and a reasoned narrative.

## How review-quality safety nets compose

Two independent mechanisms, each catching a different failure mode:

1. **Grounding constraints** (above) reduce the *rate* of ungrounded claims
   at generation time.
2. **The Reflection + bounded revision loop** (see
   `LANGGRAPH_ORCHESTRATION.md`) catches what gets through anyway, and
   actually re-runs the relevant agents with specific feedback — confirmed in
   a real run this session to change a real verdict (Novelty's rating
   shifted between passes after reflection surfaced a missed overlap).

Neither alone is the whole answer — grounding constraints reduce how often
something needs catching; reflection is the backstop for when they don't.

## Current integration status (the honest picture)

| Layer | State |
|---|---|
| All 10 agents individually | Built, tested, verified against real Ollama |
| Wired into one bounded LangGraph run | **Done** — parallel fan-out, grounded revision loop (now including the Adversarial Critic's own AND-join, re-fires automatically on a revision pass), verified end-to-end for real |
| Wired into the live dashboard | **Done** — every agent has a real card with real structured rendering; no placeholder cards remain except `human_approval`'s persistence step (see below) |
| Review persistence to MySQL | **Done** — `reviewed_papers`/`review_assessments`/`reflection_flags` are written for real as each run streams, best-effort (a DB hiccup never breaks a live review); `GET /api/history` and `GET /api/history/{trace_id}` surface past runs |
| Human-in-the-loop approval | **Partially done** — the dashboard's Human Approval view shows the real final review, and `POST /api/approval/{run_id}` genuinely persists a decision to `human_approvals`. What's *not* built: the graph doesn't actually pause/interrupt mid-run to wait for that decision (a deliberate scope call — see `docs/CONTEXT.md` §7) |
| Evaluated against PeerRead's labeled test split | **Done** — `core/eval/peerread_harness.py` + `scripts/run_peerread_evaluation.py`; see `docs/CONTEXT.md` §7 item 3 for the latest real accuracy/F1/κ numbers |
| Agent-output quality gates (DeepEval/RAGAS) | **Done, optional** — see `docs/QUALITY_GATES.md`; a second, complementary signal to the PeerRead accuracy numbers, checking *how* an agent reasoned, not just whether its final call matched ground truth |

**The multi-agent controller itself — the thing the problem statement
actually asks for — is real, working, persisted, and measured against the
graded eval set.** What's left is the true interrupt/resume half of
human-in-the-loop (Phase 2) and any further iteration the eval numbers
motivate (see `docs/CONTEXT.md`'s honest read on the current κ score).
