# Multi-Agent System Design — The Big Picture

*A system-level view of how parsing, RAG, and 9 specialized agents combine
into one reviewer, mapped against the problem statement's "Multi-Agent
Controller" requirement. Complements `AGENTS_ARCHITECTURE.md` (what each
agent does) and `LANGGRAPH_ORCHESTRATION.md` (the graph mechanics) — this
doc is about the design philosophy connecting them.*

## The core design bet: specialization over one giant prompt

The problem statement's architecture diagram calls for a "Multi-Agent
Controller" dispatching to specialized agents (Novelty, Methodology,
Evidence, Citation) rather than one LLM call asked to produce an entire
review at once. This system commits to that fully — 9 agents, each with a
narrow, well-defined job:

| Agent | Sole responsibility |
|---|---|
| Paper Understanding | Condense the paper into a briefing every other agent reads instead of re-reading the full text |
| Literature RAG | Retrieve prior-art context (no LLM call — pure retrieval) |
| Novelty (×2) | Judge what's genuinely new vs. what overlaps with retrieved work |
| Methodology | Judge experimental soundness |
| Citation | Judge reference-list coverage |
| Evidence & Reproducibility | Judge whether claims are backed by tables, and whether the work is reproducible |
| Figure & Table | Summarize visual/tabular content |
| Reflection | Audit the above four for unsupported or inconsistent claims |
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
| All 9 agents individually | Built, tested, verified against real Ollama |
| Wired into one bounded LangGraph run | **Done this session** (Phase 1) — parallel fan-out, grounded revision loop, verified end-to-end for real |
| Wired into the live dashboard | **Not yet** — UI still shows placeholder cards for 5 of the 9 agents |
| Human-in-the-loop approval before final output | **Not yet** (Phase 2) |
| Writing a completed review to the database | **Not yet** (Phase 2) — nothing persists a review today |
| Evaluated against PeerRead's labeled test split | **Not yet** (Phase 3 — the graded core of the problem statement) |

**The multi-agent controller itself — the thing the problem statement
actually asks for — is real and working.** What's left is connecting it to
persistence, human review, the live UI, and the evaluation harness that
measures how well it actually performs.
