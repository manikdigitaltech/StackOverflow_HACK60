# LangGraph Orchestration — Wiring the Agents Into One Review Run

*Covers `core/graph/`. This is Phase 1 of the post-merge build plan — the
piece that turns 9 independently-callable agent classes into an actual
reviewer. Verified both structurally (mocked) and with a real end-to-end
Ollama run.*

## The graph shape

```
                     ┌──────────────────┐
              ┌──────► paper_understanding ─────┐
              │      └──────────────────┘       │
   START ─────┤      ┌──────────────────┐       ├──► novelty ──────┐
              ├──────► literature_rag ──────────┘                  │
              │      └──────────────────┘ ───────────► citation ───┤
              ├──────► figure_table (also feeds synthesis, below)  │
              ├──────► methodology ────────────────────────────────┤
              └──────► evidence_reproducibility ────────────────────┤
                                                                     ▼
                                                              reflection
                                                              /         \
                                                      [revise]           [proceed]
                                                          │                  │
                                                  prepare_revision   ready_for_synthesis
                                                          │                  │ (+ figure_table)
                                            (loops back to the 4      final_review
                                             assessment agents)             │
                                                                            ▼
                                                                           END
```

## State (`state.py`)

`ReviewGraphState` — a `TypedDict` with one key per agent's output
(`paper_understanding`, `literature_context`, `novelty_assessment`, etc.),
plus `revision_count` and `revision_feedback`. Parallel branches never write
the same key — that's what lets them run concurrently without a custom
reducer function.

## Nodes (`nodes.py`)

`ReviewGraphNodes` constructs all 10 agents **once** (not per node-call,
including the Adversarial Critic — see `AGENTS_ARCHITECTURE.md`), and each
method is a thin wrapper: read the relevant keys out of state, call the
agent's `.run()`, return a partial state update.

Two nodes exist purely for orchestration, not review logic:

- **`route_after_reflection`** — the conditional-edge router: revise (once,
  bounded by `settings.reflection.max_revision_passes`, default 1) if
  `ReflectionNotes.needs_revision` is true, else proceed straight to synthesis.
- **`prepare_revision`** — builds the actual feedback text fed back into the
  4 assessment agents (from `ReflectionNotes.flags`, capped at 8) and
  increments the bounded counter. **Without this node, "revision" would just
  be an identical re-run of the same agents on the same inputs** — this is
  what makes the second pass a genuine revision (see `NOVELTY_AGENT.md` for
  a concrete example of this changing a real verdict).
- **`ready_for_synthesis`** — a pure pass-through sync node (see the fan-in
  gotcha below for why it exists).

## The revision loop is real, not cosmetic

`core/agents/revision.py` centralizes one shared feedback-block format,
threaded through `novelty_agent`, `methodology_agent`, `citation_agent`, and
`evidence_reproducibility_agent`'s `run()` calls and their
`prompts.yaml` templates (each gained a `{revision_feedback}` placeholder,
empty on a first pass). Confirmed working in the real Ollama run this
session: after the first reflection pass flagged 2 issues, all four
assessment agents genuinely re-ran with that specific feedback — Novelty's
rating changed between passes as a direct result.

## Two real LangGraph fan-in bugs, found and fixed

Both were caught by a dedicated, fast, fully-mocked topology test
(`scripts/test_graph_topology.py`, runs in milliseconds) written specifically
to stress this — worth understanding since they're non-obvious and will
recur if the graph is extended without knowing this rule:

> **`add_edge(a, "x")` and `add_edge(b, "x")` called separately are
> independent OR-triggers, not an AND-join.** A true "wait for all of these"
> join in LangGraph requires listing every source together in **one**
> `add_edge([a, b], "x")` call.

**Bug 1** — `reflection` has 4 real prerequisites (novelty, methodology,
citation, evidence_reproducibility). Wired as 4 separate `add_edge(...)`
calls, `reflection` fired the instant `methodology`/`evidence_reproducibility`
completed (they only need `START`, finishing one superstep before
`novelty`/`citation`, which need `paper_understanding`/`literature_rag` first)
— crashing with `KeyError: 'novelty_assessment'`. **Fix:**
`add_edge(["novelty", "methodology", "citation", "evidence_reproducibility"],
"reflection")`.

**Bug 2** — `figure_table` (feeds `final_review`) and `reflection`'s
"proceed" branch (also targets `final_review`) are two *independent*
triggers from two different mechanisms (`add_edge` vs.
`add_conditional_edges`) — mixing them doesn't create a join either.
`final_review` fired the instant `figure_table` completed, before reflection
had even run once. **Fix:** a conditional edge's dynamically-chosen target
can't itself be one entry in a list-join, so route "proceed" to a trivial
new node (`ready_for_synthesis`) instead, then
`add_edge(["figure_table", "ready_for_synthesis"], "final_review")`. This
node runs at most once (nothing loops back to it), so there's no ambiguity
about needing a "fresh" `figure_table` signal on a later revision pass.

## Checkpointing

Compiled with `InMemorySaver` for now — sufficient for Phase 1's scope
(one bounded, in-process revision loop). A real `SqliteSaver` swap is a
Phase 2 concern, once human-in-the-loop interrupt/resume across separate API
calls actually needs crash-safe persistence.

## Verified

- **Structural test** (`scripts/test_graph_topology.py`, all 9 agents
  mocked): confirms correct fan-out order, the revision loop re-runs exactly
  the 4 assessment agents with `revision_feedback` set on pass 2,
  `figure_table` runs exactly once (never re-triggered by the loop),
  `revision_count` bounds correctly, `final_review` receives
  `figure_table_summary`. All assertions pass.
- **Real end-to-end run against local Ollama** (no mocks, ~20.6 minutes,
  synthetic short paper): parallel fan-out ran correctly; first reflection
  pass found 2 flags (`needs_revision=True`) and triggered the loop; all 4
  assessment agents genuinely re-ran with feedback; second reflection pass
  found 3 flags but none "major" (`needs_revision=False`) and correctly
  proceeded; `revision_count: 1` in the final state; final review synthesized
  a coherent, accurately-grounded recommendation
  (`borderline`/`medium confidence`, correctly citing the test paper's actual
  gaps — no ablations, no compute details).

## Not yet done

Not wired into `server/pipeline.py` / the live dashboard yet — the SSE
pipeline still shows the old placeholder "not yet implemented" cards for
methodology/citation/evidence/reflection/final_review instead of real
graph-driven events. No DB persistence and no human-in-the-loop interrupt
(both Phase 2).
