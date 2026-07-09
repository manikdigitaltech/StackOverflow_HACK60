# LangGraph Orchestration — Wiring the Agents Into One Review Run

*Covers `core/graph/`. Turns 10 independently-callable agent classes into
an actual reviewer, gated behind a mandatory human-approval interrupt.
Verified structurally (mocked), with real end-to-end Ollama runs, and live
through the dashboard.*

## The graph shape

```
                     ┌──────────────────┐
              ┌──────► paper_understanding ─────┐
              │      └──────────────────┘       │
   START ─────┤      ┌──────────────────┐       ├──► novelty ────────────┐
              ├──────► literature_rag ──────────┘                        │
              │      └──────────────────┘ ───────────► citation ─────────┤
              ├──────► figure_table (also feeds synthesis, below)        │
              ├──────► methodology ───────────────────┬───────────────────┤
              └──────► evidence_reproducibility ───────┴──► adversarial_critic
                                                                            │
                                                              (methodology/citation/
                                                               evidence only, not novelty)
                                                                            │
                                                                            ▼
                                                                     reflection
                                                                     /         \
                                                             [revise]           [proceed]
                                                                 │                  │
                                                         prepare_revision   ready_for_synthesis
                                                                 │                  │ (+ figure_table)
                                                   (loops back to the 4      final_review
                                                    assessment agents --            │
                                                    adversarial_critic re-fires            ▼
                                                    "for free" via its own AND-join)  human_approval
                                                                                            │
                                                                                    (interrupt(): pauses
                                                                                     until a human resumes
                                                                                     with a decision)
                                                                                            │
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

## Human-in-the-loop approval gate

`final_review --> human_approval --> END` (see `build_graph.py`). The
`human_approval` node (`nodes.py`) calls `langgraph.types.interrupt()` with
the drafted review (recommendation, confidence, summary, strengths,
weaknesses, questions for authors) as the interrupt payload — this genuinely
pauses graph execution; `graph.stream()` yields `{"__interrupt__": (...)}`
and returns control to the caller. `server/pipeline.py` surfaces this as a
`human_approval`/`awaiting_approval` SSE event. Resuming requires
`Command(resume=decision_payload)` against the *same* `thread_id` the run
was invoked with (see `server/pipeline.py::resume_with_approval`) —
`decision_payload` accepts either a terse string (`"approve"`) or a dict
(`{"decision": ..., "approver": ..., "comment": ..., "override_recommendation": ...}`).
A decision of `"revised"` with `override_recommendation` set rewrites
`final_review.final_recommendation` before the graph reaches `END`.

## Checkpointing

Compiled with `InMemorySaver` — a parked run (waiting at `human_approval`)
only survives within the same server process. A real `SqliteSaver` swap
would be needed for a parked run to survive a server restart; not yet a
real requirement since MySQL (not the graph checkpointer) is the durable
copy of record for a decided approval.

## Verified

- **Structural test** (`scripts/test_graph_topology.py`, all 10 agents
  mocked): confirms correct fan-out order, the revision loop re-runs exactly
  the 4 assessment agents with `revision_feedback` set on pass 2 (and
  `adversarial_critic` re-fires "for free" via its own AND-join),
  `figure_table` runs exactly once (never re-triggered by the loop),
  `revision_count` bounds correctly, `final_review` receives
  `figure_table_summary`. All assertions pass.
- **Human-approval interrupt/resume test** (`scripts/test_human_approval.py`,
  all agents mocked): 4 scenarios — approve as-drafted, reject outright,
  human overrides the recommendation, terse bare-string resume (`"approve"`)
  — confirming the graph genuinely parks at `__interrupt__` with the drafted
  recommendation, and each resume path produces the correct final state.
- **Real end-to-end run against local Ollama** (no mocks, ~20.6 minutes,
  synthetic short paper): parallel fan-out ran correctly; first reflection
  pass found 2 flags (`needs_revision=True`) and triggered the loop; all 4
  assessment agents genuinely re-ran with feedback; second reflection pass
  found 3 flags but none "major" (`needs_revision=False`) and correctly
  proceeded; `revision_count: 1` in the final state; final review synthesized
  a coherent, accurately-grounded recommendation
  (`borderline`/`medium confidence`, correctly citing the test paper's actual
  gaps — no ablations, no compute details).
- **Live, through the dashboard**: uploaded real PDFs and watched the graph
  stream via `server/pipeline.py`, including a genuine revision pass, park at
  `human_approval`/`awaiting_approval` with the real interrupt payload, and
  resume correctly via `POST /api/approval/{run_id}` with the decision
  landing in MySQL (`human_approvals`, `reviewed_papers.status=completed`).

## Not yet done

A parked run only survives within the same server process (in-memory
checkpointer) — see Checkpointing above.
