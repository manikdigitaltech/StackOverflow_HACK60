# Rebuttal-Aware Re-Review — Revising the Verdict After an Author Response

*Covers `core/graph/rebuttal.py`, the `rebuttal_feedback` channel in
`core/agents/revision.py` + the 4 assessment agents, and the
`POST /api/rebuttal/{run_id}` server endpoint. This is the problem statement's
section-8 "brownie point": the agent revises its verdict after a simulated
author rebuttal.*

## In very simple terms (read this first)

When a scientist submits a paper, reviewers read it and say "here's what's
wrong." Normally the authors don't just accept that — they **write back** and
defend their work: *"You said we're missing an experiment — actually, look at
Table 4, we did it."* That written defense is called a **rebuttal**.

**Rebuttal feedback = taking that author's reply and showing it to our AI
reviewer so it can reconsider its score.** Maybe the authors made a good point
and the AI upgrades its verdict from "reject" to "accept" — or maybe the reply
is weak and the AI keeps its original opinion.

Think of it like a teacher grading an essay, the student saying "but I did
answer that on page 3," and the teacher re-checking and possibly raising the
grade.

### Where it fits in the app's flow

```
1. Upload PDF
        │
2. AI reads & understands the paper
        │
3. Specialist agents judge it (novelty, methods, evidence, citations)
        │
4. Self-reflection double-checks their work
        │
5. Final verdict drafted  →  e.g. "weak reject"
        │
6. 🧑 Human approves the verdict
        │
   ┌────┴──────────────────────────────────────┐
   │  Authors send a REBUTTAL                    │  ◄── this is the new part
   │  "We added the missing baseline, Table 4"   │
   └────┬──────────────────────────────────────┘
        │
7. RE-REVIEW: the app runs the WHOLE review again,
   but this time each agent also reads the rebuttal
   and reconsiders  →  new verdict "weak accept"
        │
8. 🧑 Human approves the revised verdict too
        │
9. App reports the change: weak_reject → weak_accept
   ("the rebuttal changed the reviewer's mind, for the better")
```

**The key point:** rebuttal feedback kicks in **after** the first full review is
approved (step 6 → 7). It is not a small separate step — it re-runs the *entire*
review pipeline, just with the authors' reply added to what each agent reads. So
the new verdict still goes through every quality check **and** still needs a
human's approval before it's final.

The rest of this document explains the same thing in technical detail.

## What a rebuttal is, and why re-review

Real peer review is a conversation, not a verdict handed down once. After
reviewers submit their assessment, the paper's **authors get to respond** — a
*rebuttal*. They answer specific concerns:

> "The reviewer says we lack an ImageNet baseline — we've added it in Table 4."
> "The reviewer flags reproducibility — we've now released the code and full
> hyperparameters."

Reviewers then **reconsider** in light of that response and frequently revise
their score. A rebuttal-aware re-review reproduces exactly this second round:
feed the system the authors' rebuttal, let it re-examine its own assessment,
and see whether — and which way — its recommendation moves.

## The core idea: a rebuttal is just a second run of the same graph

Rather than a bespoke "re-review" code path, a re-review is **the same compiled
review graph, run again, seeded with the author's `rebuttal_text`**. The four
assessment agents fold the rebuttal into their prompts and reconsider their
verdicts; reflection and final review then synthesize a revised recommendation,
which — like any recommendation — parks at the **mandatory human-approval gate**
before it is issued (see `LANGGRAPH_ORCHESTRATION.md` for the gate).

Reusing the whole graph is the point: the revised verdict passes through the
*exact same* grounding rules, self-reflection loop, and human-in-the-loop
approval as the original. There is no second, less-disciplined path a rebuttal
can sneak a verdict change through.

```
Original review ── final_recommendation: weak_reject ──▶ human approves
        │
        │  authors submit a rebuttal
        ▼
run_rebuttal_rereview(parsed_paper, rebuttal_text)
        │  (same graph, new thread, rebuttal_text seeded into state)
        ▼
paper_understanding ─┐
literature_rag ──────┤
figure_table ────────┤   the 4 assessment agents now also see the
methodology ─────────┤   AUTHOR REBUTTAL block and reconsider ──▶ reflection
citation ────────────┤                                              │
evidence_repro ──────┘                                              ▼
                                                            final_review
                                       final_recommendation: weak_accept
                                                                    │
                                                    human approves the REVISED verdict
                                                                    ▼
                            compare_recommendations() -> changed=True,
                            direction=more_favorable, steps=+2
```

## What "rebuttal feedback" actually is (the injected block)

`core/agents/revision.py::rebuttal_feedback_block(inputs)` turns the author's
rebuttal into a prompt-ready block that the assessment agents render. It returns
`""` when `inputs["rebuttal_text"]` is absent (i.e. a normal first-pass review),
so it can always be passed straight into `PromptManager.render()` without a
None-check — the same contract as `revision_feedback_block`.

When a rebuttal is present, each assessment agent's prompt gains:

```
AUTHOR REBUTTAL: The authors submitted the following rebuttal in response to
the initial review. Reconsider your assessment in light of it -- revise a
verdict only where the rebuttal genuinely resolves the concern with evidence,
and hold your ground where it does not (do not concede a point merely because
the authors pushed back):
<the author's rebuttal text>
```

### It is a *separate channel* from `revision_feedback` — on purpose

The codebase already had one injection channel: `revision_feedback`, set by the
self-reflection revision loop. These two look similar but say opposite things,
so they are kept distinct rather than overloading one field:

| | `revision_feedback` | `rebuttal_feedback` |
|---|---|---|
| Set by | The self-reflection/verifier step (internal) | An external author rebuttal |
| Framing | "You were flagged — fix this before finalizing." | "The authors responded — reconsider, but concede only where warranted." |
| Trigger key | `state["revision_feedback"]` | `state["rebuttal_text"]` |
| When | Bounded revision loop, mid-review | A whole new re-review run, after the original |

A single re-review can carry both (a rebuttal pass can still trigger the normal
reflection loop internally), which is why they're independent state keys, not
one reused field.

## The "did it change the reviewer's mind?" delta

`core/graph/rebuttal.py::compare_recommendations(original, revised)` reports how
the verdict moved on the 5-way scale
(`reject < weak_reject < borderline < weak_accept < accept`):

```python
{
  "original_recommendation": "weak_reject",
  "revised_recommendation":  "weak_accept",
  "changed":   True,
  "direction": "more_favorable",   # or "less_favorable" / "unchanged"
  "steps":     2,                   # signed: +ve toward accept, -ve toward reject
}
```

This delta is the actual output of interest — not just the new verdict, but
*whether the rebuttal was persuasive and in which direction*.

## How to use it

### Programmatically

```python
from core.graph.rebuttal import run_rebuttal_rereview

result = run_rebuttal_rereview(
    parsed_paper,                          # the same ParsedPaper the original ran on
    rebuttal_text="We added the missing ImageNet baseline (Table 4) and released code...",
    original_recommendation="weak_reject", # optional; enables the comparison
)
result["revised_review"]      # the drafted revised FinalReview (parked at approval)
result["comparison"]          # the before/after delta above
result["awaiting_approval"]   # True -- the revised verdict still needs human sign-off
result["thread_id"]           # resume this thread with an approval decision
```

`graph` and `thread_id` are injectable (the graph defaults to a fresh
`build_review_graph()`; the thread id defaults to `rebuttal-<uuid>` so it never
collides with the original run).

### Over HTTP

```
POST /api/rebuttal/{run_id}
{ "rebuttal_text": "We added the missing baseline (Table 4) and released code." }
```

Returns the revised review, the `comparison` delta, and a `rebuttal_run_id`.
Because the revised verdict also parks at the approval gate, you sign it off
through the existing flow:

```
POST /api/approve/{rebuttal_run_id}   { "decision": "approve" }
```

The server registers the rebuttal thread in its in-memory run table so the
same `resume_with_approval` path works for it unchanged (`server/pipeline.py`).

## What changed to add this

- `core/agents/revision.py` — added `rebuttal_feedback_block()` alongside the
  existing `revision_feedback_block()`.
- `core/config/prompts.yaml` — added a `{rebuttal_feedback}` placeholder to the
  4 assessment-agent prompts (novelty, methodology, citation,
  evidence_reproducibility).
- The 4 assessment agents — pass `rebuttal_feedback=rebuttal_feedback_block(inputs)`
  into their `.render()` call.
- `core/graph/state.py` — added the `rebuttal_text` state key.
- `core/graph/nodes.py` — the 4 assessment nodes thread `rebuttal_text` through
  to their agent.
- `core/graph/rebuttal.py` — new: `run_rebuttal_rereview()` +
  `compare_recommendations()`.
- `server/pipeline.py` / `server/main.py` — new `run_rebuttal_rereview()` server
  function + `POST /api/rebuttal/{run_id}` endpoint.

## Verified

`scripts/test_rebuttal_rereview.py` (fast, fully mocked — runs in
milliseconds): confirms the rebuttal actually reaches the assessment agents on a
re-review (and is absent on a first-pass review), the revised verdict is drafted
and parks at the approval gate, and the comparison is correct — a rebuttal that
resolves the reviewer's concern moves the recommendation
`weak_reject → weak_accept` (`direction=more_favorable, steps=+2`). The existing
`test_human_approval.py` and `test_graph_topology.py` still pass unchanged.
