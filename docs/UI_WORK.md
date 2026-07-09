# Live Pipeline Dashboard — UI Work

*Covers `server/` (FastAPI backend) and `ai_paper_reviewer_ui.html` (the
dashboard). This is a real-time monitoring surface for the pipeline, built
and iterated on this session — not a mockup.*

## Why a custom FastAPI + SSE dashboard, not Streamlit

Streamlit is the problem statement's suggested UI tool, and is genuinely the
faster path for a Python-only team (no separate frontend, built-in
upload/session-state/chart widgets). It wasn't used here for one specific
reason: **Streamlit's rerun-on-interaction execution model isn't built for
push-based live updates.** Watching a pipeline stage update *while it runs*
— the actual requirement — needs Server-Sent Events or WebSockets pushing
state as it changes, not a script re-executing top-to-bottom on each
interaction. A real backend + SSE gets genuine push-based real-time for free.

manik's original Streamlit page stubs (`app/main.py`, `app/pages/*.py`) were
never built out (still 0 lines each) — left in place as dead scaffolding in
case a later, non-real-time page (history, settings, an eventual metrics
dashboard) wants Streamlit's native chart/table widgets instead.

## Backend (`server/`)

- **`server/pipeline.py`** — `run_pipeline(run_id, pdf_path)`: a generator
  yielding one event per real stage as it *actually* finishes (parse → vision
  → chunk → paper-RAG build → literature-RAG → novelty). Stages that don't
  exist in the codebase yet are yielded with `status="not_implemented"`
  rather than faked — the UI never shows an agent "completing" work no code
  did. Also exposes `query_paper_index()` (live hybrid retrieval against the
  just-built paper index) and `check_system_health()` (real probes: Ollama
  reachability + pulled models, MySQL reachability, docling installed,
  literature index built, checkpoint DB present — every status is a real
  check, not a hardcoded "Healthy").
- **`server/main.py`** — `POST /api/upload`, `GET /api/stream/{run_id}` (SSE),
  `GET /api/query/{run_id}`, `GET /api/health`. Serves the dashboard HTML
  directly via scoped routes, not a directory mount over the whole repo.

## Frontend evolution (`ai_paper_reviewer_ui.html`)

Started from a hand-designed static mockup with 100% hardcoded fake data;
every piece below replaced fake JS with real backend calls, one iteration at
a time based on direct feedback:

1. **Real wiring** — file upload → `EventSource` → live-updating agent cards,
   a top progress bar computed from real stage counts, a docked detail panel
   showing real per-stage data (parsed title/sections, VLM figure
   descriptions, RAG chunk counts + a live query box).
2. **Sidebar removed.** A full persistent nav sidebar was mostly dead weight
   for a page with exactly one job (watch a live run) — replaced with a
   56px top bar (logo + 3 tabs), freeing width for a real 3-pane layout.
3. **3-pane redesign**: paper info + compact pipeline status list (left,
   sticky) · full-detail Live Activity log (center, wide) · docked
   auto-following detail panel (right, sticky) — mirroring the same sticky
   treatment on both flanks so neither scrolls out of view while the center
   log grows.
4. **Redundancy fix.** The activity log and the docked panel were both
   rendering the exact same derived summary sentence for a completed stage —
   split their jobs: the log is now a lean *timing* record (stage, status,
   timestamp/elapsed), the docked panel is the *only* place with the rich
   derived description.
5. **Honesty fix.** Every non-running status was defaulting to the same
   green checkmark — including the 7 "not yet implemented" stages, which
   read as if they'd succeeded. Now `not_implemented`/`not_available`/
   `skipped` get a distinct neutral dash icon; only a real completed stage
   gets the green check.
6. **Terminal-style trace, "like Claude's own output."** Rebuilt the Live
   Activity log as a monospace, dark-console trace with entries threaded on
   a connecting vertical line + colored dot per entry (green/blue-pulsing/
   red/gray by status) — a commit-graph-style timeline rather than a stack
   of generic cards.
7. **Feels live, not just accurate.** Each new line now slides/fades in on
   arrival, and its detail message **types out character-by-character with a
   blinking cursor** rather than snapping fully-formed into view — the
   underlying event was already real and instant; this reveals it the way a
   generated response reads, rather than a static log dump. (Explicitly not
   literal token-by-token LLM streaming — that requires a real LLM call
   streaming, which nothing here does yet; this is an honest client-side
   reveal of an already-complete message.)

## Performance note

Removed `backdrop-filter: blur()` from the repeated per-stage cards (up to 11
in the compact left column) — stacking that many expensive blur layers next
to sticky-positioned panels is a known cause of janky scroll; a solid darker
background reads almost identically at a fraction of the repaint cost.

## Verified

Real end-to-end runs through the actual dashboard this session, including
uploading genuine sample PDFs and watching Docling parsing, RAG chunking/
indexing, and the embedding-based Novelty Agent complete live with correct,
real data at every stage.

## Not yet done

Doesn't show the new LangGraph orchestration's per-agent progress yet — still
displays the old placeholder cards for methodology/citation/evidence/
reflection/final review (see `LANGGRAPH_ORCHESTRATION.md`). No human-in-the-loop
approval view wired to real data yet (Phase 2).
