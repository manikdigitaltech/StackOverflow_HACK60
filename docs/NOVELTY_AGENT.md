# Novelty Assessment — Two Independent Implementations

*There are deliberately two separate novelty implementations in this codebase,
answering different questions. Covers `core/agents/novelty/` (embedding-based)
and `core/agents/novelty_agent.py` (LLM-based).*

## Why two, not one

| | Embedding-based (`core/agents/novelty/`) | LLM-based (`core/agents/novelty_agent.py`) |
|---|---|---|
| Answers | "How similar is this, numerically, to known work?" | "*Which specific* contributions are novel, and why?" |
| Method | Cosine similarity over section embeddings | LLM reasoning grounded in retrieved literature |
| LLM calls | **None** | One structured JSON call |
| Hallucination risk | None — pure math | Mitigated via hard grounding constraints (below) |
| Output | Score (0–100), confidence, recommendation, decision trace | Per-contribution verdict + reasoning prose |
| Speed | Fast (embedding + FAISS lookup) | Slower (one LLM round-trip) |

They're complementary signals, not competitors — a reviewer could reasonably
see both: "87% similarity to a known paper" (deterministic) *and* "the
token-routing mechanism specifically is novel, but the compression scheme
overlaps with DynamicKV" (qualitative, grounded reasoning).

## Embedding-based Novelty Agent (`core/agents/novelty/`)

**Fully local, zero LLM calls** — sentence-transformers (`all-MiniLM-L6-v2`)
embeds four sections independently (abstract, methodology, conclusion,
references), compared against a local FAISS corpus.

### Pipeline

1. **`text_extractor.py`** — pulls title/abstract/methodology/conclusion/
   references/keywords out of a PeerRead-shaped JSON dict (handles both flat
   and `metadata`-nested shapes).
2. **`embedding_service.py`** — embeds each section independently. If the
   real model can't load (no network, e.g.), **silently falls back** to a
   deterministic `HashingVectorizer` of matching dimensionality — the agent
   keeps working end-to-end in a fully offline environment, degraded but never down.
3. **`faiss_retriever.py`** — `IndexFlatIP` over L2-normalized vectors
   (inner product = cosine similarity), with an `exclude_paper_id` option so
   a paper already in the indexed corpus can be evaluated against its
   neighbors without matching itself.
4. **`similarity_service.py`** — section-wise cosine similarity, combined
   into one **weighted overall score**: abstract 40%, methodology 35%,
   conclusion 15%, references 10%. Missing sections yield `None` (not `0.0`)
   so "no data" is never confused with "genuinely dissimilar" downstream.
5. **`novelty_scorer.py`** — `novelty_score = 100 − overall_similarity`,
   bucketed into 5 novelty bands (Very High → Very Low), each mapped to a
   recommendation (Strong Accept → Reject), plus a **near-duplicate override**:
   ≥95% similarity forces "Duplicate" / "Strong Reject" regardless of band.
   Confidence blends signal clarity (spread across section scores) with
   evidence volume (how full the retrieved neighborhood was).
6. **`decision_trace_builder.py`** — turns the numbers into a human-readable
   chain, e.g. `High methodology similarity -> High abstract similarity ->
   Closest match: 2 -> Low Novelty -> Weak Reject` — pure narration of
   already-computed numbers, no independent judgment.

### Verified

19/19 unit tests pass (real FAISS, not mocked). A real end-to-end smoke test
with genuine `all-MiniLM-L6-v2` embeddings against a synthetic 3-paper corpus
correctly identified a near-duplicate KV-cache-compression paper (85.6%
similarity) and produced a sensible Low Novelty / Reject verdict. Also
verified live through the orchestration graph against a real seeded corpus.

## LLM-based Novelty Agent (`core/agents/novelty_agent.py`)

The first agent in the 9-agent pipeline that combines two prior agents'
outputs (Paper Understanding + Literature RAG) — and the first where
grounding actually matters: a hallucinated "this overlaps with paper X"
citation would be a real, visible failure, not a rough edge.

### Grounding discipline (why this doesn't just make things up)

- The prompt **forces a per-item verdict**: exactly one `contribution_verdict`
  entry per stated contribution (novel / overlaps / partial) — no merging
  multiple contributions into one vague judgment, no skipping any.
- Any "overlaps" or "partial" verdict **must name the specific retrieved
  paper** justifying it.
- `overlapping_work` entries' `compared_paper_title` **must be copied exactly**
  from the retrieved literature text handed to the model — never paraphrased
  or invented. (`core/utils/grounding.py` provides a non-strict title-match
  checker for exactly this kind of verification, tolerant of trivial
  reformatting like an appended year.)
- If retrieved literature is weakly related or off-topic, the prompt
  explicitly instructs the model to say so honestly rather than force a
  comparison.

### Revision-aware (Phase 1 addition)

On a bounded self-reflection revision pass, this agent — like Methodology,
Citation, and Evidence/Reproducibility — receives a `revision_feedback` block
(via `core/agents/revision.py`) describing exactly what the reflection step
flagged, so a second pass is an actual revision addressing specific
criticism, not an identical re-run of the same prompt.

### Verified

Constructs and imports correctly; exercised for real inside the LangGraph
orchestration's real end-to-end Ollama run this session — correctly shifted
its verdict between passes (novelty_rating went from initial assessment to
"low" after the revision pass surfaced an overlap the first pass had missed),
demonstrating the revision loop is functionally meaningful, not cosmetic.
