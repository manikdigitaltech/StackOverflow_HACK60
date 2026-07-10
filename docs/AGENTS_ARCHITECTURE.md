# The Review Agents - Architecture & Individual Agents

*Covers `core/agents/*.py` (the 9 LLM-based agents + Literature RAG) and the
shared infrastructure they're built on. Novelty is covered separately
(`NOVELTY_AGENT.md`) since it has two distinct implementations; the
orchestration wiring them together is covered in `LANGGRAPH_ORCHESTRATION.md`.*

## Shared foundation

### `BaseAgent` (`base_agent.py`)

A deliberately minimal ABC: `__init__(llm, prompt_manager, logger=None)` and
one abstract method, `run(inputs: Dict) -> Any`. Agents are kept pure —
**no DB session, no FAISS access** (except `LiteratureRAGAgent`, which wraps
a retriever precisely because that's its whole job) - so every agent is
unit-testable with just a mocked or real LLM, with persistence and retrieval
concerns pushed to the layer above (the graph / server).

### Structured output, without relying on function-calling (`structured_output.py`)

**Deliberate design choice**: rather than LangChain's `with_structured_output()`
(which, with Ollama, depends on function-calling support that varies
unpredictably by model/version), every agent:

1. Calls `get_llm(json_mode=True)` - forces Ollama's native `format="json"`
   syntax enforcement (syntactically valid JSON *guaranteed*, schema
   conformance **not** guaranteed - the prompt still has to describe the
   shape).
2. Strips markdown code fences from the raw response.
3. Validates the parsed JSON against the agent's actual Pydantic schema via
   `invoke_for_json()`.
4. On failure, raises `StructuredOutputError` with **both** the validation
   error and the raw model response attached - so a failure is debuggable
   without needing to re-run the LLM call blind.

### Schemas (`core/schemas/agent_output_schemas.py`)

Every agent's output is a fully-typed Pydantic model matching the problem
statement's requirements closely:

- `PaperUnderstandingOutput`, `NoveltyAssessment`, `MethodologyAssessment`,
  `CitationAssessment`, `EvidenceReproducibilityAssessment`,
  `FigureTableSummary`, `ReflectionNotes`, `FinalReview`.
- **Deterministic fields, not LLM-guessed ones**: e.g. `FinalReview.
  missing_baselines` is *copied* from `MethodologyAssessment`'s own grounded
  list at synthesis time, never independently re-derived by the Final Review
  LLM call - this avoids drift between what Methodology actually found and
  what the final report claims.
- `FinalReview.final_recommendation` is a 5-way enum
  (`reject`/`weak_reject`/`borderline`/`weak_accept`/`accept`) matching the
  problem statement's required rating scale exactly.

### `assessment_formatters.py`

Pure formatting functions (`format_understanding`, `format_novelty`, etc.)
that turn a structured assessment back into readable text for a *downstream*
agent's prompt (used heavily by Reflection and Final Review, which both need
to read every prior agent's output). Kept separate from the agents
themselves so formatting logic doesn't get duplicated or drift between the
two consumers.

## The individual agents

### `PaperUnderstandingAgent`

Reads the token-budgeted paper context (via `context_builder.build_paper_context`)
and produces a condensed briefing - summary, stated contributions, key terms —
that every downstream agent consumes instead of re-reading the full paper
each time. First agent in the pipeline; needs only `parsed_paper`.

**Verified live** against real Ollama this session (correct summary,
contributions, and key terms for a synthetic sparse-attention paper).

### `LiteratureRAGAgent`

The one agent that makes **no LLM call** - pure retrieval, wrapping
`core.rag.indexes.literature_index.LiteratureIndex` (Index B). Builds a
title+abstract query, calls `search_literature()`, and adapts the RAG
subsystem's `RetrievalResult` shape into the agent-schema's
`LiteratureMatch`/`LiteratureContext` shape. Lazily loads the index at most
once; if no corpus has been built yet, returns an honest empty context
rather than erroring the whole review over one missing optional-enrichment source.

*(Rewritten during the RAG merge - the original implementation depended on a
retired single-index retriever; see `RAG_ARCHITECTURE.md`.)*

**Live-source merge (on by default, disable via
`RAG_SETTINGS.live_sources.enable_arxiv` / `enable_semantic_scholar`)**:
the agent supplements Index B hits with live results from
`core.rag.retrieval.tools.search_arxiv` / `search_semantic_scholar` —
the seam those clients were already built behind. Index B matches always
take priority; live results only fill remaining slots up to `top_k`,
deduped by normalized title. Each `LiteratureMatch` carries a `source`
field (`"literature_index"` / `"arxiv"` / `"semantic_scholar"`) so
`NoveltyAgent` and `CitationAgent`'s formatters can label a live web hit
(`[via arXiv]`) distinctly from a curated corpus hit in the prompt - a
live hit is weaker "not cited" evidence for Citation's coverage-gap check
than a curated one. Both live clients are resilient by design (never
raise, degrade to `[]`), so an outage never breaks the run.

### `MethodologyAgent`

Evaluates methodological soundness - baseline comparisons, ablation
coverage, hyperparameter justification, experimental setup clarity,
statistical rigor - needing only the paper itself, no literature comparison.
**Grounding rule for `missing_baselines`**: only lists a method the paper's
own text (e.g. Related Work) mentions by name but doesn't experimentally
compare against - never invents a baseline from general ML knowledge the
paper never referenced. Produces exactly one verdict per fixed aspect
(forces coverage, not cherry-picking).

### `CitationAgent`

Checks whether literature retrieved as highly relevant is actually present in
the paper's *own* reference list - a coverage-gap check, not a formatting
check. Must produce exactly one `coverage_verdicts` entry per retrieved
paper (no silent omissions), and titles must be copied exactly from the
retrieved text. Deliberately skips `build_paper_context()` to leave full
token budget for the paper's (potentially long) reference list instead.

### `EvidenceReproducibilityAgent`

Checks two separate things: (1) are the paper's headline quantitative claims
actually backed by numbers in its own tables (evidence uses tables as the
*only* ground truth - a claim with no matching table number is "unsupported,"
not charitably assumed true), and (2) is the work realistically reproducible
based only on what's explicitly stated (missing compute/dataset/hyperparameter
details are flagged as real gaps, never assumed to have been done "the
standard way"). The first agent needing table data, not just section text.

### `FigureTableAgent`

Summarizes figures/tables from **caption text + table markdown only** - its
own docstring is explicit that real visual interpretation is the
Vision-Optional extension, which this agent doesn't do (see
`VLM_FIGURE_TABLE.md` for the piece that actually does). Also runs a
**deterministic, non-LLM consistency check**: does the paper's prose
reference "Figure 4" when only 2 figures were extracted? A cheap, reliable
way to catch a parsing gap before ever blaming the model for missing content
it was never given. Skips the LLM call entirely if no figures/tables exist.

### `AdversarialCriticAgent`

Added after the PeerRead evaluation harness's first real numbers showed a
specific, concrete failure mode worth targeting: agents that settle on a
comfortable middling rating ("fair", "adequate") rather than actually
committing to a strong verdict, which upstream biases the whole review
toward over-predicting accept (see `docs/CONTEXT.md` §7 item 3's "honest
read" for the numbers that motivated this).

Attacks **Methodology, Citation, and Evidence & Reproducibility only** —
deliberately not Novelty, which already has its own grounding discipline
and a second, independent embedding-based signal (see `NOVELTY_AGENT.md`).
For each of the three target assessments, finds the single weakest verdict
and constructs a genuine counter-argument against it - `attacked_verdict`
must quote or closely paraphrase an exact verdict from the assessment being
attacked (never a vague summary of the whole thing), and the counter-argument
must explain specifically why the cited evidence doesn't support it as
strongly as claimed. Outputs `AdversarialCritique` (a list of `attacks`,
each with `source_agent`/`attacked_verdict`/`counter_argument`/`severity`,
plus a `weakest_agent` verdict).

**Wired as an input to `ReflectionAgent`, not as its own veto**: the
critique doesn't unilaterally force a flag - Reflection is instructed to
cross-check each attack against the paper content and the original
assessment itself before deciding whether to raise a matching flag,
treating a "major" attack that holds up as strong evidence for a "major"
flag, but never rubber-stamping the critic's severity label blindly.

**Graph wiring** (see `LANGGRAPH_ORCHESTRATION.md`): runs on its own 3-source
AND-join (`methodology`/`citation`/`evidence_reproducibility`, NOT
`novelty`) in parallel with `reflection`'s own join. Deliberately has **no**
direct edge from `prepare_revision` - it re-fires automatically on a
revision pass because its 3 sources get re-triggered anyway, and adding a
second explicit edge would reintroduce the exact OR-trigger-not-AND-join bug
documented in `LANGGRAPH_ORCHESTRATION.md`. Confirmed empirically (not just
by reasoning) in `scripts/test_graph_topology.py`: fires exactly twice
across one revision pass, never receives `novelty_assessment`.

**Verified live**: real attacks against a real paper's Methodology/Citation/
Evidence output (e.g. attacking a "fair" soundness rating for lacking
baseline comparisons/ablations), Reflection's flags traced directly back to
the critic's attacks in the same run, `final_review` still completed a full
report despite the added node and revision pass.

### `ReflectionAgent`

Reads all four assessments (Novelty, Methodology, Citation, Evidence &
Reproducibility) **plus the Adversarial Critique** plus the original paper,
and flags anything speculative, unsupported, inconsistent, or an attack that
holds up under its own verification. Does **not** re-run retrieval or
re-parse the paper - purely reviews what the other agents already concluded.
Outputs `ReflectionNotes` (flags with severity, `needs_revision` - true only
if at least one flag is `"major"` - and overall confidence), which drives
the orchestration graph's bounded revision loop.

### `FinalReviewAgent`

Synthesizes all seven upstream outputs into the exact structure the problem
statement asks for: Paper Summary, Strengths, Weaknesses, Questions for
Authors, Novelty/Citation/Reproducibility/Evidence analysis, Missing
Baselines, Final Recommendation, Confidence. Deliberately does **not**
re-read the paper's own text - all paper-grounding already happened
upstream; this agent only synthesizes from other agents' already-grounded
outputs, keeping its own prompt smaller despite covering the most sources.

## Verified

All 10 agent classes (9 here + the LLM-based Novelty Agent) import and
construct successfully together. `PaperUnderstandingAgent` and
`FinalReviewAgent` verified individually against real Ollama /
mocked-but-realistic fixture data; all 10 verified together in real,
non-mocked end-to-end runs through the LangGraph orchestration and the live
SSE dashboard (correct revision-loop behavior - including the Adversarial
Critic re-firing on the second pass - coherent final output, real MySQL
persistence of every assessment).
