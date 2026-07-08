# RAG Subsystem — Autonomous AI Paper Reviewer

This package implements **retrieval only**: two independent RAG indexes and
the tool functions that will let agents (built separately) query them. No
agent, LLM orchestration, or UI code lives here.

## Two indexes, two jobs

```
                    ┌───────────────────────────────────────────┐
                    │              PAPER UNDER REVIEW            │
                    └───────────────────────┬─────────────────────┘
                                            │
              ┌─────────────────────────────┴─────────────────────────────┐
              │                                                            │
              ▼                                                            ▼
  ┌───────────────────────────────┐                      ┌───────────────────────────────────┐
  │   INDEX A — Paper-RAG          │                      │  INDEX B — Literature-RAG          │
  │   (grounding)                  │                      │  (novelty)                         │
  │                                 │                      │                                     │
  │  lifecycle: EPHEMERAL           │                      │  lifecycle: PERSISTENT              │
  │    rebuilt per review run       │                      │    built once, offline              │
  │    discarded after               │                      │    loaded at startup                │
  │                                 │                      │                                     │
  │  granularity: chunk-level       │                      │  granularity: paper-level            │
  │    section-aware split          │                      │    one vector = title + abstract     │
  │    (abstract/intro/method/...)  │                      │                                     │
  │                                 │                      │  corpus: PeerRead                    │
  │  embed: bge-small-en-v1.5       │                      │  embed: specter2_base                │
  │  index: FAISS IndexFlatIP       │                      │  index: FAISS IndexFlatIP            │
  │  retrieve: dense + BM25, fused   │                      │  retrieve: dense only                │
  │            via RRF               │                      │                                     │
  │                                 │                      │  + leakage guard: excludes the       │
  │  tool: retrieve_from_paper()     │                      │    paper under review                │
  │                                 │                      │  tool: search_literature()            │
  └───────────────────────────────┘                      └───────────────────────────────────┘

                                    ┌───────────────────────────────────────┐
                                    │  LIVE SOURCES (enhancement, not        │
                                    │  guarantee — fail soft to [])          │
                                    │                                         │
                                    │  search_semantic_scholar()             │
                                    │  search_arxiv()                        │
                                    └───────────────────────────────────────┘
```

**Why two indexes, not one:** Index A answers "what does *this* paper say?" —
it must be rebuilt every run because the paper changes every run, and it
needs hybrid dense+sparse retrieval because reviewers care about exact terms
(metric names, hyperparameters) that pure semantic search blurs over. Index B
answers "has *anyone* said this before?" — it's built once from a large
static corpus, uses a citation-similarity-tuned embedding model instead of a
general one, and carries an explicit leakage guard so a paper can never
"find itself" as prior art. Merging them into one index would force a single
embedding model and a single lifecycle onto two questions that need
different answers to both.

## Build order (matches `TODO(Phase N)` markers in code)

1. `Chunk` model + section-aware chunker (`chunking/section_chunker.py`)
2. Index A build + `retrieve_from_paper` — dense only (`indexes/paper_index.py`, `embeddings/embedding_provider.py::BgeSmallEmbeddingProvider`)
3. Add BM25 + RRF hybrid to Index A (`retrieval/fusion.py`, rest of `paper_index.py`)
4. `CorpusRecord` + offline `build_corpus.py` (PeerRead → SPECTER2 → FAISS) + `search_literature` (`ingestion/build_corpus.py`, `indexes/literature_index.py`, `embeddings/embedding_provider.py::Specter2EmbeddingProvider`)
5. Live tools: `search_semantic_scholar`, `search_arxiv` with fail-soft wrappers (`live_sources/`)
6. Query helpers: `decompose_query`, then `hyde_query` (`retrieval/query_helpers.py`)
7. Wire tools into a ReAct loop — deferred to the agent-layer task; `retrieval/tools.py` is the stable interface that task will import against

## Concepts to learn, mapped to modules

| Module | Concept |
|---|---|
| `chunking/section_chunker.py` | Structure-aware chunking vs. naive fixed-window splitting; why chunk boundaries should respect document structure first, size second |
| `embeddings/embedding_provider.py` | Why embedding model choice is task-specific (general semantic search vs. scientific-similarity search) |
| `indexes/paper_index.py` | Dense (embedding/ANN) retrieval vs. sparse (BM25/lexical) retrieval — what each catches that the other misses |
| `retrieval/fusion.py` | Reciprocal Rank Fusion — merging rankings on incomparable score scales by rank position instead of raw score |
| `indexes/literature_index.py` | Persistent vs. ephemeral index lifecycles; leakage guards in retrieval-augmented evaluation |
| `ingestion/build_corpus.py` | Offline batch embedding pipelines; keeping a FAISS index and its metadata store in row-order lockstep |
| `retrieval/query_helpers.py` | Multi-query expansion (decomposition) and HyDE (Hypothetical Document Embeddings) as query-side retrieval-quality techniques |
| `live_sources/` | Graceful degradation — designing external dependencies so their failure mode is "reduced quality," never "system down" |
| `retrieval/tools.py` | The tool-interface pattern for agentic retrieval (ReAct-style): agents call stable tool signatures, not indexes directly |

## Running tests

```
pip install -r requirements.txt
pytest tests/unit -v
```

All test bodies currently `raise NotImplementedError` alongside the code
they cover — fill in the implementation, then the test, phase by phase.
