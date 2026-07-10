# PeerRead Corpus Module - Technical Deep Dive

*Covers how real PeerRead ICLR-2017 data got into this repo and what it feeds.
Written for a team presentation - every number below came from an actual run
on this machine, not from the PeerRead paper or README.*

## The problem this closes

Two of the 9 review agents were code-complete but data-starved:

| Agent | Depended on | Without real data |
|---|---|---|
| Literature RAG Agent | `data/literature_index/` (Index B) | `search_literature` always returned `[]` |
| Citation Agent | Literature RAG's output | Checked 0 papers, and - worse - the LLM would **hallucinate** a single `"" ` (empty-title) "not cited" verdict when handed an empty retrieved list |
| Novelty Evaluation Agent (embedding-based) | `data/novelty_corpus/` | Only 2 hand-written toy papers to compare against |

`build_corpus.py` (Index B's ingestion script) and the embedding-based novelty
agent were both real, tested code with nothing to point them at - this module
is that "nothing" becoming "something real."

## Getting PeerRead data in without a full clone

The full `allenai/PeerRead` repo is several GB - mostly PDFs and
science-parse'd full text (`parsed_pdfs/`) that neither consumer needs; both
only read `reviews/*.json` (title, abstract, `accepted` label, id, venue).
A plain `git clone` or even a directory-level sparse-checkout of `data/`
timed out pulling PDFs we'd immediately discard.

What worked - a partial clone (`--filter=blob:none`) with a **file-pattern**
sparse-checkout, not a directory-level one:

```
/*
!/*/
/data/
!/data/*/
/data/iclr_2017/
/data/iclr_2017/*/
!/data/iclr_2017/*/pdfs/
!/data/iclr_2017/*/parsed_pdfs/
!/data/iclr_2017/*/reviews_raw/
```

Git's blob-level partial clone still fetches a blob the moment a path enters
scope, then leaves already-materialized files in place with a warning if a
later pattern excludes them - so the actual sequence was: checkout, then
`git sparse-checkout reapply`, then a manual `rm -rf` of the now-excluded
`pdfs/`/`parsed_pdfs/`/`reviews_raw/` directories. End result: **11 MB** of
`reviews/*.json` instead of several GB, cloned into a scratch directory and
copied into the repo at `data/peerread_raw/iclr_2017/{train,dev,test}/reviews/`
(gitignored - raw external data, not something this repo should carry).

| Split | Papers |
|---|---|
| train | 349 |
| dev | 40 |
| test | 38 |
| **Total** | **427** |

## Split policy - the same leakage guard on both consumers

Both `build_corpus.py` and the new `scripts/build_novelty_corpus.py` follow
one rule, because the eventual PeerRead evaluation harness (accuracy/F1/Cohen's
κ, the graded core - see `CONTEXT.md` §7) must score against a `test` split
that no upstream agent has ever seen:

```
CORPUS_SPLITS   = ("train", "dev")   # → goes into the corpus
HELD_OUT_SPLITS = ("test",)          # → ids collected, then excluded
```

`test`-split paper ids are loaded **only** to build an exclusion set, then
filtered out of the `train`+`dev` records before anything is embedded or
written to disk. 349 + 40 = **389 records survive** into both corpora; the
38 `test` ids never enter either index. (Index B additionally re-checks this
at query time via `exclude_paper_id` - see `RAG_ARCHITECTURE.md` - defense in
depth against a paper ever matching itself.)

## Two consumers, two embedding spaces, one source

```mermaid
flowchart TB
    PR[("PeerRead reviews/*.json\n427 papers, iclr_2017")]
    PR --> Split{split filter}
    Split -->|train+dev, 389| A["build_corpus.py\n(existing, Index B)"]
    Split -->|train+dev, 389| B["build_novelty_corpus.py\n(new, this session)"]
    Split -.->|test, 38 - excluded from both.-> X[Held out for the eval harness]

    A --> IdxB[("data/literature_index/\nFAISS IndexFlatIP, dim=768\nspecter2_base embeddings")]
    B --> IdxN[("data/novelty_corpus/*.json\n+ data/novelty_index/ (built on first run)\nall-MiniLM-L6-v2 embeddings, dim=384")]

    IdxB --> LitAgent[Literature RAG Agent] --> CiteAgent[Citation Agent]
    IdxN --> NovAgent[Embedding-based Novelty Evaluation Agent]
```

The two scripts intentionally don't share an embedding model or output format
— see `RAG_ARCHITECTURE.md` and `NOVELTY_AGENT.md` for why Index B needs
SPECTER2 (scientific-similarity-tuned) while the embedding-based novelty agent
uses a lighter general-purpose encoder over four separately-weighted sections.
`build_novelty_corpus.py` writes one flat JSON per paper
(`{title, abstract, sections: [], references: [], year}` - the schema
`core/agents/novelty/text_extractor.py` expects), namespaced
`iclr_2017_<id>.json` so they can't collide with the two pre-existing toy
seed papers (`1.json`, `2.json`, left in place).

## What actually runs, and how fast

Both builds ran on GPU once `settings.embeddings.device` was wired to `cuda`
(previously dead config - every embedding call was silently forced onto CPU
regardless of this setting; see the GPU-performance pass earlier this session):

```bash
python -m core.rag.ingestion.build_corpus --peerread-dir data/peerread_raw --venue iclr_2017
python -m scripts.build_novelty_corpus     --peerread-dir data/peerread_raw --venue iclr_2017
```

| Build | Records embedded | Output size | Wall time |
|---|---|---|---|
| Index B (`build_corpus.py`) | 389 | 1.7 MB (`index.faiss` + `records.jsonl`) | ~10s |
| Novelty corpus (`build_novelty_corpus.py`) | 389 (+2 toy) | 1.6 MB | (index built lazily on first agent run, ~15s for a full 391-paper self-evaluation) |

## Verified - before and after

**Literature RAG Agent**, same query, before vs. after:

```
before: [LiteratureRAGAgent] No literature corpus built yet -- returning empty literature context.
after:  [LiteratureRAGAgent] Retrieved 10 literature matches.
        [0.9010] GRAM: Graph-based Attention Model for Healthcare Representation Learning (2017)
        [0.8943] Deep Learning with Dynamic Computation Graphs (2017)
        ...
```

**Citation Agent**, same paper, before vs. after:

```
before: [CitationAgent] Citation quality: poor, 1 papers checked, 1 not cited.
        (hallucinated a coverage_verdict for a "" title - nothing was actually retrieved)
after:  [CitationAgent] Citation quality: poor, 10 papers checked, 10 not cited.
        (10 real ICLR-2017 titles individually checked against the paper's own reference list)
```

The "10 not cited" result itself is expected, not a bug: the test PDF used for
verification (a clinical-oncology orchestration framework) has essentially no
topical overlap with 2017-vintage ICLR ML papers, so a correctly-working
citation checker should find no genuine matches in its own references - that's
the corpus doing its job, not failing at it.

**Full 9-agent chain**, final review, re-run with the real corpus in place:
Novelty Agent's `overlapping_work` went from 0 cited overlaps to 1 (a
genuine, grounded match); Citation Agent's rating shifted to reflect real
coverage checking instead of a degenerate empty-corpus case.

**Embedding-based Novelty Evaluation Agent**, re-run against the 391-paper
corpus (self-evaluation - every paper scored against its 388 neighbors):
correctly produced uniformly **low novelty / weak-reject** scores in the
20s–30s range, which is the expected degenerate result of evaluating
same-venue, same-year ML papers against each other as background - a
meaningfully different (and correct) distribution from the 2-toy-paper
corpus's near-arbitrary scores.

Regression: 43/43 unit tests and the graph topology test still pass after
both builds and the citation-agent fix.

## The one behavioral fix this motivated

`core/agents/citation_agent.py` now short-circuits when
`literature_context.matches` is empty, returning a deterministic
`CitationAssessment` (empty `coverage_verdicts`, `"fair"` rating, an honest
reasoning string) instead of calling the LLM - which, given zero retrieved
papers, had been observed to invent one hallucinated verdict rather than
return zero. This fix stands on its own (an empty corpus is still a possible
runtime state - e.g. a paper in an unfamiliar subfield), but real data is what
surfaced it.

## Honest status - what's real vs. what's still a gap

| Piece | Status |
|---|---|
| `iclr_2017` train/dev/test reviews | **Real, downloaded, in `data/peerread_raw/` (gitignored)** |
| Index B (literature corpus) | **Real data now** - 389 vectors, GPU-embedded, query-verified |
| Novelty corpus | **Real data now** - 389 real papers + 2 original toy papers |
| `acl_2017` (the second venue in `IngestionSettings.peerread_venues`) | **Not fetched** - only `iclr_2017` was pulled, since `CONTEXT.md`'s graded core is scoped to the ICLR-2017 subset specifically |
| Train/dev/test split discipline | **Enforced identically in both build scripts** - `test` ids never enter either corpus |
| PeerRead evaluation harness (accuracy/F1/Cohen's κ against the `test` split) | **Still not started** - this module supplies the `train`/`dev` background corpus the harness's agents will query against; the harness itself (loading `test`, running the full graph, scoring) is separate, larger work (`CONTEXT.md` §7 item 3) |

**Bottom line:** both previously-empty corpora now hold real, correctly-scoped
ICLR-2017 data with the same test-split exclusion the eventual grading run
depends on. What's left is building the harness that actually consumes the
held-out `test` split - this module deliberately stopped short of touching it.
