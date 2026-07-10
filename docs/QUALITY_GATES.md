# Quality Gates - DeepEval, RAGAS, and Prompt-Injection Guardrails

*Covers `core/eval/deepeval_quality.py`, `core/eval/ragas_quality.py`, and
`core/utils/guardrails.py`. All three were ported from the shelved
`vivek-RAGAS-deepeval` branch (see `docs/CONTEXT.md`'s branch history table)
— that branch was correctly judged "premature" at the time (nothing existed
yet for it to evaluate); once the eval harness existed, the *concepts* were
worth reviving, but none of the three files ran as originally written. This
doc covers what was actually broken and how it was fixed.*

## Why a second quality signal, alongside the PeerRead harness

`core/eval/peerread_harness.py` answers one question: does the *final*
accept/reject call match ground truth? It says nothing about *how* an agent
got there - a methodology verdict can happen to land on the right side of
"accept" while still being built on a hallucinated claim about the paper.
DeepEval and RAGAS answer that different question, at the level of
individual agent outputs and individual retrieval calls, rather than the
final synthesized recommendation.

## Guardrails (`core/utils/guardrails.py`) - the one file that worked as-is

Prompt-injection defenses for untrusted PDF text: strips zero-width
obfuscation characters, detects and defangs adversarial command patterns
(`"ignore all prior instructions"`, `"override system scoring"`, etc.),
wraps untrusted content in escaped XML tags so a paper can't break out of
its own prompt section, and flags LLM output that looks like a leaked system
prompt. Pure stdlib, no external dependencies, no bugs found.

Wired into `core/parsing/docling_parser.py`: every free-text field pulled
out of a PDF (title, abstract, section text, table markdown/captions,
figure captions, reference text) is sanitized once, immediately after
Docling parsing produces the `ParsedPaper`, before any of the 10 agents
ever sees it. One fix was needed: the original `ADVERSARIAL_PATTERNS` list
put an inline `(?i)` flag on every individual pattern, then joined them all
with `|` into one compiled regex - only the *first* inline flag in a joined
expression is honored by Python's `re` module; the rest raise a
`DeprecationWarning` today and will hard-error in a future Python version.
Fixed by moving case-insensitivity to a single `re.IGNORECASE` on the
compiled pattern instead. 10 unit tests in `tests/unit/test_guardrails.py`.

## DeepEval (`core/eval/deepeval_quality.py`) - per-agent reasoning quality

Runs two metrics against real agent output, both scoped narrowly (matching
the original design) to the two agents with the highest hallucination risk:

- **G-Eval** on the Novelty Agent's output - checks whether every novelty
  claim is actually compared against retrieved literature, in a strict
  area-chair tone, rather than asserting similarity/difference ungrounded.
- **Hallucination metric** on the Methodology Agent's output - checks
  claims against the paper's own content as retrieval context.

**Two real bugs fixed, both about local-model routing being claimed but
never implemented:**

1. The original code set `model=settings.llm.provider` (our internal alias,
   e.g. `"qwen2.5-7b"`) directly on DeepEval's metric constructors. DeepEval
   defaults to OpenAI-recognized model names; a bare alias string doesn't
   route to Ollama no matter what the code comment claimed. Fixed with
   `OllamaJudgeModel`, a real `DeepEvalBaseLLM` subclass wrapping this
   project's own `get_llm()` (`json_mode=False` - DeepEval's metric prompts
   expect free-form judge reasoning that DeepEval parses itself, not this
   project's `format="json"` enforcement, which would fight DeepEval's own
   parsing instead of helping it).
2. `parsed_paper.contributions` was read off `ParsedPaper`, which has no
   such field in this project's actual schema
   (`core/schemas/agent_output_schemas.py`) - contributions live on
   `PaperUnderstandingOutput.stated_contributions`, a different agent's
   real output. `evaluate_agent_outputs()` now takes that as an explicit
   argument instead of guessing at an attribute that was never there.

**Verified live**: real G-Eval score 0.8 (threshold 0.7, passed) on a real
Novelty Agent output; real Hallucination score 0.0 (threshold 0.3, passed,
lower is better) on a real Methodology Agent output - both fully local via
Ollama, no OpenAI key anywhere.

## RAGAS (`core/eval/ragas_quality.py`) - retrieval quality for Index A

Scores one (question, retrieved chunks, generated answer) triple against
four metrics: faithfulness, answer relevancy, context precision, context
recall (the last one only when a ground-truth answer is supplied - scoring
recall against an unknown ground truth is meaningless, not just optional,
so it's skipped rather than faked).

**Three real problems fixed, the first being a genuine environment blocker,
not just a design gap:**

1. **`ragas` didn't import at all.** Every version tried (latest 0.4.3, and
   the older 0.2.15 matching the original script's function-style API)
   eagerly executes
   `from langchain_community.chat_models.vertexai import ChatVertexAI` the
   moment you `import ragas` - and that submodule has been fully removed
   from the `langchain-community` version (0.4.2) this project already
   depends on for everything else. `ragas` has no version pin protecting
   against this. Downgrading `langchain-community` to satisfy `ragas` was
   ruled out - it's a shared dependency `langchain`/`langgraph`/
   `langchain-classic` all need current, and breaking those to fix an
   optional add-on would be exactly backwards. Fixed with a contained
   `sys.modules` shim (`_install_vertexai_import_shim()`, called before the
   first `import ragas`) that satisfies the one unused import with a stub
   class - we only ever evaluate against local Ollama, `ChatVertexAI` is
   never actually instantiated.
2. The original code used the now-deprecated `langchain_community.llms.Ollama`
   and RAGAS's now-deprecated function-style `ragas.metrics.faithfulness`
   API. Rewritten against the current `ragas.metrics.collections` class API
   and `ragas.llms.llm_factory`, which bridges to any OpenAI-compatible
   endpoint - Ollama exposes exactly that at `{base_url}/v1`, the
   officially documented path for a non-OpenAI local model (mirrors
   DeepEval's own local-routing fix above; two independent libraries, same
   underlying mistake of assuming OpenAI by default).
3. The original code passed `settings.embeddings.provider`
   (`"bge-large-en-v1.5"`, a SentenceTransformers model id) as an *Ollama*
   embedding model name for `AnswerRelevancy`'s internal similarity check —
   but this project never serves embeddings through Ollama at all; they run
   locally via `sentence-transformers`
   (`core/rag/embeddings/embedding_provider.py`). Fixed to use RAGAS's own
   `HuggingFaceEmbeddings` wrapper pointed at the *same* model + device
   Index A's own retrieval actually uses (`BAAI/bge-small-en-v1.5`,
   `settings.embeddings.device`), for consistency rather than introducing a
   third embedding model into the project.

A fourth bug surfaced only during live testing, not code review: the
`AsyncOpenAI`/sync-`OpenAI` client distinction. RAGAS's `.ascore()` calls
`agenerate()` internally, which requires an async client - using a
synchronous `OpenAI()` client raised `"Cannot use agenerate() with a
synchronous client"` on every metric. Fixed by constructing `AsyncOpenAI`
instead.

**Verified live** against real Index A retrieval (a real built paper index,
a real 3-chunk retrieval, a real synthesized answer): faithfulness 1.0,
answer_relevancy 0.85, context_precision 0.0 (correctly low - one of the
two test chunks genuinely wasn't relevant to the query, which is the metric
doing its job, not a bug), context_recall 0.5. One real characteristic
worth knowing: a full 4-metric `evaluate_retrieval()` call takes on the
order of 1-3 minutes under shared GPU load (each metric is its own LLM
judge call, run sequentially) - this is an optional, offline quality-gate
tool, not something to call from the live per-paper SSE pipeline.

## Installed, not yet wired into a specific eval run

Both `DeepEvalRunner` and `RagasRunner` are real, tested, importable classes
today (`deepeval`, `ragas`, `datasets` added to `requirements.txt`) - what's
*not* done yet is hooking either one into `scripts/run_peerread_evaluation.py`
as an automatic per-paper step. Given both add real LLM-call latency (RAGAS
alone: 1-3 extra minutes per paper), wiring them in should be an opt-in flag
on the eval script, not the default, so the existing ~35-40 minute/38-paper
accuracy run doesn't silently double or triple in length.
