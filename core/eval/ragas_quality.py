"""
RAG retrieval-quality scoring via RAGAS -- checks whether Index A (the
paper's own hybrid dense+BM25 index, see RAG_ARCHITECTURE.md) actually
retrieves chunks that ground what an agent says, and whether the retrieved
chunks are relevant/complete for the question asked. Complementary to
core/eval/deepeval_quality.py (which checks agent *reasoning* quality) and
core/eval/peerread_harness.py (which checks the *final* accept/reject call).

Ported from the shelved vivek-RAGAS-deepeval branch (see docs/CONTEXT.md's
branch history) -- kept the four metrics chosen there (faithfulness,
answer relevancy, context recall, context precision), but the rest needed
real fixes:

1. `ragas` itself doesn't import in this project's environment: every
   version tried (0.4.3, 0.2.15) eagerly does
   `from langchain_community.chat_models.vertexai import ChatVertexAI` at
   `import ragas` time, and that submodule was fully removed from the
   `langchain-community` version this project already depends on for
   everything else (0.4.2) -- ragas has no version pin protecting against
   this. Downgrading langchain-community to satisfy ragas would risk
   breaking langchain/langgraph/langchain-classic, which the whole review
   pipeline depends on. Fix: inject a stub module for that one unused
   import (we only ever evaluate against local Ollama, never Vertex AI)
   before importing ragas -- contained, doesn't touch any real dependency
   version. See `_install_vertexai_import_shim()`.
2. The original code used `langchain_community.llms.Ollama` (deprecated)
   and RAGAS's now-deprecated function-style `ragas.metrics.faithfulness`
   API. Rewritten against the current `ragas.metrics.collections` class API
   and `ragas.llms.llm_factory`, which bridges to Ollama through its
   OpenAI-compatible endpoint (`{base_url}/v1`) -- the officially
   documented path for a local/non-OpenAI model, not a bare "model name
   string" the way the original DeepEval half of this same branch tried
   and failed the same way (see deepeval_quality.py's module docstring).
3. The original code passed `settings.embeddings.provider`
   ("bge-large-en-v1.5", a SentenceTransformers model id) as an *Ollama*
   embedding model name -- Ollama never serves that model, embeddings in
   this project run locally via sentence-transformers, not through Ollama
   at all (core/rag/embeddings/embedding_provider.py). Fixed to use RAGAS's
   own HuggingFaceEmbeddings wrapper pointed at the SAME model + device
   Index A actually uses (BAAI/bge-small-en-v1.5, settings.embeddings.device),
   for consistency rather than introducing a third embedding model into the
   project.
"""
from __future__ import annotations

import logging
import sys
import types
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _install_vertexai_import_shim() -> None:
    """Must run before the first `import ragas` anywhere in the process.
    See module docstring point 1. Idempotent -- safe to call repeatedly."""
    module_path = "langchain_community.chat_models.vertexai"
    if module_path in sys.modules:
        return
    stub = types.ModuleType(module_path)

    class ChatVertexAI:  # never actually instantiated -- ragas only imports the symbol
        pass

    stub.ChatVertexAI = ChatVertexAI
    sys.modules[module_path] = stub


_install_vertexai_import_shim()

from openai import AsyncOpenAI  # noqa: E402 -- must follow the shim; async since evaluate_retrieval is async

from ragas.embeddings import HuggingFaceEmbeddings  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.metrics.collections import (  # noqa: E402
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from core.config.settings import settings  # noqa: E402
from core.llm.llm_provider import get_llm  # noqa: E402


class RagasRunner:
    """Runs RAGAS metrics against one (question, retrieved_contexts, answer)
    triple, optionally with a ground_truth for context_recall. All four
    metrics run fully locally: the judge LLM is this project's own Ollama
    model via its OpenAI-compatible endpoint, and embeddings are the same
    bge-small-en-v1.5 (on settings.embeddings.device) Index A itself uses.
    """

    def __init__(self):
        model_tag = get_llm().model  # resolves the real pulled Ollama tag, not our internal alias
        client = AsyncOpenAI(base_url=f"{settings.llm.base_url}/v1", api_key="ollama-local-no-key-needed")
        self._llm = llm_factory(model_tag, client=client)
        self._embeddings = HuggingFaceEmbeddings(
            model="BAAI/bge-small-en-v1.5", device=settings.embeddings.device
        )

        self._faithfulness = Faithfulness(llm=self._llm)
        self._answer_relevancy = AnswerRelevancy(llm=self._llm, embeddings=self._embeddings)
        self._context_recall = ContextRecall(llm=self._llm)
        self._context_precision = ContextPrecision(llm=self._llm)

    async def evaluate_retrieval(
        self,
        query: str,
        retrieved_contexts: List[str],
        answer: str,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Scores one retrieval+generation instance.

        Args:
            query: the question that was asked (e.g. a claim-grounding query
                an agent sent to retrieve_from_paper).
            retrieved_contexts: the chunk texts Index A actually returned.
            answer: the agent's generated text that used those chunks.
            ground_truth: the ideal answer, if known -- only context_recall
                uses this; it's skipped (not scored) when omitted, since
                recall against an unknown ground truth is meaningless, not
                just optional.

        Returns:
            Dict with one entry per metric that completed. A metric that
            itself errors (judge model unreachable, malformed response) is
            omitted rather than raising -- this is a supplementary quality
            signal, never allowed to break the run it's scoring.
        """
        results: Dict[str, Any] = {}

        for name, coro in (
            ("faithfulness", self._faithfulness.ascore(
                user_input=query, response=answer, retrieved_contexts=retrieved_contexts,
            )),
            ("answer_relevancy", self._answer_relevancy.ascore(
                user_input=query, response=answer,
            )),
            ("context_precision", self._context_precision.ascore(
                user_input=query, retrieved_contexts=retrieved_contexts,
                reference=ground_truth or answer,
            )),
        ):
            try:
                score = await coro
                results[name] = score.value if hasattr(score, "value") else score
            except Exception as exc:
                logger.warning("RAGAS %s scoring failed, skipping: %s", name, exc)

        if ground_truth:
            try:
                score = await self._context_recall.ascore(
                    user_input=query, retrieved_contexts=retrieved_contexts, reference=ground_truth,
                )
                results["context_recall"] = score.value if hasattr(score, "value") else score
            except Exception as exc:
                logger.warning("RAGAS context_recall scoring failed, skipping: %s", exc)

        return results
