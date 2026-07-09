"""
Per-agent output quality gates via DeepEval -- an optional, second signal
alongside the PeerRead accuracy/F1/kappa harness (core/eval/peerread_harness.py):
that harness checks whether the *final* accept/reject call matches ground
truth, this checks whether individual agents' *reasoning* is actually
grounded, not just directionally correct by luck.

Ported from the shelved vivek-RAGAS-deepeval branch (see docs/CONTEXT.md's
branch history) -- kept the two metrics chosen there (G-Eval for novelty
rigor, Hallucination for methodology grounding), fixed two real bugs that
blocked it from ever running:
  1. DeepEval defaults to OpenAI; passing a bare model-name string (e.g.
     "qwen2.5-7b") for `model=` does NOT route through local Ollama -- it
     silently expects an OpenAI-recognized model name. Needs a real
     DeepEvalBaseLLM subclass wrapping our own get_llm(), which is what
     OllamaJudgeModel below does.
  2. `parsed_paper.contributions` was read off ParsedPaper, which has no
     such field in this project's schema (core/schemas/agent_output_schemas.py)
     -- contributions live on PaperUnderstandingOutput.stated_contributions,
     the actual agent output, which this module now takes as an explicit
     argument instead of guessing at an attribute that was never there.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from deepeval.metrics import GEval, HallucinationMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams

from core.llm.llm_provider import get_llm
from core.schemas.agent_output_schemas import (
    MethodologyAssessment,
    NoveltyAssessment,
    PaperUnderstandingOutput,
    ParsedPaper,
)

logger = logging.getLogger(__name__)


class OllamaJudgeModel(DeepEvalBaseLLM):
    """Wraps this project's get_llm() (a real local Ollama ChatOllama client)
    as a DeepEval-compatible judge model, so G-Eval/Hallucination scoring
    runs fully locally like everything else in this project -- no OpenAI
    key, no network call outside localhost:11434.

    json_mode=False: DeepEval's own metric prompts expect free-form judge
    reasoning (which DeepEval parses itself), not this project's
    format="json" enforcement -- forcing json_mode here would fight
    DeepEval's own response parsing, not help it.
    """

    def __init__(self):
        self._client = get_llm(json_mode=False)
        super().__init__(model=self._client.model)

    def load_model(self):
        return self._client

    def generate(self, prompt: str, *args, **kwargs) -> str:
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        response = await self.model.ainvoke(prompt)
        return response.content

    def get_model_name(self) -> str:
        return f"ollama:{self._client.model}"


class DeepEvalRunner:
    """Runs DeepEval quality checks against a subset of agent outputs.

    Deliberately narrow scope (novelty + methodology only, matching the
    original design): these two carry the highest hallucination risk --
    novelty because it must ground every claim in retrieved literature it
    could easily misread, methodology because "the paper does X" claims
    are trivially checkable against paper content and trivially faked.
    """

    def __init__(self, judge_model: Optional[DeepEvalBaseLLM] = None):
        self._judge = judge_model or OllamaJudgeModel()

    def evaluate_agent_outputs(
        self,
        parsed_paper: ParsedPaper,
        paper_understanding: PaperUnderstandingOutput,
        agent_assessments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Args:
            parsed_paper: the real parsed paper (for grounding context).
            paper_understanding: PaperUnderstandingAgent's real output --
                the actual source of stated contributions (not a field on
                ParsedPaper itself, see module docstring).
            agent_assessments: dict with optional "novelty"/"methodology"
                keys holding the corresponding assessment (either the
                Pydantic model or its already-serialized dict).

        Returns:
            Dict keyed by agent name, each with a score/reason/passed verdict.
            An agent absent from `agent_assessments`, or a check that itself
            errors (judge model unreachable, malformed response), is simply
            omitted from the result rather than raising -- a quality-gate
            failure must never take down the run it's grading.
        """
        results: Dict[str, Any] = {}

        if "novelty" in agent_assessments:
            try:
                results["novelty"] = self._evaluate_novelty(
                    parsed_paper, paper_understanding, agent_assessments["novelty"]
                )
            except Exception as exc:
                logger.warning("DeepEval novelty check failed, skipping: %s", exc)

        if "methodology" in agent_assessments:
            try:
                results["methodology"] = self._evaluate_methodology(
                    parsed_paper, agent_assessments["methodology"]
                )
            except Exception as exc:
                logger.warning("DeepEval methodology check failed, skipping: %s", exc)

        return results

    def _evaluate_novelty(
        self, parsed_paper: ParsedPaper, paper_understanding: PaperUnderstandingOutput, novelty_data: Any
    ) -> Dict[str, Any]:
        novelty_json = self._as_json(novelty_data)
        contributions = "\n".join(f"- {c}" for c in paper_understanding.stated_contributions)
        source_context = f"Abstract: {parsed_paper.abstract}\nStated contributions:\n{contributions}"

        test_case = LLMTestCase(
            input=source_context,
            actual_output=novelty_json,
            retrieval_context=[source_context],
        )
        metric = GEval(
            name="Scientific Novelty Rigor",
            criteria=(
                "Determine whether the actual output correctly distinguishes between "
                "genuinely novel work and incremental steps without hallucinating "
                "unmentioned similarities."
            ),
            evaluation_steps=[
                "Check if every novelty claim is explicitly compared against retrieved literature context.",
                "Verify that the tone is that of a strict journal reviewer (ICLR/NeurIPS Area Chair).",
                "Ensure the output strictly follows valid structured JSON rules.",
            ],
            evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.RETRIEVAL_CONTEXT],
            threshold=0.7,
            model=self._judge,
        )
        metric.measure(test_case)
        return {"score": metric.score, "reason": metric.reason, "passed": metric.is_successful()}

    def _evaluate_methodology(self, parsed_paper: ParsedPaper, methodology_data: Any) -> Dict[str, Any]:
        methodology_json = self._as_json(methodology_data)
        paper_context = self._paper_context_text(parsed_paper)

        test_case = LLMTestCase(
            input="Evaluate methodological soundness, missing ablations, and baseline weaknesses.",
            actual_output=methodology_json,
            context=[paper_context],
        )
        metric = HallucinationMetric(threshold=0.3, model=self._judge)
        metric.measure(test_case)
        return {"hallucination_score": metric.score, "passed": metric.is_successful(), "reason": metric.reason}

    @staticmethod
    def _as_json(value: Any) -> str:
        if hasattr(value, "model_dump_json"):
            return value.model_dump_json()
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        return str(value)

    @staticmethod
    def _paper_context_text(parsed_paper: ParsedPaper) -> str:
        sections = "\n\n".join(f"[{s.name}]\n{s.text}" for s in parsed_paper.sections[:10])
        return f"Title: {parsed_paper.title}\nAbstract: {parsed_paper.abstract}\n\n{sections}"
