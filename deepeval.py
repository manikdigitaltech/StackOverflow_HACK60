import json
from typing import Dict, Any
from deepeval.test_case import LLMTestCase
from deepeval.metrics import GEval, HallucinationMetric
from deepeval.evaluation import evaluate
from core.config.settings import AppSettings

class DeepEvalRunner:
    def __init__(self, settings: AppSettings):
        # DeepEval defaults to OpenAI, so we configure it to route through your local Ollama instance
        self.settings = settings
        self.model_name = settings.llm.provider  # e.g., 'llama3.1-8b' or 'qwen2.5-7b'
        self.base_url = settings.llm.base_url   # e.g., 'http://localhost:11434'

    def evaluate_agent_outputs(self, parsed_paper: Any, agent_assessments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates the output quality of individual review agents using G-Eval custom criteria.
        """
        results = {}
        
        # 1. Evaluate Novelty Agent for text grounding and scientific framing
        if "novelty" in agent_assessments:
            novelty_data = agent_assessments["novelty"]
            # Extract raw text context from parsed paper schema for evaluation grounding
            source_context = f"Abstract: {parsed_paper.abstract}\nContributions: {parsed_paper.contributions if hasattr(parsed_paper, 'contributions') else ''}"
            
            test_case = LLMTestCase(
                input=source_context,
                actual_output=json.dumps(novelty_data),
                retrieval_context=[source_context]
            )
            
            # Define G-Eval metric for peer-review tone and rigor
            novelty_metric = GEval(
                name="Scientific Novelty Rigor",
                criteria="Determine whether the actual output correctly distinguishes between genuinely novel work and incremental steps without hallucinating unmentioned similarities.",
                evaluation_steps=[
                    "Check if every novelty claim is explicitly compared against retrieved literature context.",
                    "Verify that the tone is that of a strict journal reviewer (ICLR/NeurIPS Area Chair).",
                    "Ensure the output strictly follows valid structured JSON rules."
                ],
                threshold=0.7,
                model=self.model_name
            )
            
            novelty_metric.measure(test_case)
            results["novelty"] = {
                "score": novelty_metric.score,
                "reason": novelty_metric.reason,
                "passed": novelty_metric.is_successful()
            }

        # 2. Evaluate Methodology Agent for hallucination against original paper
        if "methodology" in agent_assessments:
            method_data = agent_assessments["methodology"]
            paper_context = json.dumps(parsed_paper, default=lambda o: o.__dict__)
            
            test_case = LLMTestCase(
                input="Evaluate methodological soundness, missing ablations, and baseline weaknesses.",
                actual_output=json.dumps(method_data),
                retrieval_context=[paper_context]
            )
            
            # Catching unsupported claims made by the reviewer agent
            hallucination_metric = HallucinationMetric(threshold=0.3, model=self.model_name)
            hallucination_metric.measure(test_case)
            
            results["methodology"] = {
                "hallucination_score": hallucination_metric.score,
                "passed": hallucination_metric.is_successful(),
                "reason": hallucination_metric.reason
            }
            
        return results