"""
Novelty Evaluation Agent - fully local paper novelty scoring.

Public API:
    from core.agents.novelty import NoveltyEvaluationAgent
"""

from .novelty_evaluation_agent import NoveltyEvaluationAgent, NoveltyEvaluationAgentError

__all__ = ["NoveltyEvaluationAgent", "NoveltyEvaluationAgentError"]
__version__ = "1.0.0"
