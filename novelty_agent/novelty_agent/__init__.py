"""
Novelty Evaluation Agent - fully local paper novelty scoring.

Public API:
    from novelty_agent import NoveltyEvaluationAgent
"""

from .novelty_evaluation_agent import NoveltyEvaluationAgent, NoveltyEvaluationAgentError

__all__ = ["NoveltyEvaluationAgent", "NoveltyEvaluationAgentError"]
__version__ = "1.0.0"
