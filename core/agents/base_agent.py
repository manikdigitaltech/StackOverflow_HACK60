"""
Abstract contract every scientific-evaluation agent implements.

Agents are deliberately kept pure: no DB session, no FAISS access (except
LiteratureRAGAgent, which wraps the retriever), no checkpoint awareness.
Persistence is the graph NODE's job (Step 8), not the agent's -- this
keeps every agent unit-testable with just a mocked or real LLM.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class AgentExecutionError(Exception):
    pass


class BaseAgent(ABC):
    def __init__(self, llm, prompt_manager, logger=None):
        self._llm = llm
        self._prompts = prompt_manager
        self._logger = logger

    @abstractmethod
    def run(self, inputs: Dict[str, Any]) -> Any:
        """Execute the agent's task and return a structured, validated output."""
        raise NotImplementedError

    def _log(self, message: str) -> None:
        if self._logger:
            self._logger.info(message)
        else:
            print(f"[{self.__class__.__name__}] {message}")
