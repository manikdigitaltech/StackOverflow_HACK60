"""
Literature RAG Agent: thin wrapper around LiteratureRetriever (built and
proven in Step 6). Unlike most agents, this one makes NO LLM call -- it's
pure retrieval, not generation. Its purpose is just to give graph nodes a
uniform .run() interface around the retriever, same as every other agent.

Note: overrides BaseAgent's __init__ (llm/prompt_manager don't apply here)
since this agent genuinely only needs a retriever. The enforced contract
from BaseAgent is run() -- constructors are free to differ per agent.
"""

from typing import Any, Dict, Optional
from core.agents.base_agent import BaseAgent
from core.schemas.agent_output_schemas import ParsedPaper, LiteratureContext
from core.rag.retrievers.literature_retriever import LiteratureRetriever


class LiteratureRAGAgent(BaseAgent):
    def __init__(self, retriever: Optional[LiteratureRetriever] = None, logger=None):
        self._retriever = retriever or LiteratureRetriever()
        self._logger = logger

    def run(self, inputs: Dict[str, Any]) -> LiteratureContext:
        parsed_paper: ParsedPaper = inputs["parsed_paper"]
        self._log(f"Retrieving literature context for: {parsed_paper.title[:60]}...")
        context = self._retriever.retrieve(parsed_paper)
        self._log(f"Retrieved {len(context.matches)} literature matches.")
        return context
