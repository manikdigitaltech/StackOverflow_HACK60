"""
Literature RAG Agent: thin wrapper around kanishka's persistent Literature-RAG
Index B (core.rag.indexes.literature_index.LiteratureIndex). Unlike most
agents, this one makes NO LLM call -- it's pure retrieval, not generation.

Rewritten during the manik+kanishka merge: the original implementation wrapped
core.rag.retrievers.literature_retriever.LiteratureRetriever (manik's original
single bge-large FAISS index), which was retired in favor of kanishka's
two-index design (see core/rag/README.md). This agent now queries Index B
directly and adapts its RetrievalResult output into the LiteratureContext/
LiteratureMatch shape every downstream agent (Novelty, Citation) already
expects, so nothing above this agent had to change.

If no literature corpus has been built yet (data/literature_index/index.faiss
missing -- run core/rag/ingestion/build_corpus.py first), returns an empty
LiteratureContext rather than failing the whole review over one missing
optional-enrichment source.

Note: overrides BaseAgent's __init__ (llm/prompt_manager don't apply here)
since this agent genuinely only needs an index. The enforced contract
from BaseAgent is run() -- constructors are free to differ per agent.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from core.agents.base_agent import BaseAgent
from core.config.rag_settings import RAG_SETTINGS
from core.rag.indexes.literature_index import LiteratureIndex
from core.schemas.agent_output_schemas import LiteratureContext, LiteratureMatch, ParsedPaper


class LiteratureRAGAgent(BaseAgent):
    def __init__(self, index: Optional[LiteratureIndex] = None, top_k: int = 10, logger=None):
        self._index = index
        self._top_k = top_k
        self._logger = logger
        self._load_attempted = index is not None

    def _ensure_index(self) -> Optional[LiteratureIndex]:
        """Lazily load the persistent index on first use, at most once --
        a missing corpus is a valid, expected state (not every dev env has
        run build_corpus.py), not an error to retry every call."""
        if self._index is not None or self._load_attempted:
            return self._index
        self._load_attempted = True
        if not Path(RAG_SETTINGS.literature_index.index_path).exists():
            self._log("No literature corpus built yet -- returning empty literature context.")
            return None
        self._index = LiteratureIndex.load()
        return self._index

    def run(self, inputs: Dict[str, Any]) -> LiteratureContext:
        parsed_paper: ParsedPaper = inputs["parsed_paper"]
        query_text = f"{parsed_paper.title}. {parsed_paper.abstract}"
        self._log(f"Retrieving literature context for: {parsed_paper.title[:60]}...")

        index = self._ensure_index()
        if index is None:
            return LiteratureContext(query_text=query_text, matches=[])

        results = index.search_literature(query_text, k=self._top_k)
        matches = [
            LiteratureMatch(
                paper_id=r.metadata.get("paper_id", ""),
                title=r.metadata.get("title", ""),
                year=r.metadata.get("year"),
                chunk_text=r.content,
                section_type=None,
                similarity_score=r.score,
            )
            for r in results
        ]
        self._log(f"Retrieved {len(matches)} literature matches.")
        return LiteratureContext(query_text=query_text, matches=matches)
