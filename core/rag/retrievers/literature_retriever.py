"""
Joins FAISS's vector similarity search back to MySQL's chunk/paper
metadata -- this is the actual "Literature RAG" retrieval agents call at
review time. FAISS answers "which vectors are closest"; this module
answers "...and what paper/text do those actually belong to."
"""

from typing import List, Optional
from core.config.settings import settings
from core.schemas.agent_output_schemas import ParsedPaper, LiteratureContext, LiteratureMatch
from core.rag.embeddings.embedding_provider import EmbeddingProvider
from core.rag.vectorstore.faiss_index_manager import load_or_create_index
from core.db.session import get_session
from core.db.repositories.chunk_repository import ChunkRepository


class LiteratureRetriever:
    def __init__(self):
        self._embedder = EmbeddingProvider()
        self._faiss_store = load_or_create_index()

    def retrieve(self, parsed_paper: ParsedPaper, k: Optional[int] = None) -> LiteratureContext:
        k = k or settings.faiss.top_k
        query_text = self._build_query_text(parsed_paper)
        query_vector = self._embedder.embed_query(query_text)

        raw_results = self._faiss_store.similarity_search(query_vector, k=k)
        if not raw_results:
            return LiteratureContext(query_text=query_text, matches=[])

        faiss_ids = [faiss_id for faiss_id, _score in raw_results]
        score_by_faiss_id = {faiss_id: score for faiss_id, score in raw_results}

        matches: List[LiteratureMatch] = []
        with get_session() as session:
            chunk_repo = ChunkRepository(session)
            chunks = chunk_repo.get_active_by_faiss_ids(faiss_ids)

            for chunk in chunks:
                matches.append(LiteratureMatch(
                    paper_id=chunk.paper_id,
                    title=chunk.paper.title,       # lazy-loaded relationship, fine while session is open
                    year=chunk.paper.year,
                    chunk_text=chunk.chunk_text,
                    section_type=chunk.section_type,
                    similarity_score=score_by_faiss_id.get(chunk.faiss_id, 0.0),
                ))

        # MySQL's WHERE faiss_id IN (...) doesn't preserve FAISS's rank order -- restore it.
        matches.sort(key=lambda m: m.similarity_score, reverse=True)

        return LiteratureContext(query_text=query_text, matches=matches)

    @staticmethod
    def _build_query_text(parsed_paper: ParsedPaper) -> str:
        # Abstract carries the most concentrated "what is this paper about"
        # signal for retrieval purposes -- title alone is often too short/generic.
        return f"{parsed_paper.title}. {parsed_paper.abstract}"
