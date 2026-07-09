"""
novelty_evaluation_agent.py

The Novelty Evaluation Agent: orchestrates extraction, embedding, FAISS
retrieval, similarity scoring, novelty/confidence/recommendation
scoring, and decision-trace generation into a single structured report.

This is the only public entry point most callers need. Every
collaborator is injected (constructor-based dependency injection), so
each stage can be swapped, mocked, or extended independently without
modifying this class (Open/Closed Principle).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .config import DEFAULT_PATHS, TOP_K, get_logger
from .corpus_indexer import CorpusIndexer
from .decision_trace_builder import DecisionTraceBuilder
from .embedding_service import EmbeddingService, EmbeddingServiceError
from .faiss_retriever import FaissRetriever, FaissRetrieverError
from .models import NoveltyReport, PaperRecord
from .novelty_scorer import NoveltyScorer
from .similarity_service import SimilarityService
from .text_extractor import PaperExtractionError, PaperTextExtractor

logger = get_logger(__name__)


class NoveltyEvaluationAgentError(Exception):
    """Raised when the agent cannot produce a novelty evaluation."""


class NoveltyEvaluationAgent:
    """Evaluates how novel a paper is against a local corpus.

    Fully local: no online APIs, no LLM calls. Uses sentence-transformers
    (all-MiniLM-L6-v2) for embeddings and a local FAISS index for
    retrieval.

    Example:
        >>> agent = NoveltyEvaluationAgent()
        >>> agent.index_corpus("data/corpus")
        >>> report = agent.evaluate(paper_json, paper_id="304.pdf")
        >>> print(report.to_dict())
    """

    def __init__(
        self,
        extractor: Optional[PaperTextExtractor] = None,
        embedding_service: Optional[EmbeddingService] = None,
        retriever: Optional[FaissRetriever] = None,
        similarity_service: Optional[SimilarityService] = None,
        scorer: Optional[NoveltyScorer] = None,
        trace_builder: Optional[DecisionTraceBuilder] = None,
        top_k: int = TOP_K,
    ) -> None:
        self.extractor = extractor or PaperTextExtractor()
        self.embedding_service = embedding_service or EmbeddingService()
        self.retriever = retriever or FaissRetriever()
        self.similarity_service = similarity_service or SimilarityService()
        self.scorer = scorer or NoveltyScorer()
        self.trace_builder = trace_builder or DecisionTraceBuilder()
        self.top_k = top_k

        self.indexer = CorpusIndexer(
            extractor=self.extractor, embedding_service=self.embedding_service, retriever=self.retriever
        )
        logger.info("NoveltyEvaluationAgent initialized (top_k=%d)", top_k)

    # ------------------------------------------------------------------
    # Corpus management
    # ------------------------------------------------------------------

    def index_corpus(self, directory: Path = DEFAULT_PATHS.corpus_dir) -> None:
        """Load, embed, and index every paper JSON in a directory.

        Args:
            directory: Directory containing paper JSON files.

        Raises:
            NoveltyEvaluationAgentError: If the corpus cannot be indexed.
        """
        try:
            self.indexer.load_directory(directory)
        except Exception as exc:  # noqa: BLE001
            raise NoveltyEvaluationAgentError(f"Failed to index corpus '{directory}': {exc}") from exc

    def save_index(self, directory: Path = DEFAULT_PATHS.index_dir) -> None:
        """Persist the FAISS index to disk for reuse without re-embedding."""
        try:
            self.retriever.save(directory)
        except FaissRetrieverError as exc:
            raise NoveltyEvaluationAgentError(f"Failed to save index: {exc}") from exc

    def load_index(self, directory: Path = DEFAULT_PATHS.index_dir) -> None:
        """Load a previously saved FAISS index from disk."""
        try:
            self.retriever.load(directory)
        except FaissRetrieverError as exc:
            raise NoveltyEvaluationAgentError(f"Failed to load index: {exc}") from exc

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, paper_json: Dict, paper_id: str) -> NoveltyReport:
        """Evaluate a single paper's novelty against the indexed corpus.

        Args:
            paper_json: Parsed paper JSON (PeerRead schema).
            paper_id: Identifier to assign to this paper.

        Returns:
            A ``NoveltyReport``.

        Raises:
            NoveltyEvaluationAgentError: If any stage fails (extraction,
                embedding, retrieval, scoring).
        """
        try:
            record = self.extractor.extract(paper_json, paper_id=paper_id)
            report = self._evaluate_record(record)
        except (PaperExtractionError, EmbeddingServiceError, FaissRetrieverError) as exc:
            raise NoveltyEvaluationAgentError(f"Failed to evaluate paper '{paper_id}': {exc}") from exc
        except NoveltyEvaluationAgentError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise NoveltyEvaluationAgentError(f"Unexpected failure evaluating paper '{paper_id}': {exc}") from exc

        return report

    def evaluate_indexed_paper(self, paper_id: str) -> NoveltyReport:
        """Evaluate a paper that is already part of the indexed corpus.

        Useful for batch scoring an entire corpus against itself.

        Args:
            paper_id: Identifier of a paper already indexed via ``index_corpus``.

        Returns:
            A ``NoveltyReport``.

        Raises:
            NoveltyEvaluationAgentError: If the paper is not indexed or scoring fails.
        """
        record = self.indexer.records_by_id.get(paper_id)
        if record is None:
            raise NoveltyEvaluationAgentError(f"Paper '{paper_id}' is not in the indexed corpus")
        try:
            return self._evaluate_record(record, exclude_self=True)
        except Exception as exc:  # noqa: BLE001
            raise NoveltyEvaluationAgentError(f"Failed to evaluate indexed paper '{paper_id}': {exc}") from exc

    def evaluate_all_indexed(self) -> List[NoveltyReport]:
        """Evaluate every paper currently in the indexed corpus.

        Returns:
            List of ``NoveltyReport``, skipping any paper that fails
            (e.g. isolated papers with no comparable neighbours).
        """
        reports = []
        for paper_id in self.indexer.records_by_id:
            try:
                reports.append(self.evaluate_indexed_paper(paper_id))
            except NoveltyEvaluationAgentError as exc:
                logger.error("Skipping paper '%s': %s", paper_id, exc)
        return reports

    def save_report(self, report: NoveltyReport, filepath: Path) -> None:
        """Write a report to disk as pretty-printed JSON."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        logger.info("Saved report for '%s' to %s", report.paper_id, filepath)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_record(self, record: PaperRecord, exclude_self: bool = False) -> NoveltyReport:
        if self.retriever.size == 0:
            raise NoveltyEvaluationAgentError("No corpus has been indexed yet - call index_corpus() first")

        target_embedding = self.embedding_service.embed(record)
        query_vector = target_embedding.get(self.retriever.section)
        if query_vector is None:
            raise NoveltyEvaluationAgentError(
                f"Paper '{record.paper_id}' has no '{self.retriever.section}' text to search with"
            )

        top_similar = self.retriever.search(
            query_vector, top_k=self.top_k, exclude_paper_id=record.paper_id if exclude_self else None
        )
        if not top_similar:
            raise NoveltyEvaluationAgentError(f"No similar papers found for '{record.paper_id}'")

        # The closest match (highest similarity) anchors the novelty/section scoring.
        closest_id = top_similar[0].paper_id
        closest_embedding = self.indexer.embeddings_by_id.get(closest_id)
        if closest_embedding is None:
            raise NoveltyEvaluationAgentError(f"Closest match '{closest_id}' has no stored embedding")

        breakdown, overall_similarity = self.similarity_service.compute(target_embedding, closest_embedding)
        novelty_score, confidence, novelty_band, recommendation = self.scorer.score(
            overall_similarity, breakdown, top_similar
        )
        trace = self.trace_builder.build(
            breakdown, novelty_band=novelty_band, recommendation=recommendation, closest_paper_id=closest_id
        )

        return NoveltyReport(
            paper_id=record.paper_id,
            novelty_score=novelty_score,
            overall_similarity=overall_similarity,
            confidence=confidence,
            recommendation=recommendation,
            similarity_breakdown=breakdown,
            top_similar_papers=top_similar,
            decision_trace=trace,
        )
