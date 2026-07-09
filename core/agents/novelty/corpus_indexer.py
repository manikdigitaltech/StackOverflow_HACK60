"""
corpus_indexer.py

Loads a directory of paper JSON files, extracts + embeds every paper,
and builds the FAISS index used for retrieval. Kept separate from the
agent orchestrator so indexing (a batch, offline concern) is decoupled
from per-paper evaluation (an online concern).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .config import get_logger
from .embedding_service import EmbeddingService
from .faiss_retriever import FaissRetriever
from .models import PaperEmbedding, PaperRecord
from .text_extractor import PaperExtractionError, PaperTextExtractor

logger = get_logger(__name__)


class CorpusIndexerError(Exception):
    """Raised when the corpus cannot be loaded or indexed."""


class CorpusIndexer:
    """Loads, embeds, and indexes a local corpus of paper JSON files.

    Example:
        >>> indexer = CorpusIndexer()
        >>> indexer.load_directory("data/corpus")
        >>> retriever = indexer.retriever
        >>> embeddings_by_id = indexer.embeddings_by_id
    """

    def __init__(
        self,
        extractor: PaperTextExtractor = None,
        embedding_service: EmbeddingService = None,
        retriever: FaissRetriever = None,
    ) -> None:
        # Dependency injection: each collaborator can be swapped/mocked independently.
        self.extractor = extractor or PaperTextExtractor()
        self.embedding_service = embedding_service or EmbeddingService()
        self.retriever = retriever or FaissRetriever()
        self.embeddings_by_id: Dict[str, PaperEmbedding] = {}
        self.records_by_id: Dict[str, PaperRecord] = {}

    def load_directory(self, directory: Path) -> None:
        """Load every ``*.json`` file in a directory, embed it, and build the FAISS index.

        Args:
            directory: Directory containing paper JSON files.

        Raises:
            CorpusIndexerError: If the directory is missing/empty or no
                papers could be embedded.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise CorpusIndexerError(f"Corpus directory not found: {directory}")

        filepaths = sorted(directory.glob("*.json"))
        if not filepaths:
            raise CorpusIndexerError(f"No JSON files found in '{directory}'")

        records: List[PaperRecord] = []
        for filepath in filepaths:
            try:
                paper_json = json.loads(filepath.read_text(encoding="utf-8"))
                paper_id = str(paper_json.get("id") or filepath.stem)
                record = self.extractor.extract(paper_json, paper_id=paper_id)
                records.append(record)
            except (OSError, json.JSONDecodeError, PaperExtractionError) as exc:
                logger.error("Skipping unreadable/invalid file '%s': %s", filepath, exc)

        if not records:
            raise CorpusIndexerError(f"No valid papers could be extracted from '{directory}'")

        self.records_by_id = {r.paper_id: r for r in records}

        embeddings = self.embedding_service.embed_batch(records)
        if not embeddings:
            raise CorpusIndexerError("No embeddings could be generated for the corpus")

        self.embeddings_by_id = {e.paper_id: e for e in embeddings}
        self.retriever.build(embeddings)

        logger.info("Indexed corpus: %d papers loaded, %d embedded and indexed", len(records), len(embeddings))
