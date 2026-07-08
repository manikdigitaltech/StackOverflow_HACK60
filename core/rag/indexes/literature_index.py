"""Index B — Literature-RAG: persistent, paper-level, dense-only.

Built once offline by `core.rag.ingestion.build_corpus` from the PeerRead
dataset, then loaded at process startup and reused across every review run.
Answers "has this contribution been done before?" by nearest-neighbor search
over `specter2_base` embeddings of title+abstract. Never rebuilt per-request —
that's what makes it cheap to query from every paper review.
"""
from __future__ import annotations

from pathlib import Path

import faiss

from core.config.rag_settings import RAG_SETTINGS
from core.rag.embeddings.embedding_provider import Specter2EmbeddingProvider
from core.rag.models import CorpusRecord, RetrievalResult


class LiteratureIndex:
    """Loads and queries the persistent PeerRead FAISS index."""

    def __init__(self, embedding_provider: Specter2EmbeddingProvider | None = None):
        self._embedding_provider = embedding_provider or Specter2EmbeddingProvider()
        self._faiss_index: faiss.Index | None = None
        self._records: list[CorpusRecord] = []  # parallel to FAISS row order

    @classmethod
    def load(
        cls,
        index_path: Path = RAG_SETTINGS.literature_index.index_path,
        records_path: Path = RAG_SETTINGS.literature_index.records_path,
        embedding_provider: Specter2EmbeddingProvider | None = None,
    ) -> "LiteratureIndex":
        """Load a prebuilt index + its parallel records file from disk.

        Args:
            index_path: path to the FAISS index file written by
                `build_corpus.py`.
            records_path: path to the newline-delimited JSON `CorpusRecord`
                file, in the same row order as the FAISS index.
            embedding_provider: injected for testability; defaults to a real
                `Specter2EmbeddingProvider`.

        Returns:
            A ready-to-query `LiteratureIndex`.
        """
        instance = cls(embedding_provider=embedding_provider)
        instance._faiss_index = faiss.read_index(str(index_path))
        instance._records = [
            CorpusRecord.model_validate_json(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if instance._faiss_index.ntotal != len(instance._records):
            raise ValueError(
                f"FAISS index has {instance._faiss_index.ntotal} vectors but "
                f"records file has {len(instance._records)} rows - they must "
                "be built together and stay in lockstep."
            )
        return instance

    def search_literature(
        self,
        query: str,
        k: int = RAG_SETTINGS.literature_index.default_top_k,
        exclude_paper_id: str | None = None,
    ) -> list[RetrievalResult]:
        """Nearest-neighbor search over the persistent literature corpus.

        Args:
            query: novelty-style question, e.g. the paper's stated
                contribution in one sentence.
            k: number of results to return.
            exclude_paper_id: paper_id of the submission under review; any
                hit with this id is dropped even if it slipped into the
                corpus, per the leakage guard described in the README.

        Returns:
            Up to `k` `RetrievalResult`s with source="literature_rag".
        """
        if self._faiss_index is None:
            raise RuntimeError("LiteratureIndex.load(...) must be called before search_literature.")

        # over-fetch so we still have k results left after dropping the
        # excluded paper, if it happens to be among the nearest neighbors.
        fetch_k = k + 1 if exclude_paper_id else k
        query_vector = self._embedding_provider.embed([query])
        scores, indices = self._faiss_index.search(query_vector, fetch_k)

        results: list[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            record = self._records[idx]
            if exclude_paper_id is not None and record.paper_id == exclude_paper_id:
                continue
            results.append(
                RetrievalResult(
                    source="literature_rag",
                    score=float(score),
                    content=f"{record.title}\n{record.abstract}",
                    metadata=record.model_dump(),
                )
            )
            if len(results) == k:
                break
        return results
