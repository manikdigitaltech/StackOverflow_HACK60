"""Offline script: PeerRead -> CorpusRecord[] -> SPECTER2 embeddings -> persistent FAISS index.

Run once (or whenever the PeerRead source data changes), not per-review.
Output lands in `RAG_SETTINGS.literature_index.{index_path,records_path}`
and is loaded at process startup by `LiteratureIndex.load`.

Usage:
    python -m core.rag.ingestion.build_corpus \\
        --peerread-dir ../PeerRead/data \\
        --venue iclr_2017 --venue arxiv.cs.ai_2007-2017
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import faiss

from core.config.rag_settings import RAG_SETTINGS
from core.config.settings import settings
from core.rag.embeddings.embedding_provider import Specter2EmbeddingProvider
from core.rag.models import CorpusRecord

logger = logging.getLogger(__name__)

# Splits that go INTO the searchable corpus vs. splits held out as demo
# "papers under review" (the runtime leakage guard in LiteratureIndex.search_literature
# is the second line of defense; this list is the first).
CORPUS_SPLITS = ("train", "dev")
HELD_OUT_SPLITS = ("test",)


def load_peerread_records(peerread_dir: Path, venues: list[str], splits: tuple[str, ...]) -> list[CorpusRecord]:
    """Parse raw PeerRead `reviews/*.json` files into `CorpusRecord`s.

    Args:
        peerread_dir: the PeerRead repo's `data/` directory.
        venues: venue folder names to read, e.g. ["iclr_2017", "arxiv.cs.ai_2007-2017"].
        splits: which split subfolders to read within each venue (e.g. `CORPUS_SPLITS`
            for building the index, `HELD_OUT_SPLITS` for collecting excluded ids).

    Returns:
        One `CorpusRecord` per paper with a non-empty abstract. `paper_id` is
        prefixed with the venue name (e.g. "iclr_2017:304") since raw PeerRead
        ids are not unique across venues (ICLR uses small ints, arXiv uses
        arXiv ids - no guaranteed disjointness).
    """
    records: list[CorpusRecord] = []
    for venue in venues:
        for split in splits:
            reviews_dir = peerread_dir / venue / split / "reviews"
            if not reviews_dir.is_dir():
                logger.warning("Skipping missing directory: %s", reviews_dir)
                continue
            for json_path in sorted(reviews_dir.glob("*.json")):
                data = json.loads(json_path.read_text(encoding="utf-8"))
                abstract = data.get("abstract")
                if not abstract:
                    continue
                records.append(
                    CorpusRecord(
                        paper_id=f"{venue}:{data['id']}",
                        title=data["title"],
                        abstract=abstract,
                        year=_extract_year(data),
                        venue=data.get("conference"),
                        accepted=data.get("accepted"),
                    )
                )
    return records


def _extract_year(data: dict) -> int | None:
    """Best-effort year extraction: prefer an explicit `year` field, else parse
    a 4-digit year out of the `conference` string (e.g. "ICLR 2017 conference submission")."""
    if isinstance(data.get("year"), int):
        return data["year"]
    conference = data.get("conference") or ""
    for token in conference.split():
        if token.isdigit() and len(token) == 4:
            return int(token)
    return None


def exclude_review_split(
    records: list[CorpusRecord], excluded_paper_ids: set[str]
) -> list[CorpusRecord]:
    """Drop any record whose paper_id is in the set under active review.

    This is the build-time half of the leakage guard described in the
    README; `LiteratureIndex.search_literature`'s `exclude_paper_id` param is
    the runtime half, for defense in depth.

    Args:
        records: full parsed PeerRead corpus.
        excluded_paper_ids: paper_ids that must never appear in the
            literature index (e.g. the held-out test-split papers used as
            demo submissions).

    Returns:
        Filtered list of records.
    """
    return [r for r in records if r.paper_id not in excluded_paper_ids]


def build_and_persist_index(
    records: list[CorpusRecord],
    index_path: Path = RAG_SETTINGS.literature_index.index_path,
    records_path: Path = RAG_SETTINGS.literature_index.records_path,
    embedding_provider: Specter2EmbeddingProvider | None = None,
    batch_size: int = 32,
) -> None:
    """Embed records with SPECTER2, build a FAISS IndexFlatIP, and write both
    the index and a parallel records.jsonl to disk.

    Args:
        records: output of `load_peerread_records` (post `exclude_review_split`).
        index_path: where to write the FAISS index file.
        records_path: where to write the newline-delimited `CorpusRecord`
            file, in the same row order as the FAISS index - this ordering
            invariant is what lets `LiteratureIndex.load` zip them back
            together correctly.
        embedding_provider: injected for testability; defaults to a real
            `Specter2EmbeddingProvider`.
        batch_size: papers embedded per forward pass, to bound peak memory
            on CPU over a corpus of thousands of papers.
    """
    provider = embedding_provider or Specter2EmbeddingProvider(device=settings.embeddings.device)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.parent.mkdir(parents=True, exist_ok=True)

    all_vectors = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        # Literal "[SEP]" matches BERT/SciBERT's registered special-token text,
        # which the tokenizer maps to the real [SEP] token id during encoding -
        # this is the exact input format SPECTER2 was trained on.
        texts = [f"{r.title}[SEP]{r.abstract}" for r in batch]
        vectors = provider.embed(texts)
        all_vectors.append(vectors)
        logger.info("Embedded %d/%d records", min(start + batch_size, len(records)), len(records))

    import numpy as np

    matrix = np.vstack(all_vectors).astype("float32")
    dim = matrix.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    faiss.write_index(index, str(index_path))

    with records_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")

    logger.info("Wrote FAISS index (%d vectors, dim=%d) to %s", index.ntotal, dim, index_path)
    logger.info("Wrote %d records to %s", len(records), records_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peerread-dir", type=Path, required=True, help="Path to PeerRead's data/ directory")
    parser.add_argument(
        "--venue", action="append", dest="venues", required=True,
        help="Venue folder name under --peerread-dir; pass multiple times for multiple venues",
    )
    args = parser.parse_args()

    logger.info("Loading corpus splits %s for venues %s", CORPUS_SPLITS, args.venues)
    records = load_peerread_records(args.peerread_dir, args.venues, CORPUS_SPLITS)
    logger.info("Loaded %d candidate corpus records", len(records))

    logger.info("Loading held-out splits %s to build the exclusion set", HELD_OUT_SPLITS)
    held_out_records = load_peerread_records(args.peerread_dir, args.venues, HELD_OUT_SPLITS)
    excluded_ids = {r.paper_id for r in held_out_records}
    logger.info("Excluding %d held-out paper ids from the corpus", len(excluded_ids))

    records = exclude_review_split(records, excluded_ids)
    logger.info("Final corpus size after exclusion: %d records", len(records))

    build_and_persist_index(records)


if __name__ == "__main__":
    main()
