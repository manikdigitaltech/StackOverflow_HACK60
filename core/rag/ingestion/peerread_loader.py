"""
PeerRead ingestion (mandatory KB1 source, per hackathon requirements).

PeerRead is a static local dataset (cloned from allenai/PeerRead), not a
live API -- so this loader reads JSON files from disk rather than making
network calls. Each paper's full-text parse (via science-parse) typically
lives under a `parsed_pdfs/` folder per venue/split, alongside a `reviews/`
folder with reviewer scores (not used here -- we only need paper content
for KB1, not the review scores themselves).

This is written defensively because the exact repo layout can't be verified
from outside your machine: if venue folders aren't found where expected,
this prints out what it DOES find under the base path, so we can fix the
path pattern together from real information rather than guessing further.
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from core.config.settings import settings
from core.db.session import get_session
from core.db.repositories.paper_repository import PaperRepository
from core.db.repositories.chunk_repository import ChunkRepository
from core.db.models import PaperSource
from core.rag.embeddings.embedding_provider import EmbeddingProvider
from core.rag.vectorstore.faiss_index_manager import load_or_create_index, save_index
from core.utils.token_budget import truncate_to_token_budget

_EMBEDDING_MODEL_VERSION = "bge-large-en-v1.5"  # recorded per-chunk for future model-migration tracking
_MAX_SECTIONS_PER_PAPER = 3
_SECTION_TOKEN_BUDGET = 800


def _find_venue_dir(base_path: Path, venue: str) -> Optional[Path]:
    """Tries a few plausible repo layouts and returns the first that exists."""
    candidates = [
        base_path / "data" / venue,   # PeerRead/data/<venue>/  (the layout in the actual repo)
        base_path / venue,            # in case base_path already points at the inner data/ folder
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_parsed_pdf_json(file_path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [skip] Failed to parse {file_path.name}: {e}")
        return None


def _extract_paper_fields(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    PeerRead's parsed_pdfs JSON nests real content under 'metadata' in most
    versions of the dataset. Some tooling versions may have it flattened --
    this checks both shapes defensively.
    """
    metadata = raw.get("metadata", raw)
    title = metadata.get("title")
    abstract = metadata.get("abstractText") or metadata.get("abstract")
    sections = metadata.get("sections") or []
    year = metadata.get("year")

    if not title:
        return None  # a paper without a real title isn't useful for citation/novelty grounding,
                      # even if it has an abstract -- skip it and let another candidate file fill the slot

    return {
        "title": title or "(untitled)",
        "abstract": abstract or "",
        "sections": sections,  # list of {"heading": ..., "text": ...}, if present
        "year": year,
    }


def run_peerread_ingestion() -> None:
    base_path = Path(settings.ingestion.peerread_data_path)
    if not base_path.exists():
        print(f"ERROR: PeerRead path '{base_path}' doesn't exist. "
              f"Did the git clone into this path succeed?")
        return

    embedder = EmbeddingProvider()
    faiss_store = load_or_create_index()

    papers_ingested = 0
    papers_skipped = 0
    chunks_created = 0
    max_papers = settings.ingestion.max_peerread_papers

    with get_session() as session:
        paper_repo = PaperRepository(session)
        chunk_repo = ChunkRepository(session)

        for venue in settings.ingestion.peerread_venues:
            if papers_ingested >= max_papers:
                break

            venue_dir = _find_venue_dir(base_path, venue)
            if venue_dir is None:
                print(f"\n[WARNING] Could not find venue '{venue}' under {base_path}.")
                print(f"Contents of {base_path}:")
                try:
                    for item in sorted(base_path.iterdir())[:20]:
                        print(f"  - {item.name}")
                except Exception as e:
                    print(f"  (couldn't list directory: {e})")
                continue

            print(f"\nSearching for parsed PDFs under: {venue_dir}")
            parsed_pdf_files = sorted(venue_dir.glob("**/parsed_pdfs/*.json"))
            print(f"Found {len(parsed_pdf_files)} candidate files for venue '{venue}'.")

            for file_path in parsed_pdf_files:
                if papers_ingested >= max_papers:
                    break

                raw = _load_parsed_pdf_json(file_path)
                if raw is None:
                    papers_skipped += 1
                    continue

                fields = _extract_paper_fields(raw)
                if fields is None:
                    print(f"  [skip] {file_path.name}: no title/abstract found")
                    papers_skipped += 1
                    continue

                external_id = f"{venue}_{file_path.stem}"
                paper = paper_repo.get_or_create(
                    source=PaperSource.peerread,
                    external_id=external_id,
                    title=fields["title"],
                    abstract=fields["abstract"],
                    year=fields["year"],
                )

                # --- Build this paper's chunks: abstract + first N sections ---
                chunk_texts: List[Dict[str, str]] = []
                if fields["abstract"]:
                    chunk_texts.append({"text": fields["abstract"], "section_type": "abstract"})

                for section in fields["sections"][:_MAX_SECTIONS_PER_PAPER]:
                    text = section.get("text", "")
                    heading = section.get("heading") or "section"  # .get(key, default) only helps if key is MISSING; some PeerRead sections have "heading": null explicitly
                    if text and text.strip():
                        truncated = truncate_to_token_budget(text, _SECTION_TOKEN_BUDGET)
                        chunk_texts.append({"text": truncated, "section_type": heading[:64]})

                if not chunk_texts:
                    papers_skipped += 1
                    continue

                vectors = embedder.embed_batch([c["text"] for c in chunk_texts])
                faiss_ids = faiss_store.add_vectors(vectors)

                for chunk_info, faiss_id in zip(chunk_texts, faiss_ids):
                    chunk_repo.insert_chunk(
                        paper_id=paper.id,
                        faiss_id=faiss_id,
                        chunk_text=chunk_info["text"],
                        embedding_model_version=_EMBEDDING_MODEL_VERSION,
                        section_type=chunk_info["section_type"],
                    )
                    chunks_created += 1

                papers_ingested += 1
                if papers_ingested % 5 == 0:
                    print(f"  ...ingested {papers_ingested} papers so far")

        # session commits automatically on clean exit from get_session()

    save_index(faiss_store)

    print(f"\n--- PeerRead ingestion complete ---")
    print(f"Papers ingested: {papers_ingested}")
    print(f"Papers skipped:  {papers_skipped}")
    print(f"Chunks created:  {chunks_created}")
    print(f"FAISS index total vectors: {faiss_store.ntotal}")


if __name__ == "__main__":
    run_peerread_ingestion()
