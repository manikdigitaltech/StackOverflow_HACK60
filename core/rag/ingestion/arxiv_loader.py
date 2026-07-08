"""
arXiv ingestion (OPTIONAL KB1 enrichment source -- PeerRead is mandatory;
this supplements it with more topically relevant/recent literature).

Unlike PeerRead (a static local dataset), arXiv is a LIVE API -- this
loader makes a network call to export.arxiv.org's Atom XML query endpoint
at ingestion time. Only runs when settings.ingestion.enable_arxiv=True.

Only title + abstract are available from this API (no full text), so
each arXiv paper contributes exactly one chunk (its abstract) -- unlike
PeerRead papers, which get an abstract chunk plus a few section chunks.
"""

import re
import xml.etree.ElementTree as ET
from urllib.request import urlopen, Request
from urllib.parse import urlencode
from typing import List, Dict, Any

from core.config.settings import settings
from core.db.session import get_session
from core.db.repositories.paper_repository import PaperRepository
from core.db.repositories.chunk_repository import ChunkRepository
from core.db.models import PaperSource
from core.rag.embeddings.embedding_provider import EmbeddingProvider
from core.rag.vectorstore.faiss_index_manager import load_or_create_index, save_index

_ARXIV_API_URL = "http://export.arxiv.org/api/query"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
_EMBEDDING_MODEL_VERSION = "bge-large-en-v1.5"
_VERSION_SUFFIX_PATTERN = re.compile(r"v\d+$")


def _normalize_arxiv_id(raw_id: str) -> str:
    """
    arXiv can return different VERSIONS of the same paper (e.g. '2401.12345v1'
    and '2401.12345v2') as separate search entries. For our purposes -- one
    paper, one KB1 entry -- these should collapse to the same identity.
    Stripping the trailing version suffix makes get_or_create() correctly
    treat them as the same paper instead of creating duplicates.
    """
    return _VERSION_SUFFIX_PATTERN.sub("", raw_id)


def _fetch_arxiv_entries(query: str, max_results: int) -> List[Dict[str, Any]]:
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = f"{_ARXIV_API_URL}?{urlencode(params)}"
    print(f"[arxiv_loader] Querying: {url}")

    req = Request(url, headers={"User-Agent": "PaperReviewerHackathonBot/1.0"})
    with urlopen(req, timeout=30) as response:
        xml_bytes = response.read()

    root = ET.fromstring(xml_bytes)
    entries = []
    seen_ids = set()
    for entry in root.findall("atom:entry", _ATOM_NS):
        arxiv_id_full = entry.findtext("atom:id", default="", namespaces=_ATOM_NS)
        raw_id = (arxiv_id_full or "").rstrip("/").split("/")[-1]
        arxiv_id = _normalize_arxiv_id(raw_id)

        if arxiv_id in seen_ids:
            continue  # same paper, different version already captured in this fetch
        seen_ids.add(arxiv_id)

        title = (entry.findtext("atom:title", default="", namespaces=_ATOM_NS) or "").strip().replace("\n", " ")
        summary = (entry.findtext("atom:summary", default="", namespaces=_ATOM_NS) or "").strip()
        published = entry.findtext("atom:published", default="", namespaces=_ATOM_NS) or ""
        year = int(published[:4]) if published[:4].isdigit() else None

        if not title or not summary:
            continue

        entries.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": summary,
            "year": year,
            "url": arxiv_id_full,
        })
    return entries


def run_arxiv_ingestion() -> None:
    if not settings.ingestion.enable_arxiv:
        print("[arxiv_loader] Skipped -- settings.ingestion.enable_arxiv is False. "
              "Set INGESTION__ENABLE_ARXIV=true in .env to enable.")
        return

    query = settings.ingestion.arxiv_search_query
    max_results = settings.ingestion.arxiv_max_results

    try:
        entries = _fetch_arxiv_entries(query, max_results)
    except Exception as e:
        print(f"[arxiv_loader] ERROR fetching from arXiv: {e}")
        print("This is almost always a network/DNS issue (same category as the "
              "earlier DNS problems) -- check connectivity to export.arxiv.org, "
              "or try again on a different network if this persists.")
        return

    print(f"[arxiv_loader] Fetched {len(entries)} entries for query: {query!r}")
    if not entries:
        print("[arxiv_loader] No entries returned -- try broadening arxiv_search_query.")
        return

    embedder = EmbeddingProvider()
    faiss_store = load_or_create_index()

    papers_ingested = 0
    with get_session() as session:
        paper_repo = PaperRepository(session)
        chunk_repo = ChunkRepository(session)

        for entry in entries:
            paper = paper_repo.get_or_create(
                source=PaperSource.arxiv,
                external_id=entry["arxiv_id"],
                title=entry["title"],
                abstract=entry["abstract"],
                year=entry["year"],
                url=entry["url"],
            )

            vector = embedder.embed(entry["abstract"])
            faiss_id = faiss_store.add_vector(vector)

            chunk_repo.insert_chunk(
                paper_id=paper.id,
                faiss_id=faiss_id,
                chunk_text=entry["abstract"],
                embedding_model_version=_EMBEDDING_MODEL_VERSION,
                section_type="abstract",
            )
            papers_ingested += 1

    save_index(faiss_store)
    print(f"[arxiv_loader] Ingested {papers_ingested} arXiv papers into KB1.")


if __name__ == "__main__":
    run_arxiv_ingestion()
