"""
Optional: runs arXiv ingestion, adding topically-relevant literature into
KB1 alongside the mandatory PeerRead papers. Does nothing unless
INGESTION__ENABLE_ARXIV=true is set in your .env.

Run with: python -m scripts.run_arxiv_ingestion
"""

from core.rag.ingestion.arxiv_loader import run_arxiv_ingestion

run_arxiv_ingestion()
