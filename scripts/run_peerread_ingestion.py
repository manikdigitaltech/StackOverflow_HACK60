"""
Step 5 test: runs PeerRead ingestion once, populating both FAISS and MySQL.

Run with: python -m scripts.run_peerread_ingestion

Safe to re-run: get_or_create() means re-running won't duplicate `papers`
rows, though it WILL add duplicate `chunks`/FAISS vectors if re-run after
a successful prior run (no dedup logic on chunks yet -- fine for a single
initial-build run, worth knowing if you re-run after tweaking settings).
"""

from core.rag.ingestion.peerread_loader import run_peerread_ingestion

run_peerread_ingestion()
