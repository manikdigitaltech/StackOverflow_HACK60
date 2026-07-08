"""
Clears the FAISS index and MySQL papers/chunks tables for a clean re-run of
ingestion. Use this whenever re-running ingestion after a code/settings
change, to avoid mixing old and new data (duplicate FAISS vectors, stale
low-quality papers, etc.).

Run with: python -m scripts.reset_ingestion
"""

from pathlib import Path
from sqlalchemy import delete
from core.db.session import get_session
from core.db.models import Chunk, Paper
from core.rag.vectorstore.faiss_index_manager import get_index_path

index_path = get_index_path()
if index_path.exists():
    index_path.unlink()
    print(f"Deleted FAISS index: {index_path}")
else:
    print(f"No FAISS index found at {index_path} (nothing to delete)")

with get_session() as session:
    chunks_deleted = session.execute(delete(Chunk)).rowcount
    papers_deleted = session.execute(delete(Paper)).rowcount

print(f"Deleted {chunks_deleted} chunk rows and {papers_deleted} paper rows from MySQL.")
print("Clean slate -- ready to re-run: python -m scripts.run_peerread_ingestion")
