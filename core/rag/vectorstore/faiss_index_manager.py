"""
Manages the FAISS index's lifecycle: load existing index from disk, or
create a fresh empty one if none exists yet.
"""

from pathlib import Path
from core.config.settings import settings
from core.rag.vectorstore.faiss_store import FAISSStore

_INDEX_FILENAME = "literature.index"


def get_index_path() -> Path:
    return Path(settings.faiss.index_path) / _INDEX_FILENAME


def load_or_create_index() -> FAISSStore:
    dim = settings.embeddings.dimension
    path = get_index_path()
    if path.exists():
        print(f"[faiss_index_manager] Loading existing index from {path}")
        return FAISSStore.load(str(path), dim)
    print(f"[faiss_index_manager] No existing index at {path} -- creating a new one.")
    return FAISSStore(dim)


def save_index(store: FAISSStore) -> None:
    path = get_index_path()
    store.save(str(path))
    print(f"[faiss_index_manager] Saved index ({store.ntotal} vectors) to {path}")
