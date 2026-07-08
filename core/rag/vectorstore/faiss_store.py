"""
Thin wrapper around FAISS. Uses IndexFlatIP (exact inner-product search) --
since EmbeddingProvider normalizes all vectors to unit length, inner product
is equivalent to cosine similarity, and "Flat" is the simplest, most exact
index type (appropriate at PeerRead-subset scale; revisit if KB1 ever grows
into the hundreds of thousands of chunks).

Design choice worth understanding: FAISS's own sequential internal ID
(0, 1, 2, ... in insertion order) IS the faiss_id stored in MySQL's `chunks`
table -- no separate ID-generation scheme needed. This works because we
never delete from the index directly (stale chunks are marked in MySQL,
per the blueprint's design; physical FAISS cleanup happens via periodic
full reindex, not live deletion).
"""

from pathlib import Path
from typing import List, Tuple
import numpy as np
import faiss


class FAISSStore:
    def __init__(self, dim: int):
        self._dim = dim
        self._index = faiss.IndexFlatIP(dim)

    def add_vector(self, vector: List[float]) -> int:
        return self.add_vectors([vector])[0]

    def add_vectors(self, vectors: List[List[float]]) -> List[int]:
        start_id = self._index.ntotal
        arr = np.array(vectors, dtype="float32")
        self._index.add(arr)
        return list(range(start_id, start_id + len(vectors)))

    def similarity_search(self, query_vector: List[float], k: int = 5) -> List[Tuple[int, float]]:
        """Returns [(faiss_id, similarity_score), ...], best match first."""
        arr = np.array([query_vector], dtype="float32")
        scores, ids = self._index.search(arr, k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    @property
    def ntotal(self) -> int:
        return self._index.ntotal

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, path)

    @classmethod
    def load(cls, path: str, dim: int) -> "FAISSStore":
        instance = cls(dim)
        instance._index = faiss.read_index(path)
        return instance
