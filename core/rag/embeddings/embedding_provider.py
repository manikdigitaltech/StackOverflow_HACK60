"""
Wraps sentence-transformers for local embedding generation.

bge-large-en-v1.5 produces 1024-dim vectors; e5-large (the configured
alternative) produces the same dimensionality via a different model.

Note on query vs. document embedding: BGE models are trained to expect an
instruction prefix on QUERIES at retrieval time, but NOT on the documents/
passages being indexed. Skipping this halves the model's intended retrieval
quality, so this module exposes embed_query() (prefixed) separately from
embed()/embed_batch() (unprefixed, used at ingestion time for KB1 chunks).
"""

from typing import List
from sentence_transformers import SentenceTransformer

from core.config.settings import settings

_MODEL_MAP = {
    "bge-large-en-v1.5": "BAAI/bge-large-en-v1.5",
    "e5-large": "intfloat/e5-large-v2",
}

_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class EmbeddingProvider:
    def __init__(self):
        model_name = _MODEL_MAP.get(settings.embeddings.provider, settings.embeddings.provider)
        self._model = SentenceTransformer(model_name, device=settings.embeddings.device)
        self._is_bge = "bge" in settings.embeddings.provider

    def embed(self, text: str) -> List[float]:
        """Use for documents/passages being ingested into FAISS (no instruction prefix)."""
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        vectors = self._model.encode(
            texts, normalize_embeddings=True, batch_size=16, show_progress_bar=False
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Use for retrieval queries (adds the BGE instruction prefix, when applicable)."""
        prefixed = _BGE_QUERY_INSTRUCTION + text if self._is_bge else text
        return self.embed(prefixed)
