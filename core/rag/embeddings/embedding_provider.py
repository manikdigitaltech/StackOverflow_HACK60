"""Embedding model wrappers, one per index.

Index A and Index B use different embedding models on purpose (see README):
`bge-small-en-v1.5` is a general-purpose semantic encoder tuned for
short-passage retrieval, while `specter2_base` is trained specifically to
place scientifically-similar papers near each other by their title+abstract.
Sharing one embedder across both indexes would quietly degrade one of them.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class EmbeddingProvider(Protocol):
    """Common interface both concrete providers implement."""

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts, returning L2-normalized float32 vectors."""
        ...

    @property
    def dimension(self) -> int:
        ...


class BgeSmallEmbeddingProvider:
    """Wraps `BAAI/bge-small-en-v1.5` for Index A (paper chunks).

    Also serves as the documented fallback for Index B if SPECTER2's
    adapter-transformers dependency chain fails to load.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cpu"):
        self._model_name = model_name
        self._device = device
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._model_name, device=self._device)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Encode `texts` and L2-normalize so FAISS IndexFlatIP == cosine similarity.

        Args:
            texts: chunk texts to embed.

        Returns:
            float32 array of shape (len(texts), self.dimension), each row
            unit-normalized.
        """
        self._ensure_loaded()
        vectors = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return vectors.astype("float32")

    @property
    def dimension(self) -> int:
        self._ensure_loaded()
        return self._model.get_sentence_embedding_dimension()


class Specter2EmbeddingProvider:
    """Wraps `allenai/specter2_base` for Index B (literature corpus).

    Falls back to `BgeSmallEmbeddingProvider` if SPECTER2's adapter-transformers
    dependency chain can't be installed in time - see README "fallback" note.
    """

    def __init__(
        self,
        model_name: str = "allenai/specter2_base",
        adapter_name: str = "allenai/specter2",
        device: str = "cpu",
    ):
        self._model_name = model_name
        self._adapter_name = adapter_name
        self._device = device
        self._tokenizer = None
        self._model = None
        self._fallback: BgeSmallEmbeddingProvider | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._fallback is not None:
            return
        try:
            from adapters import AutoAdapterModel
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            model = AutoAdapterModel.from_pretrained(self._model_name)
            model.load_adapter(self._adapter_name, source="hf", load_as="specter2_proximity", set_active=True)
            model.to(self._device)
            model.eval()
            self._model = model
        except Exception as exc:  # the documented fallback (see README): degraded, never down
            import logging

            logging.getLogger(__name__).warning(
                "SPECTER2 load failed (%s); falling back to bge-small for Index B "
                "- literature-similarity quality is reduced but functional.", exc
            )
            self._fallback = BgeSmallEmbeddingProvider(device=self._device)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Encode `title [SEP] abstract` strings for literature-similarity search.

        Args:
            texts: pre-joined "title [SEP] abstract" strings - SPECTER2 was
                trained on exactly this input format, so callers must not
                pass raw title/abstract separately.

        Returns:
            float32 array of shape (len(texts), self.dimension), unit-normalized.
        """
        self._ensure_loaded()
        if self._fallback is not None:
            # bge has no [SEP] convention; plain space-join degrades gracefully
            return self._fallback.embed([t.replace("[SEP]", " ") for t in texts])

        import torch

        inputs = self._tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self._model(**inputs)
        # SPECTER2's document embedding is the [CLS] token's final hidden state.
        vectors = output.last_hidden_state[:, 0, :].cpu().numpy()
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vectors / norms).astype("float32")

    @property
    def dimension(self) -> int:
        self._ensure_loaded()
        if self._fallback is not None:
            return self._fallback.dimension
        return self._model.config.hidden_size
