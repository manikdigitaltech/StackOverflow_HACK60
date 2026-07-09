"""Shared test fixtures: a deterministic fake embedding provider so index
tests exercise real FAISS/BM25/RRF logic without downloading any model."""
from __future__ import annotations

import json
import pathlib
import zlib

import numpy as np
import pytest

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


class FakeEmbeddingProvider:
    """Hashed bag-of-words embeddings: texts sharing more tokens get higher
    cosine similarity. Deterministic across runs (crc32, not hash())."""

    def __init__(self, dimension: int = 64):
        self._dimension = dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.lower().split():
                out[row, zlib.crc32(token.encode()) % self._dimension] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

    @property
    def dimension(self) -> int:
        return self._dimension


@pytest.fixture
def fake_provider() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


@pytest.fixture
def sample_paper() -> dict:
    return json.loads((FIXTURES / "sample_paper.json").read_text(encoding="utf-8"))
