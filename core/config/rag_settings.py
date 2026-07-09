"""Plain-Python configuration for the RAG subsystem - no YAML, no env-file magic.

Import `RAG_SETTINGS` and read attributes directly. If a value needs to vary
per environment, override it via constructor args when instantiating indexes
in the future (e.g. tests pass a temp directory), not by adding config files.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class ChunkingSettings:
    target_tokens: int = 400
    min_tokens: int = 300
    max_tokens: int = 500
    overlap_tokens: int = 50


@dataclass(frozen=True)
class PaperIndexSettings:
    """Index A - ephemeral, rebuilt per review run."""

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    rrf_k: int = 60
    default_top_k: int = 5


@dataclass(frozen=True)
class LiteratureIndexSettings:
    """Index B - persistent, built once offline."""

    embedding_model: str = "allenai/specter2_base"
    fallback_embedding_model: str = "BAAI/bge-small-en-v1.5"
    index_path: Path = DATA_DIR / "literature_index" / "index.faiss"
    records_path: Path = DATA_DIR / "literature_index" / "records.jsonl"
    default_top_k: int = 10


@dataclass(frozen=True)
class LiveSourceSettings:
    request_timeout_seconds: float = 5.0
    semantic_scholar_base_url: str = "https://api.semanticscholar.org/graph/v1"
    default_top_k: int = 5
    # Gate *live query-time* use during a review run. Distinct from
    # settings.IngestionSettings.enable_arxiv/enable_semantic_scholar, which
    # gate *corpus-build-time* ingestion. Safe to leave on: both clients
    # degrade to [] on any failure, so offline runs lose nothing but
    # supplementary matches -- the review itself never blocks on the network.
    enable_arxiv: bool = True
    enable_semantic_scholar: bool = True


@dataclass(frozen=True)
class RagSettings:
    chunking: ChunkingSettings = field(default_factory=ChunkingSettings)
    paper_index: PaperIndexSettings = field(default_factory=PaperIndexSettings)
    literature_index: LiteratureIndexSettings = field(default_factory=LiteratureIndexSettings)
    live_sources: LiveSourceSettings = field(default_factory=LiveSourceSettings)


RAG_SETTINGS = RagSettings()
