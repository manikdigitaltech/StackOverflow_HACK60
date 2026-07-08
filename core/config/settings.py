"""
Centralized configuration for the Autonomous AI Paper Reviewer.

All values are overridable via environment variables using the
`SECTION__FIELD` convention (e.g. DB__HOST, LLM__PROVIDER).
See .env.example for the full list.
"""

from typing import Literal, Optional
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class DBSettings(BaseModel):
    host: str = "localhost"
    port: int = 3306
    user: str = "reviewer_app"
    password: str = "changeme"
    database: str = "paper_reviewer"
    pool_size: int = 5

    @property
    def sqlalchemy_url(self) -> str:
        return (
            f"mysql+pymysql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


class CheckpointSettings(BaseModel):
    backend: Literal["sqlite", "postgres"] = "sqlite"
    sqlite_path: str = "./data/langgraph_checkpoints.db"
    postgres_url: Optional[str] = None


class LLMSettings(BaseModel):
    provider: Literal["llama3.1-8b", "qwen2.5-7b"] = "qwen2.5-7b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.2
    max_tokens: int = 4096
    json_mode: bool = True
    request_timeout_seconds: int = 120  # generous default for CPU inference


class VisionSettings(BaseModel):
    enabled: bool = False
    provider: Literal["qwen2-vl-7b", "llama3.2-vision-11b"] = "qwen2-vl-7b"
    base_url: str = "http://localhost:11434"
    max_figures_per_paper: int = 15


class EmbeddingSettings(BaseModel):
    provider: Literal["bge-large-en-v1.5", "e5-large"] = "bge-large-en-v1.5"
    dimension: int = 1024
    device: Literal["cpu", "cuda"] = "cpu"


class FAISSSettings(BaseModel):
    index_path: str = "./data/faiss_index"
    top_k: int = 5
    index_type: Literal["Flat", "IVF", "HNSW"] = "Flat"  # Flat is simplest/most robust on small KB1 sizes


class ReflectionSettings(BaseModel):
    max_revision_passes: int = 1  # bounded self-critique loop; 1 keeps demo timing sane


class IngestionSettings(BaseModel):
    peerread_data_path: str = "./data/peerread_raw"   # cloned allenai/PeerRead repo location
    peerread_venues: list[str] = ["iclr_2017", "acl_2017"]  # keep small for a fast local build
    max_peerread_papers: int = 30
    enable_arxiv: bool = False          # optional, off by default per current plan
    enable_semantic_scholar: bool = False  # optional, off by default per current plan


class AppSettings(BaseSettings):
    db: DBSettings = DBSettings()
    checkpoint: CheckpointSettings = CheckpointSettings()
    llm: LLMSettings = LLMSettings()
    vision: VisionSettings = VisionSettings()
    embeddings: EmbeddingSettings = EmbeddingSettings()
    faiss: FAISSSettings = FAISSSettings()
    reflection: ReflectionSettings = ReflectionSettings()
    ingestion: IngestionSettings = IngestionSettings()
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )


# Single shared instance — import this, don't re-instantiate AppSettings elsewhere.
settings = AppSettings()
