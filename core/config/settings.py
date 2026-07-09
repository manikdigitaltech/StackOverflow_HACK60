"""
Centralized configuration for the Autonomous AI Paper Reviewer.

All values are overridable via environment variables using the
`SECTION__FIELD` convention (e.g. DB__HOST, LLM__PROVIDER).
See .env.example for the full list.
"""

from typing import Literal, Optional
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


from urllib.parse import quote_plus


class DBSettings(BaseModel):
    host: str = "localhost"
    port: int = 3307
    user: str = "reviewer_app"
    password: str = "changeme"
    database: str = "paper_reviewer"
    pool_size: int = 5

    @property
    def sqlalchemy_url(self) -> str:
        safe_user = quote_plus(self.user)
        safe_password = quote_plus(self.password)
        return (
            f"mysql+pymysql://{safe_user}:{safe_password}"
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
    context_window: int = 8192          # passed as Ollama's num_ctx — default Ollama context (2048-4096) is too small


class ParsingSettings(BaseModel):
    max_pages_hard_cap: int = 60        # safety valve for accidental huge uploads; generous for any normal paper/thesis chapter
    prompt_token_budget: int = 6000     # total budget for paper content injected into a single agent prompt


class VisionSettings(BaseModel):
    enabled: bool = False
    provider: Literal["qwen2-vl-7b", "llama3.2-vision-11b"] = "qwen2-vl-7b"
    base_url: str = "http://localhost:11434"
    max_figures_per_paper: int = 15
    temperature: float = 0.1        # low: we want a factual description, not creative variation
    max_tokens: int = 512           # a figure caption/description doesn't need essay length
    crop_dpi: int = 300             # passed to figure_cropper.crop_figure


class FormulaSettings(BaseModel):
    # Region detection (bbox/page) + cropping to PNG always run when Docling
    # finds a "formula" layout region -- no extra model needed for that.
    # `enabled` gates ONLY Docling's do_formula_enrichment (LaTeX/plaintext
    # recognition via its CodeFormulaV2 model), which downloads that model on
    # first use -- same off-by-default, real-code-path philosophy as
    # VisionSettings.enabled for figure description.
    enabled: bool = False
    crop_dpi: int = 300               # passed to figure_cropper.crop_figure
    max_formulas_per_paper: int = 30  # a dense math paper can have 100+ detected regions


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


class LiveSourcesSettings(BaseModel):
    # Semantic Scholar's public search API works unauthenticated but shares a
    # heavily rate-limited pool (~100 req/5min across all unauthenticated
    # traffic); a free API key raises that to a per-key limit. Optional --
    # the client already degrades to [] on any non-2xx response either way.
    semantic_scholar_api_key: Optional[str] = None


class IngestionSettings(BaseModel):
    peerread_data_path: str = "./data/peerread_raw"   # cloned allenai/PeerRead repo location
    peerread_venues: list[str] = ["iclr_2017", "acl_2017"]  # keep small for a fast local build
    max_peerread_papers: int = 30
    enable_arxiv: bool = False          # optional, off by default per current plan
    enable_semantic_scholar: bool = False  # optional, off by default per current plan
    arxiv_search_query: str = "cat:cs.CV"  # override in .env to target your paper's actual subfield
    arxiv_max_results: int = 20


class AppSettings(BaseSettings):
    db: DBSettings = DBSettings()
    checkpoint: CheckpointSettings = CheckpointSettings()
    llm: LLMSettings = LLMSettings()
    vision: VisionSettings = VisionSettings()
    formula: FormulaSettings = FormulaSettings()
    embeddings: EmbeddingSettings = EmbeddingSettings()
    faiss: FAISSSettings = FAISSSettings()
    reflection: ReflectionSettings = ReflectionSettings()
    ingestion: IngestionSettings = IngestionSettings()
    live_sources: LiveSourcesSettings = LiveSourcesSettings()
    parsing: ParsingSettings = ParsingSettings()
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )


# Single shared instance — import this, don't re-instantiate AppSettings elsewhere.
settings = AppSettings()
