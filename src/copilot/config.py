from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    anthropic_api_key: str = ""

    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_path: Path = PROJECT_ROOT / "storage" / "qdrant"
    qdrant_collection: str = "clinical_guidelines"

    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"

    chunk_max_tokens: int = 600
    chunk_overlap_tokens: int = 100


@lru_cache
def get_settings() -> Settings:
    return Settings()
