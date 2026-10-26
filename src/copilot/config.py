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

    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    retrieval_candidates: int = 20  # hybrid hits passed to the reranker
    retrieval_top_k: int = 5  # reranked chunks given to the LLM
    min_relevance: float = 0.2  # best rerank score (0-1) below this -> refuse, no LLM call

    llm_model: str = "claude-opus-5-5"
    answer_effort: str = "medium"
    rewrite_effort: str = "low"
    history_turns: int = 6  # messages of chat history used to rewrite follow-ups

    postgres_url: str = ""  # chat memory checkpointer; empty -> in-process memory


@lru_cache
def get_settings() -> Settings:
    return Settings()
