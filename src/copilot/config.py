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

    model_cache_dir: Path | None = None  # fastembed model cache (bake into images for offline)

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
    reasoning_effort: str = "high"
    llm_timeout_s: float = 300.0

    evidence_per_finding: int = 3  # reranked chunks retrieved per patient finding
    max_evidence: int = 15  # cap on distinct sources given to the reasoning step
    max_revisions: int = 2  # validation -> retrieval/reasoning loops before human review

    postgres_url: str = ""  # chat memory checkpointer; empty -> in-process memory

    pii_redaction: bool = True  # pseudonymise identifiers before every LLM call
    pii_spacy_model: str = "en_core_web_lg"  # Presidio NER model; regex-only if not installed

    feedback_path: Path = PROJECT_ROOT / "storage" / "feedback.jsonl"

    mcp_server_command: str = ""

    service_name: str = "clinician-copilot"
    log_level: str = "INFO"
    log_format: str = "json"  # json | text
    otel_exporter_otlp_endpoint: str = ""  # e.g. http://jaeger:4318 (OTLP/HTTP base URL)
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_public_key: str = ""  # with the secret key, traces go to Langfuse over OTLP
    langfuse_secret_key: str = ""  # e.g. "python -m copilot.mcp_server"; empty -> in-process


@lru_cache
def get_settings() -> Settings:
    return Settings()
