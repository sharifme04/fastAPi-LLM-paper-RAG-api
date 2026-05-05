"""Application configuration via pydantic-settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    app_name: str = "paper-rag-api"
    app_version: str = "1.0.0"
    debug: bool = False

    # PostgreSQL (with pgvector extension)
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/paper_rag"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Anthropic
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"
    anthropic_input_price_per_mtok: float = 3.00
    anthropic_output_price_per_mtok: float = 15.00

    # Embedding + reranker
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedding_dim: int = 384

    # Retrieval
    top_k_vector: int = 10
    top_k_rerank: int = 5
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 100

    # Cost / rate
    daily_cost_limit: float = 10.00
    rate_limit: str = "20/minute"
    max_pdf_bytes: int = 50 * 1024 * 1024  # 50 MB

    # Upload dir (inside container or local)
    upload_dir: str = "uploads"

    # Eval
    eval_faithfulness_threshold: float = 0.7


@lru_cache
def get_settings() -> Settings:
    return Settings()
