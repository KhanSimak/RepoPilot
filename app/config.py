"""
config.py — application settings
"""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --------------------------------------------------
    # Groq
    # --------------------------------------------------
    groq_api_key: str
    groq_model: str = "openai/gpt-oss-120b"

    # --------------------------------------------------
    # OpenRouter fallback
    # --------------------------------------------------
    openrouter_api_key: str | None = None
    openrouter_model: str = "openai/gpt-oss-120b"

    coding_model_api_key: str = ""      # your OpenRouter API key
    coding_model_base_url: str = "https://openrouter.ai/api/v1"
    coding_model: str = "cohere/north-mini-code:free"
    coding_model_fallback: str = "poolside/laguna-xs-2.1:free"

    cohere_api_key: str
    cohere_rerank_model: str = "rerank-v4.0-fast"
    cohere_low_confidence_threshold: float = 0.05

    # --------------------------------------------------
    # Qdrant
    # --------------------------------------------------
    qdrant_api_key: str | None = None
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "codebase"
    vector_size: int = 384

    # --------------------------------------------------
    # Redis
    # --------------------------------------------------
    redis_url: str = "redis://localhost:6379"

    # --------------------------------------------------
    # Repository / models
    # --------------------------------------------------
    repos_dir: str = "/tmp/repos"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    min_confident_rerank_score: float = 0.55

    # --------------------------------------------------
    # Retrieval
    # --------------------------------------------------
    query_top_k: int = 20

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    """
    Cached so we only construct Settings() once per process.
    """
    return Settings()