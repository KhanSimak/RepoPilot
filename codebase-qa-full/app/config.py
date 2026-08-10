"""
config.py — application settings

The easiest file in the project. Just a typed settings object that reads
from environment variables (or a .env file). Using pydantic-settings means
every other file can do `from app.config import get_settings` and get
fully-typed, validated config instead of scattered os.getenv() calls.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    groq_api_key:      str 
    qdrant_api_key: str | None = None   # <-- ADD THIS
    qdrant_url:        str = "http://localhost:6333"
    qdrant_collection: str = "codebase"
    vector_size:       int = 384          # bge-small-en-v1.5 output dimension

    redis_url:         str = "redis://localhost:6379"

    repos_dir:         str = "/tmp/repos"
    embed_model:       str = "BAAI/bge-small-en-v1.5"
    rerank_model:      str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    min_confident_rerank_score: float = 0.55


    # llama-3.1-8b-instant: cheapest production model on Groq ($0.05/$0.08
    # per 1M tokens), 560 tok/s, and covered by Groq's no-credit-card free
    # tier (~6K TPM / 14,400 req/day) — fine for development and most
    # personal-project traffic without paying anything.
    groq_model:        str = "llama-3.1-8b-instant"

    query_top_k:       int = 20   # candidates fetched before reranking narrows to the final top_k

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    """
    Cached so we only construct Settings() once per process.
    Every router/engine module calls this instead of reading env vars directly.
    """
    return Settings()
