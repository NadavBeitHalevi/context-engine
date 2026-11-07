"""Application settings, loaded from environment variables / the .env file."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    app_name: str = "Context Engine"
    environment: str = "development"
    log_level: str = "INFO"

    # API keys
    openai_api_key: str = ""
    pinecone_api_key: str = ""

    # Models (defaults follow the book's Chapter 9 setup)
    generation_model: str = "gpt-5"
    embedding_model: str = "text-embedding-3-small"

    # Pinecone
    pinecone_index_name: str = "genai-mas-mcp-ch3"
    namespace_context: str = "ContextLibrary"
    namespace_knowledge: str = "KnowledgeStore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
