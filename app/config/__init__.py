"""Configuration management module for the Multimodal RAG Chatbot."""

import os
from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # PostgreSQL Configuration
    postgres_host: str = Field(default="localhost", description="PostgreSQL host")
    postgres_port: int = Field(default=5432, description="PostgreSQL port")
    postgres_db: str = Field(default="rag_agent", description="PostgreSQL database name")
    postgres_user: str = Field(default="postgres", description="PostgreSQL username")
    postgres_password: str = Field(default="", description="PostgreSQL password")

    @property
    def postgres_dsn(self) -> str:
        """Generate PostgreSQL DSN connection string."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # Hugging Face Configuration
    hf_api_token: Optional[str] = Field(default=None, description="Hugging Face API token")
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Hugging Face embedding model name"
    )

    # LLM Configuration
    llm_api_key: Optional[str] = Field(default=None, description="LLM API key (OpenAI compatible)")
    llm_base_url: str = Field(default="https://api.openai.com/v1", description="LLM API base URL")
    llm_model_name: str = Field(default="gpt-3.5-turbo", description="LLM model name")
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="LLM temperature")
    llm_max_tokens: int = Field(default=2048, gt=0, description="LLM max tokens")

    # Document Processing Configuration
    chunk_size: int = Field(default=512, gt=0, description="Text chunk size")
    chunk_overlap: int = Field(default=50, ge=0, description="Text chunk overlap")
    document_path: str = Field(default="Data/sample.pdf", description="Path to the document")

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int, info) -> int:
        """Ensure chunk_overlap is less than chunk_size."""
        if "chunk_size" in info.data and v >= info.data["chunk_size"]:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v

    # Retrieval Configuration
    retrieval_top_k: int = Field(default=5, gt=0, description="Number of documents to retrieve")
    retrieval_similarity_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Similarity threshold for retrieval"
    )
    enable_reranking: bool = Field(default=False, description="Enable reranking")
    reranker_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Reranker model name"
    )

    # Application Configuration
    app_secret_key: str = Field(default="change-me-in-production", description="Application secret key")
    log_level: str = Field(default="INFO", description="Logging level")
    cache_enabled: bool = Field(default=True, description="Enable caching")
    cache_ttl_seconds: int = Field(default=3600, gt=0, description="Cache TTL in seconds")

    # Database Connection Pool
    db_pool_min_conn: int = Field(default=2, ge=1, description="Minimum database connections")
    db_pool_max_conn: int = Field(default=10, ge=1, description="Maximum database connections")
    db_pool_timeout: int = Field(default=30, gt=0, description="Database connection timeout")

    # Streamlit Configuration
    streamlit_server_port: int = Field(default=8501, description="Streamlit server port")
    streamlit_server_address: str = Field(default="0.0.0.0", description="Streamlit server address")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_v = v.upper()
        if upper_v not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return upper_v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()