"""Configuration and environment management for the Financial AI Agent."""

import os
from functools import lru_cache
from typing import Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env file explicitly
load_dotenv()


class Settings(BaseSettings):
    """Application settings and API credentials."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    # Server Settings
    api_host: str = os.getenv("API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    fastapi_backend_url: str = os.getenv("FASTAPI_BACKEND_URL", "http://127.0.0.1:8000")

    # API Keys
    google_api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
    groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY")
    phi_api_key: Optional[str] = os.getenv("PHI_API_KEY")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")

    # Model Defaults
    default_model_provider: str = os.getenv("DEFAULT_MODEL_PROVIDER", "gemini")
    default_gemini_model: str = os.getenv("DEFAULT_GEMINI_MODEL", "gemini-3.5-flash-lite")
    default_groq_model: str = os.getenv("DEFAULT_GROQ_MODEL", "llama-3.3-70b-versatile")
    default_openai_model: str = os.getenv("DEFAULT_OPENAI_MODEL", "gpt-4o-mini")

    # Redis Episodic Memory Settings
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_password: Optional[str] = os.getenv("REDIS_PASSWORD", None)
    redis_db: int = int(os.getenv("REDIS_DB", "0"))
    redis_chat_ttl_seconds: int = int(os.getenv("REDIS_CHAT_TTL_SECONDS", "3600"))
    redis_max_connections: int = int(os.getenv("REDIS_MAX_CONNECTIONS", "20"))
    redis_socket_timeout: float = float(os.getenv("REDIS_SOCKET_TIMEOUT", "2.0"))
    redis_socket_connect_timeout: float = float(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "2.0"))
    redis_max_history_turns: int = int(os.getenv("REDIS_MAX_HISTORY_TURNS", "10"))


@lru_cache()
def get_settings() -> Settings:
    """Retrieve cached application settings instance."""
    return Settings()
