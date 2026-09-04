"""Unit tests for configuration loading."""

from src.rag_agent.config import get_settings, Settings


def test_get_settings_instance():
    """Verify settings singleton instance."""
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.api_port > 0
    assert settings.api_host is not None
    assert settings.fastapi_backend_url.startswith("http")


def test_default_models():
    """Verify default model configurations."""
    settings = get_settings()
    assert settings.default_gemini_model != ""
    assert settings.default_groq_model != ""


def test_redis_settings():
    """Verify Redis episodic memory default configuration."""
    settings = get_settings()
    assert settings.redis_url.startswith("redis://")
    assert settings.redis_chat_ttl_seconds == 3600
    assert settings.redis_max_connections > 0
    assert settings.redis_max_history_turns > 0
