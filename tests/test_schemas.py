"""Unit tests for Pydantic schemas and request/response models."""

import pytest
from pydantic import ValidationError
from src.rag_agent.schemas import (
    ChatMessage,
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ErrorResponse,
    SessionClearResponse,
)


def test_chat_request_valid():
    """Verify valid ChatRequest creation."""
    req = ChatRequest(message="What is Apple's stock price?", session_id="sess_001")
    assert req.message == "What is Apple's stock price?"
    assert req.session_id == "sess_001"


def test_chat_request_default_session():
    """Verify default session_id is None."""
    req = ChatRequest(message="NVDA earnings summary")
    assert req.message == "NVDA earnings summary"
    assert req.session_id is None


def test_chat_request_empty_validation():
    """Verify validation error when message is empty."""
    with pytest.raises(ValidationError):
        ChatRequest(message="")


def test_chat_response_valid():
    """Verify valid ChatResponse creation."""
    res = ChatResponse(
        response="# NVDA Analysis\nStrong buy consensus.",
        session_id="sess_123",
        status="success",
        model_used="Gemini (gemini-3.5-flash-lite)",
    )
    assert res.response.startswith("# NVDA")
    assert res.session_id == "sess_123"
    assert res.status == "success"
    assert res.model_used == "Gemini (gemini-3.5-flash-lite)"


def test_chat_message_schema():
    """Verify ChatMessage schema parsing."""
    msg = ChatMessage(role="user", content="Hello", timestamp=1234567890.0)
    assert msg.role == "user"
    assert msg.content == "Hello"
    assert msg.timestamp == 1234567890.0


def test_chat_history_response():
    """Verify ChatHistoryResponse schema."""
    history = ChatHistoryResponse(
        session_id="sess_abc",
        messages=[
            ChatMessage(role="user", content="Query"),
            ChatMessage(role="assistant", content="Answer"),
        ],
        total_messages=2,
    )
    assert history.session_id == "sess_abc"
    assert len(history.messages) == 2
    assert history.total_messages == 2


def test_session_clear_response():
    """Verify SessionClearResponse schema."""
    res = SessionClearResponse(
        session_id="sess_xyz",
        cleared=True,
        message="Purged successfully",
    )
    assert res.cleared is True
    assert res.session_id == "sess_xyz"


def test_health_response_defaults():
    """Verify HealthResponse default values and optional redis fields."""
    health = HealthResponse(
        status="healthy",
        version="0.1.0",
        agent_ready=True,
        model_provider="Gemini",
        model_id="gemini-3.5-flash-lite",
        redis_connected=True,
        redis_status="connected",
    )
    assert health.status == "healthy"
    assert health.agent_ready is True
    assert health.version == "0.1.0"
    assert health.redis_connected is True
    assert health.redis_status == "connected"


def test_error_response():
    """Verify ErrorResponse model."""
    err = ErrorResponse(detail="API key invalid")
    assert err.detail == "API key invalid"
    assert err.status == "error"
