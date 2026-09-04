"""Integration and API tests for FastAPI endpoints."""

from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient
from src.rag_agent.api import app

client = TestClient(app)


def test_root_endpoint():
    """Verify root endpoint returns service metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Financial AI Multi-Agent Backend"
    assert "version" in data
    assert data["docs"] == "/docs"


def test_health_endpoint():
    """Verify health endpoint reports healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["agent_ready"] is True
    assert "model_provider" in data
    assert "redis_connected" in data


def test_chat_endpoint_empty_message():
    """Verify validation error on empty message prompt."""
    response = client.post("/api/chat", json={"message": ""})
    assert response.status_code == 422 or response.status_code == 400


def test_chat_endpoint_invalid_payload():
    """Verify validation error on missing required field."""
    response = client.post("/api/chat", json={})
    assert response.status_code == 422


def test_chat_endpoint_successful_query():
    """Verify chat endpoint processes query with mock agent execution."""
    mock_response = (
        "### NVIDIA (NVDA) Financial Analysis\n\n"
        "| Metric | Value |\n|---|---|\n| Current Price | $125.00 |\n| Analyst Consensus | Strong Buy |\n\n"
        "**Recent News**: NVDA expands datacenter AI chip production."
    )
    with patch("src.rag_agent.api.run_financial_agent", return_value=(mock_response, "Gemini (gemini-3.5-flash-lite)")):
        response = client.post(
            "/api/chat",
            json={
                "message": "Summarize analyst recommendations for NVDA",
                "session_id": "test_session_123",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["session_id"] == "test_session_123"
        assert "NVIDIA" in data["response"]
        assert data["model_used"] == "Gemini (gemini-3.5-flash-lite)"


def test_chat_endpoint_multi_turn_with_memory():
    """Verify chat endpoint retrieves history from Redis and passes it to run_financial_agent."""
    mock_history = [
        {"role": "user", "content": "Tell me about Tesla.", "timestamp": 100.0},
        {"role": "assistant", "content": "Tesla makes EVs and energy systems.", "timestamp": 100.1},
    ]

    with patch("src.rag_agent.api.chat_memory.get_history", new_callable=AsyncMock, return_value=mock_history) as mock_get, \
         patch("src.rag_agent.api.chat_memory.add_turn", new_callable=AsyncMock, return_value=True) as mock_add, \
         patch("src.rag_agent.api.run_financial_agent", return_value=("Its PE ratio is 45.", "Gemini (gemini-3.5-flash-lite)")) as mock_run:

        response = client.post(
            "/api/chat",
            json={
                "message": "What is its PE ratio?",
                "session_id": "sess_multi_turn",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["response"] == "Its PE ratio is 45."

        # Verify history was retrieved for the session
        mock_get.assert_called_once_with("sess_multi_turn")

        # Verify run_financial_agent was called with history
        mock_run.assert_called_once_with(
            query="What is its PE ratio?",
            history=mock_history,
        )

        # Verify new turn was saved
        mock_add.assert_called_once_with(
            session_id="sess_multi_turn",
            user_message="What is its PE ratio?",
            assistant_response="Its PE ratio is 45.",
        )


def test_chat_endpoint_redis_failure_fallback_resilience():
    """Verify endpoint safely falls back to stateless turn without throwing 500 when Redis fails."""
    with patch("src.rag_agent.api.chat_memory.get_history", new_callable=AsyncMock, side_effect=RuntimeError("Redis timeout")), \
         patch("src.rag_agent.api.chat_memory.add_turn", new_callable=AsyncMock, side_effect=RuntimeError("Redis connection dropped")), \
         patch("src.rag_agent.api.run_financial_agent", return_value=("Stateless fallback response", "Gemini (gemini-3.5-flash-lite)")) as mock_run:

        response = client.post(
            "/api/chat",
            json={
                "message": "What is Microsoft stock price?",
                "session_id": "sess_broken_redis",
            },
        )

        # Must succeed with 200 OK despite Redis errors!
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["response"] == "Stateless fallback response"
        mock_run.assert_called_once_with(
            query="What is Microsoft stock price?",
            history=[],
        )


def test_get_session_history_endpoint():
    """Verify GET /api/chat/history/{session_id} returns cached turns."""
    mock_history = [
        {"role": "user", "content": "Hi", "timestamp": 123.0},
        {"role": "assistant", "content": "Hello", "timestamp": 124.0},
    ]
    with patch("src.rag_agent.api.chat_memory.get_history", new_callable=AsyncMock, return_value=mock_history):
        response = client.get("/api/chat/history/sess_history_test")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "sess_history_test"
        assert data["total_messages"] == 2
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "Hi"


def test_delete_session_history_endpoint():
    """Verify DELETE /api/chat/history/{session_id} purges memory."""
    with patch("src.rag_agent.api.chat_memory.clear_session", new_callable=AsyncMock, return_value=True):
        response = client.delete("/api/chat/history/sess_to_delete")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "sess_to_delete"
        assert data["cleared"] is True


def test_chat_endpoint_agent_failure():
    """Verify 500 error handling when agent execution raises an exception."""
    with patch("src.rag_agent.api.run_financial_agent", side_effect=RuntimeError("API quota exhausted")):
        response = client.post(
            "/api/chat",
            json={"message": "Give me Tesla stock info"},
        )
        assert response.status_code == 500
        data = response.json()
        assert data["status"] == "error"
        assert "API quota exhausted" in data["detail"]
