"""Integration and API tests for FastAPI endpoints."""

from unittest.mock import patch
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
    with patch("src.rag_agent.api.run_financial_agent", return_value=(mock_response, "Gemini (gemini-2.5-flash)")):
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
        assert data["model_used"] == "Gemini (gemini-2.5-flash)"


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
