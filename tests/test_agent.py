"""Unit tests for Multi-Agent construction and configuration."""

from unittest.mock import MagicMock, patch
import pytest
from src.rag_agent.agent import (
    create_model_instance,
    get_web_search_agent,
    get_finance_agent,
    get_financial_team_agent,
    run_financial_agent,
)
from src.rag_agent.config import get_settings


def test_create_model_instance():
    """Test that model instance is created without raising an exception."""
    model, model_info = create_model_instance()
    assert model is not None
    assert isinstance(model_info, str)
    assert len(model_info) > 0


def test_web_search_agent_creation():
    """Test that web search agent is properly initialized with tools."""
    agent = get_web_search_agent()
    assert agent.name == "web_search_agent"
    assert agent.tools is not None
    assert len(agent.tools) > 0


def test_finance_agent_creation():
    """Test that finance agent is properly initialized with YFinance tools."""
    agent = get_finance_agent()
    assert agent.name == "Finance agent"
    assert agent.tools is not None
    assert len(agent.tools) > 0


def test_financial_team_agent_creation():
    """Test that financial team coordinator agent is initialized with child agents."""
    team_agent, model_info = get_financial_team_agent()
    assert team_agent.name == "Financial Team Coordinator"
    assert team_agent.team is not None
    assert len(team_agent.team) == 2
    assert "Gemini" in model_info or "Groq" in model_info or "OpenAI" in model_info


def test_run_financial_agent_with_history_augmentation():
    """Verify run_financial_agent incorporates history into prompt query."""
    history = [
        {"role": "user", "content": "What is NVDA?"},
        {"role": "assistant", "content": "A chip company."},
    ]

    mock_agent = MagicMock()
    mock_run_result = MagicMock()
    mock_run_result.content = "NVDA PE ratio is 50."
    mock_agent.run.return_value = mock_run_result

    with patch("src.rag_agent.agent.get_fast_financial_agent", return_value=(mock_agent, "Gemini (gemini-3.5-flash-lite)")), \
         patch("src.rag_agent.agent.execute_with_rate_limit", side_effect=lambda agent_name, target_func, **kwargs: target_func()):

        response, model_info = run_financial_agent("What is its PE ratio?", history=history)

        assert response == "NVDA PE ratio is 50."
        assert model_info == "Gemini (gemini-3.5-flash-lite)"
        mock_agent.run.assert_called_once()
        called_prompt = mock_agent.run.call_args[0][0]
        assert "### Previous Conversation History:" in called_prompt
        assert "**User**: What is NVDA?" in called_prompt
        assert "### Current User Query:" in called_prompt
        assert "What is its PE ratio?" in called_prompt
