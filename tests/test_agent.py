"""Unit tests for Multi-Agent construction and configuration."""

import pytest
from src.rag_agent.agent import (
    create_model_instance,
    get_web_search_agent,
    get_finance_agent,
    get_financial_team_agent,
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
