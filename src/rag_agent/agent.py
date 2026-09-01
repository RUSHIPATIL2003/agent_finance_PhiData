"""Multi-Agent Financial System using PhiData, DuckDuckGo, and YFinance."""

import logging
from typing import Optional, Tuple
from phi.agent import Agent
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools

from src.rag_agent.config import get_settings

logger = logging.getLogger(__name__)


def create_model_instance(
    provider: Optional[str] = None,
    model_id: Optional[str] = None,
):
    """Instantiate the appropriate LLM model with fallbacks based on configured API keys."""
    settings = get_settings()
    chosen_provider = (provider or settings.default_model_provider).lower()

    # 1. Try Gemini
    if chosen_provider == "gemini" or (not provider and settings.google_api_key):
        if settings.google_api_key:
            try:
                from phi.model.google import Gemini

                m_id = model_id or settings.default_gemini_model
                logger.info("Initializing Gemini model: %s", m_id)
                return Gemini(id=m_id, api_key=settings.google_api_key), f"Gemini ({m_id})"
            except Exception as e:
                logger.warning("Failed to initialize Gemini model: %s. Trying fallback...", e)

    # 2. Try Groq
    if chosen_provider == "groq" or settings.groq_api_key:
        if settings.groq_api_key:
            try:
                from phi.model.groq import Groq

                m_id = model_id or settings.default_groq_model
                logger.info("Initializing Groq model: %s", m_id)
                return Groq(id=m_id, api_key=settings.groq_api_key), f"Groq ({m_id})"
            except Exception as e:
                logger.warning("Failed to initialize Groq model: %s. Trying fallback...", e)

    # 3. Try OpenAI
    if settings.openai_api_key:
        try:
            from phi.model.openai import OpenAIChat

            m_id = model_id or settings.default_openai_model
            logger.info("Initializing OpenAI model: %s", m_id)
            return OpenAIChat(id=m_id, api_key=settings.openai_api_key), f"OpenAI ({m_id})"
        except Exception as e:
            logger.warning("Failed to initialize OpenAI model: %s", e)

    # Fallback default Gemini instance
    from phi.model.google import Gemini

    m_id = model_id or settings.default_gemini_model
    return Gemini(id=m_id), f"Gemini ({m_id})"


def get_web_search_agent(model=None) -> Agent:
    """Create and return the Web Search Agent."""
    if model is None:
        model, _ = create_model_instance()

    return Agent(
        name="web_search_agent",
        role="Search the web for information",
        model=model,
        tools=[DuckDuckGo()],
        instructions=["Always use sources", "Provide relevant citations and links where available"],
        show_tool_calls=True,
        markdown=True,
    )


def get_finance_agent(model=None) -> Agent:
    """Create and return the Financial Analysis Agent."""
    if model is None:
        model, _ = create_model_instance()

    return Agent(
        name="Finance agent",
        role="Get financial data",
        model=model,
        tools=[
            YFinanceTools(
                stock_price=True,
                analyst_recommendations=True,
                stock_fundamentals=True,
                company_news=True,
            )
        ],
        instructions=[
            "Use tables to display data",
            "Present key financial ratios, stock prices, analyst consensus, and company news clearly",
        ],
        show_tool_calls=True,
        markdown=True,
    )


def get_financial_team_agent(
    provider: Optional[str] = None,
    model_id: Optional[str] = None,
) -> Tuple[Agent, str]:
    """Create and return the Multi-Agent Financial Coordinator."""
    model, model_info = create_model_instance(provider=provider, model_id=model_id)

    web_agent = get_web_search_agent(model=model)
    finance_agent = get_finance_agent(model=model)

    team_agent = Agent(
        name="Financial Team Coordinator",
        team=[web_agent, finance_agent],
        model=model,
        instructions=[
            "Always include sources and citations",
            "Use markdown tables to display the financial and numerical data",
            "Synthesize both market metrics and latest news into actionable, structured analysis",
        ],
        show_tool_calls=True,
        markdown=True,
    )

    return team_agent, model_info


def run_financial_agent(
    query: str,
    provider: Optional[str] = None,
    model_id: Optional[str] = None,
) -> Tuple[str, str]:
    """Execute a query against the multi-agent financial coordinator.

    Returns:
        Tuple[str, str]: (response_content, model_info)
    """
    team_agent, model_info = get_financial_team_agent(provider=provider, model_id=model_id)
    result = team_agent.run(query)

    if hasattr(result, "content") and result.content:
        return str(result.content), model_info
    elif isinstance(result, str):
        return result, model_info
    else:
        return str(result), model_info
