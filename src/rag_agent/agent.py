"""High-Performance Multi-Agent Financial System with Global Rate Coordination."""

import logging
from functools import lru_cache
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
from phi.agent import Agent
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools

from src.rag_agent.config import get_settings
from src.rag_agent.rate_limiter import rate_coordinator, execute_with_rate_limit
from src.rag_agent.memory import RedisChatMemory

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def create_model_instance(
    provider: Optional[str] = None,
    model_id: Optional[str] = None,
):
    """Instantiate and cache the LLM model instance for minimal latency."""
    settings = get_settings()
    chosen_provider = (provider or settings.default_model_provider).lower()

    # 1. Try Gemini (Fastest default: gemini-2.5-flash / gemini-1.5-flash)
    if chosen_provider == "gemini" or (not provider and settings.google_api_key):
        if settings.google_api_key:
            try:
                from phi.model.google import Gemini

                m_id = model_id or settings.default_gemini_model or "gemini-2.5-flash"
                logger.info("Initializing Gemini model: %s", m_id)
                return Gemini(id=m_id, api_key=settings.google_api_key), f"Gemini ({m_id})"
            except Exception as e:
                logger.warning("Failed to initialize Gemini model: %s. Trying fallback...", e)

    # 2. Try Groq
    if chosen_provider == "groq" or settings.groq_api_key:
        if settings.groq_api_key:
            try:
                from phi.model.groq import Groq

                m_id = model_id or settings.default_groq_model or "llama-3.1-70b-versatile"
                logger.info("Initializing Groq model: %s", m_id)
                return Groq(id=m_id, api_key=settings.groq_api_key), f"Groq ({m_id})"
            except Exception as e:
                logger.warning("Failed to initialize Groq model: %s. Trying fallback...", e)

    # 3. Try OpenAI
    if settings.openai_api_key:
        try:
            from phi.model.openai import OpenAIChat

            m_id = model_id or settings.default_openai_model or "gpt-4o-mini"
            logger.info("Initializing OpenAI model: %s", m_id)
            return OpenAIChat(id=m_id, api_key=settings.openai_api_key), f"OpenAI ({m_id})"
        except Exception as e:
            logger.warning("Failed to initialize OpenAI model: %s", e)

    # Fallback default Gemini instance
    from phi.model.google import Gemini

    m_id = model_id or settings.default_gemini_model or "gemini-2.5-flash"
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
        show_tool_calls=False,
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
        show_tool_calls=False,
        markdown=True,
    )


@lru_cache(maxsize=2)
def get_fast_financial_agent(
    provider: Optional[str] = None,
    model_id: Optional[str] = None,
) -> Tuple[Agent, str]:
    """Optimized direct-tool agent executing queries with minimum latency in a single pass."""
    model, model_info = create_model_instance(provider=provider, model_id=model_id)

    agent = Agent(
        name="Fast Financial Coordinator",
        model=model,
        tools=[
            DuckDuckGo(),
            YFinanceTools(
                stock_price=True,
                analyst_recommendations=True,
                stock_fundamentals=True,
                company_news=True,
            ),
        ],
        instructions=[
            "Always include sources and citations when web search is used.",
            "Use markdown tables to clearly present financial metrics, prices, and numerical data.",
            "Only call the specific tools required to accurately answer the prompt.",
            "Synthesize both market metrics and news concisely and rapidly.",
        ],
        show_tool_calls=False,
        markdown=True,
    )

    return agent, model_info


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
        show_tool_calls=False,
        markdown=True,
    )

    return team_agent, model_info


def run_financial_agent(
    query: str,
    provider: Optional[str] = None,
    model_id: Optional[str] = None,
    history: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, str]:
    """Execute a query under global rate limits and token budgeting.

    Args:
        query: The current user query or prompt.
        provider: Optional LLM provider override.
        model_id: Optional model ID override.
        history: Optional previous conversation messages from episodic memory.

    Returns:
        Tuple[str, str]: (response_content, model_info)
    """
    agent, model_info = get_fast_financial_agent(provider=provider, model_id=model_id)
    effective_query = RedisChatMemory.format_history_context(history, query) if history else query

    def _execute():
        result = agent.run(effective_query)
        if hasattr(result, "content") and result.content:
            return str(result.content)
        elif isinstance(result, str):
            return result
        else:
            return str(result)

    response_text = execute_with_rate_limit(
        agent_name="FinancialAgent",
        target_func=_execute,
        estimated_tokens=max(len(effective_query) // 4, 500),
    )
    return response_text, model_info


def run_financial_agent_stream(
    query: str,
    provider: Optional[str] = None,
    model_id: Optional[str] = None,
    history: Optional[List[Dict[str, Any]]] = None,
) -> Generator[str, None, None]:
    """Stream response tokens with rate coordination permission, token budgeting, and memory context."""
    effective_query = RedisChatMemory.format_history_context(history, query) if history else query

    rate_coordinator.acquire_permission_sync(
        agent_name="FinancialStreamAgent",
        estimated_tokens=max(len(effective_query) // 4, 500),
    )
    agent, _ = get_fast_financial_agent(provider=provider, model_id=model_id)
    stream_response = agent.run(effective_query, stream=True)

    accumulated_chars = 0
    for chunk in stream_response:
        if hasattr(chunk, "content") and chunk.content:
            text = str(chunk.content)
            accumulated_chars += len(text)
            yield text
        elif isinstance(chunk, str) and chunk:
            accumulated_chars += len(chunk)
            yield chunk

    # Record token usage into rolling 60-second window
    rate_coordinator.record_token_usage(accumulated_chars // 4 + 500)
