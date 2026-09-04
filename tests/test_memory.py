"""Unit and resilience tests for Redis episodic chat memory caching layer."""

import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from redis.exceptions import ConnectionError, TimeoutError, RedisError

from src.rag_agent.config import Settings
from src.rag_agent.memory import RedisChatMemory, chat_memory, get_chat_memory


def test_format_history_context_empty():
    """Verify format_history_context returns original query when history is empty."""
    query = "What is the price of Apple?"
    result = RedisChatMemory.format_history_context([], query)
    assert result == query


def test_format_history_context_with_turns():
    """Verify format_history_context structures previous turns into prompt context."""
    history = [
        {"role": "user", "content": "What is NVDA?"},
        {"role": "assistant", "content": "Nvidia is a semiconductor company."},
    ]
    query = "What are their latest earnings?"
    result = RedisChatMemory.format_history_context(history, query)

    assert "### Previous Conversation History:" in result
    assert "**User**: What is NVDA?" in result
    assert "**Assistant**: Nvidia is a semiconductor company." in result
    assert "### Current User Query:" in result
    assert "What are their latest earnings?" in result


@pytest.mark.anyio
async def test_memory_dependency():
    """Verify FastAPI dependency returns memory instance."""
    mem = await get_chat_memory()
    assert isinstance(mem, RedisChatMemory)


@pytest.mark.anyio
async def test_memory_initialization_connection_failure():
    """Verify initialize handles connection failure gracefully without raising."""
    custom_settings = Settings(
        redis_url="redis://nonexistent-host:9999/0",
        redis_socket_connect_timeout=0.1,
        redis_socket_timeout=0.1,
    )
    memory = RedisChatMemory(settings=custom_settings)
    connected = await memory.initialize()
    assert connected is False
    assert await memory.ping() is False


@pytest.mark.anyio
async def test_get_history_empty_session():
    """Verify get_history returns empty list for None or blank session_id."""
    memory = RedisChatMemory()
    assert await memory.get_history(None) == []
    assert await memory.get_history("") == []
    assert await memory.get_history("   ") == []


@pytest.mark.anyio
async def test_get_history_with_mock_redis():
    """Verify get_history correctly retrieves and parses JSON entries from Redis list."""
    memory = RedisChatMemory()
    mock_client = AsyncMock()
    mock_client.ping = AsyncMock(return_value=True)

    fake_entries = [
        json.dumps({"role": "user", "content": "Hello", "timestamp": 100.0}),
        json.dumps({"role": "assistant", "content": "Hi there!", "timestamp": 100.1}),
    ]
    mock_client.lrange = AsyncMock(return_value=fake_entries)

    memory._client = mock_client
    memory._is_initialized = True

    history = await memory.get_history("sess_test_123")
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "Hi there!"


@pytest.mark.anyio
async def test_get_history_handles_corrupt_json():
    """Verify get_history skips malformed JSON strings without crashing."""
    memory = RedisChatMemory()
    mock_client = AsyncMock()
    mock_client.lrange = AsyncMock(return_value=["not-valid-json", json.dumps({"role": "user", "content": "Valid"})])

    memory._client = mock_client
    memory._is_initialized = True

    history = await memory.get_history("sess_corrupt")
    assert len(history) == 1
    assert history[0]["content"] == "Valid"


@pytest.mark.anyio
async def test_get_history_resilience_on_redis_exception():
    """Verify get_history returns empty list when Redis raises an exception."""
    memory = RedisChatMemory()
    mock_client = AsyncMock()
    mock_client.lrange = AsyncMock(side_effect=ConnectionError("Redis connection lost"))

    memory._client = mock_client
    memory._is_initialized = True

    history = await memory.get_history("sess_err")
    assert history == []


@pytest.mark.anyio
async def test_add_turn_with_mock_pipeline_and_ttl():
    """Verify add_turn uses pipeline with rpush, ltrim, and expire with correct TTL."""
    memory = RedisChatMemory()
    mock_client = AsyncMock()

    mock_pipe = MagicMock()
    mock_pipe.rpush = MagicMock()
    mock_pipe.ltrim = MagicMock()
    mock_pipe.expire = MagicMock()
    mock_pipe.execute = AsyncMock(return_value=[1, 2, True, True])

    mock_client.pipeline = MagicMock(return_value=mock_pipe)

    memory._client = mock_client
    memory._is_initialized = True

    success = await memory.add_turn(
        session_id="sess_turn_test",
        user_message="Analyze Tesla",
        assistant_response="Tesla is an EV maker.",
        ttl=1800,
    )

    assert success is True
    assert mock_pipe.rpush.call_count == 1
    assert mock_pipe.ltrim.call_count == 1
    mock_pipe.expire.assert_called_with("chat:session:sess_turn_test:messages", 1800)


@pytest.mark.anyio
async def test_add_turn_resilience_on_pipeline_failure():
    """Verify add_turn returns False safely if pipeline execution fails."""
    memory = RedisChatMemory()
    mock_client = AsyncMock()

    mock_pipe = MagicMock()
    mock_pipe.execute = AsyncMock(side_effect=TimeoutError("Redis timed out"))
    mock_client.pipeline = MagicMock(return_value=mock_pipe)

    memory._client = mock_client
    memory._is_initialized = True

    success = await memory.add_turn("sess_fail", "test query", "test response")
    assert success is False


@pytest.mark.anyio
async def test_add_message():
    """Verify add_message stores single message and applies TTL."""
    memory = RedisChatMemory()
    mock_client = AsyncMock()

    mock_pipe = MagicMock()
    mock_pipe.execute = AsyncMock(return_value=[1, True, True])
    mock_client.pipeline = MagicMock(return_value=mock_pipe)

    memory._client = mock_client
    memory._is_initialized = True

    success = await memory.add_message("sess_msg", "user", "Hello single")
    assert success is True
    assert mock_pipe.rpush.call_count == 1
    mock_pipe.expire.assert_called_with("chat:session:sess_msg:messages", memory.settings.redis_chat_ttl_seconds)


@pytest.mark.anyio
async def test_clear_session():
    """Verify clear_session calls delete on the Redis key."""
    memory = RedisChatMemory()
    mock_client = AsyncMock()
    mock_client.delete = AsyncMock(return_value=1)

    memory._client = mock_client
    memory._is_initialized = True

    result = await memory.clear_session("sess_to_clear")
    assert result is True
    mock_client.delete.assert_called_once_with("chat:session:sess_to_clear:messages")


@pytest.mark.anyio
async def test_clear_session_resilience_on_error():
    """Verify clear_session returns False safely on error."""
    memory = RedisChatMemory()
    mock_client = AsyncMock()
    mock_client.delete = AsyncMock(side_effect=RedisError("Cluster error"))

    memory._client = mock_client
    memory._is_initialized = True

    result = await memory.clear_session("sess_to_clear")
    assert result is False


@pytest.mark.anyio
async def test_get_session_message_count():
    """Verify get_session_message_count retrieves list length."""
    memory = RedisChatMemory()
    mock_client = AsyncMock()
    mock_client.llen = AsyncMock(return_value=6)

    memory._client = mock_client
    memory._is_initialized = True

    count = await memory.get_session_message_count("sess_count")
    assert count == 6


@pytest.mark.anyio
async def test_close_lifecycle():
    """Verify close cleanly shuts down client and pool."""
    memory = RedisChatMemory()
    mock_client = AsyncMock()
    mock_client.aclose = AsyncMock()
    mock_pool = AsyncMock()
    mock_pool.aclose = AsyncMock()

    memory._client = mock_client
    memory._pool = mock_pool
    memory._is_initialized = True

    await memory.close()
    mock_client.aclose.assert_called_once()
    mock_pool.aclose.assert_called_once()
    assert memory._client is None
    assert memory._pool is None
    assert memory._is_initialized is False
