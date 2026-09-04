"""Episodic Chat Memory caching layer backed by Redis and connection pooling."""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional
import redis.asyncio as aioredis
from redis.exceptions import RedisError

from src.rag_agent.config import Settings, get_settings

logger = logging.getLogger("rag_agent.memory")


class RedisChatMemory:
    """High-performance episodic chat memory cache with connection pooling and TTL auto-expiry."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._pool: Optional[aioredis.ConnectionPool] = None
        self._client: Optional[aioredis.Redis] = None
        self._is_initialized = False

    def _get_key(self, session_id: str) -> str:
        """Format the Redis key for a given session identifier."""
        clean_id = session_id.strip()
        return f"chat:session:{clean_id}:messages"

    async def initialize(self) -> bool:
        """Initialize the Redis connection pool and verify connectivity."""
        if self._is_initialized and self._client is not None:
            return True

        try:
            logger.info("Initializing Redis connection pool for episodic chat memory...")
            if self.settings.redis_password:
                self._pool = aioredis.ConnectionPool(
                    host=self.settings.redis_host,
                    port=self.settings.redis_port,
                    password=self.settings.redis_password,
                    db=self.settings.redis_db,
                    max_connections=self.settings.redis_max_connections,
                    socket_timeout=self.settings.redis_socket_timeout,
                    socket_connect_timeout=self.settings.redis_socket_connect_timeout,
                    decode_responses=True,
                )
            else:
                self._pool = aioredis.ConnectionPool.from_url(
                    self.settings.redis_url,
                    max_connections=self.settings.redis_max_connections,
                    socket_timeout=self.settings.redis_socket_timeout,
                    socket_connect_timeout=self.settings.redis_socket_connect_timeout,
                    decode_responses=True,
                )

            self._client = aioredis.Redis(connection_pool=self._pool)
            # Test connectivity with ping
            await asyncio.wait_for(self._client.ping(), timeout=self.settings.redis_socket_connect_timeout)
            self._is_initialized = True
            logger.info(
                "Redis episodic memory connected successfully. TTL: %ds, Pool Size: %d",
                self.settings.redis_chat_ttl_seconds,
                self.settings.redis_max_connections,
            )
            return True
        except (RedisError, asyncio.TimeoutError, ConnectionError, OSError, Exception) as exc:
            logger.warning(
                "Redis connection initialization failed (%s). Memory will operate in resilient fallback mode.",
                exc,
            )
            self._is_initialized = False
            return False

    async def close(self) -> None:
        """Gracefully disconnect and close the Redis connection pool."""
        try:
            if self._client is not None:
                await self._client.aclose()
                self._client = None
            if self._pool is not None:
                await self._pool.aclose()
                self._pool = None
            self._is_initialized = False
            logger.info("Redis episodic memory connections closed.")
        except Exception as exc:
            logger.warning("Error while closing Redis connection pool: %s", exc)

    async def ping(self) -> bool:
        """Check if Redis connection is active and responsive."""
        if not self._is_initialized or self._client is None:
            return False
        try:
            res = await asyncio.wait_for(self._client.ping(), timeout=self.settings.redis_socket_timeout)
            return bool(res)
        except (RedisError, asyncio.TimeoutError, ConnectionError, OSError, Exception) as exc:
            logger.debug("Redis ping failed: %s", exc)
            return False

    async def get_history(
        self,
        session_id: Optional[str],
        max_messages: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve recent ordered conversation messages for a session.

        Returns an empty list if Redis is unavailable or session_id is not set.
        """
        if not session_id or not session_id.strip():
            return []

        if not self._is_initialized or self._client is None:
            # Attempt auto-connect if uninitialized
            connected = await self.initialize()
            if not connected or self._client is None:
                return []

        key = self._get_key(session_id)
        limit = max_messages or (self.settings.redis_max_history_turns * 2)

        try:
            # Fetch recent items from list
            start_index = -limit if limit > 0 else 0
            raw_entries = await asyncio.wait_for(
                self._client.lrange(key, start_index, -1),
                timeout=self.settings.redis_socket_timeout,
            )

            history: List[Dict[str, Any]] = []
            for item in raw_entries:
                try:
                    if isinstance(item, str):
                        history.append(json.loads(item))
                    elif isinstance(item, dict):
                        history.append(item)
                except (json.JSONDecodeError, TypeError) as parse_err:
                    logger.warning("Failed to decode cached message item '%s': %s", item, parse_err)
            return history
        except (RedisError, asyncio.TimeoutError, ConnectionError, OSError, Exception) as exc:
            logger.warning(
                "Redis get_history failed for session [%s]: %s. Falling back to empty history.",
                session_id,
                exc,
            )
            return []

    async def add_turn(
        self,
        session_id: Optional[str],
        user_message: str,
        assistant_response: str,
        ttl: Optional[int] = None,
    ) -> bool:
        """Asynchronously append a user message and assistant response turn to Redis.

        Sets strict TTL expiration and trims list to prevent memory bloat.
        """
        if not session_id or not session_id.strip():
            return False

        if not self._is_initialized or self._client is None:
            connected = await self.initialize()
            if not connected or self._client is None:
                return False

        key = self._get_key(session_id)
        ttl_seconds = ttl if ttl is not None else self.settings.redis_chat_ttl_seconds
        max_items = self.settings.redis_max_history_turns * 2
        now = time.time()

        user_payload = json.dumps({
            "role": "user",
            "content": user_message.strip(),
            "timestamp": now,
        })
        assistant_payload = json.dumps({
            "role": "assistant",
            "content": assistant_response.strip(),
            "timestamp": now + 0.001,
        })

        try:
            pipe = self._client.pipeline(transaction=True)
            pipe.rpush(key, user_payload, assistant_payload)
            # Retain only the most recent items
            pipe.ltrim(key, -max_items, -1)
            # Reset TTL on active session
            pipe.expire(key, ttl_seconds)
            await asyncio.wait_for(pipe.execute(), timeout=self.settings.redis_socket_timeout)
            logger.debug("Successfully saved conversation turn for session [%s] with TTL %ds", session_id, ttl_seconds)
            return True
        except (RedisError, asyncio.TimeoutError, ConnectionError, OSError, Exception) as exc:
            logger.warning(
                "Redis add_turn failed for session [%s]: %s. Turn completed statelessly.",
                session_id,
                exc,
            )
            return False

    async def add_message(
        self,
        session_id: Optional[str],
        role: str,
        content: str,
        ttl: Optional[int] = None,
    ) -> bool:
        """Append a single message to session memory."""
        if not session_id or not session_id.strip():
            return False

        if not self._is_initialized or self._client is None:
            connected = await self.initialize()
            if not connected or self._client is None:
                return False

        key = self._get_key(session_id)
        ttl_seconds = ttl if ttl is not None else self.settings.redis_chat_ttl_seconds
        max_items = self.settings.redis_max_history_turns * 2
        payload = json.dumps({
            "role": role.strip().lower(),
            "content": content.strip(),
            "timestamp": time.time(),
        })

        try:
            pipe = self._client.pipeline(transaction=True)
            pipe.rpush(key, payload)
            pipe.ltrim(key, -max_items, -1)
            pipe.expire(key, ttl_seconds)
            await asyncio.wait_for(pipe.execute(), timeout=self.settings.redis_socket_timeout)
            return True
        except (RedisError, asyncio.TimeoutError, ConnectionError, OSError, Exception) as exc:
            logger.warning("Redis add_message failed for session [%s]: %s", session_id, exc)
            return False

    async def clear_session(self, session_id: Optional[str]) -> bool:
        """Explicitly purge session memory cache from Redis."""
        if not session_id or not session_id.strip():
            return False

        if not self._is_initialized or self._client is None:
            connected = await self.initialize()
            if not connected or self._client is None:
                return False

        key = self._get_key(session_id)
        try:
            await asyncio.wait_for(self._client.delete(key), timeout=self.settings.redis_socket_timeout)
            logger.info("Purged session memory cache for [%s]", session_id)
            return True
        except (RedisError, asyncio.TimeoutError, ConnectionError, OSError, Exception) as exc:
            logger.warning("Redis clear_session failed for session [%s]: %s", session_id, exc)
            return False

    async def get_session_message_count(self, session_id: Optional[str]) -> int:
        """Get the total count of messages stored for a session."""
        if not session_id or not session_id.strip():
            return 0

        if not self._is_initialized or self._client is None:
            connected = await self.initialize()
            if not connected or self._client is None:
                return 0

        key = self._get_key(session_id)
        try:
            count = await asyncio.wait_for(self._client.llen(key), timeout=self.settings.redis_socket_timeout)
            return int(count)
        except (RedisError, asyncio.TimeoutError, ConnectionError, OSError, Exception) as exc:
            logger.debug("Redis llen failed for session [%s]: %s", session_id, exc)
            return 0

    @staticmethod
    def format_history_context(
        history: List[Dict[str, Any]],
        current_query: str,
    ) -> str:
        """Format historical turns into an augmented context prompt for the LLM."""
        if not history:
            return current_query.strip()

        formatted_lines = [
            "### Previous Conversation History:",
        ]
        for turn in history:
            role = turn.get("role", "user").capitalize()
            content = str(turn.get("content", "")).strip()
            if content:
                formatted_lines.append(f"**{role}**: {content}")

        formatted_lines.append("\n### Current User Query:")
        formatted_lines.append(current_query.strip())
        return "\n".join(formatted_lines)


# Global episodic memory singleton
chat_memory = RedisChatMemory()


async def get_chat_memory() -> RedisChatMemory:
    """FastAPI dependency for accessing the Redis chat memory cache."""
    return chat_memory
