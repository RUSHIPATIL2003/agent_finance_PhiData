"""Global Rate Coordinator & Token Budgeting Engine for Multi-Agent Operations.

Enforces:
  1. Global Single-Flight Request Coordination (prevents concurrent API collisions).
  2. Mandatory Non-Blocking Delay Spacers (4.5s - 5.0s between outbound requests).
  3. Rolling-Window Token Budgeting (pauses queue when approaching 200k / 250k TPM).
  4. Jittered Exponential Backoff on HTTP 429 rate limit exceptions.
"""

import asyncio
import collections
import logging
import random
import threading
import time
from typing import Any, Callable, Deque, Generator, Optional, Tuple

logger = logging.getLogger("rag_agent.rate_limiter")


class GlobalRateCoordinator:
    """Centralized rate limiting & token budgeting coordinator across all agents."""

    def __init__(
        self,
        min_delay_seconds: float = 4.5,
        max_delay_seconds: float = 5.0,
        tpm_hard_cap: int = 250_000,
        tpm_safety_threshold: int = 200_000,
        rolling_window_seconds: float = 60.0,
    ):
        self.min_delay = min_delay_seconds
        self.max_delay = max_delay_seconds
        self.tpm_hard_cap = tpm_hard_cap
        self.tpm_safety_threshold = tpm_safety_threshold
        self.window_seconds = rolling_window_seconds

        # Synchronization primitives for both sync and async runtimes
        self._sync_lock = threading.Lock()
        self._async_lock = asyncio.Lock()

        self._last_request_time: float = 0.0
        self._token_history: Deque[Tuple[float, int]] = collections.deque()
        self._global_pause_until: float = 0.0

    def _purge_stale_token_records(self, now: float) -> None:
        """Purge token records older than the 60s rolling window."""
        cutoff = now - self.window_seconds
        while self._token_history and self._token_history[0][0] < cutoff:
            self._token_history.popleft()

    def get_current_window_tokens(self) -> int:
        """Calculate active token usage in the rolling 60s window."""
        now = time.monotonic()
        self._purge_stale_token_records(now)
        return sum(tokens for _, tokens in self._token_history)

    def record_token_usage(self, token_count: int) -> None:
        """Record consumed tokens (estimated or exact) into the rolling window."""
        now = time.monotonic()
        self._token_history.append((now, max(token_count, 100)))
        total = self.get_current_window_tokens()
        logger.debug(
            "Recorded %d tokens. Current 60s window token usage: %d / %d TPM",
            token_count,
            total,
            self.tpm_hard_cap,
        )

    def trigger_rate_limit_backoff(self, base_wait: float = 6.0, retry_count: int = 1) -> float:
        """Calculate and apply jittered exponential backoff across all agents."""
        jitter = random.uniform(0.1, 1.0)
        backoff_seconds = (base_wait * (2 ** (retry_count - 1))) + jitter
        self._global_pause_until = time.monotonic() + backoff_seconds
        logger.warning(
            "HTTP 429 Rate Limit encountered! Pausing agent queue for %.2fs (retry attempt #%d)",
            backoff_seconds,
            retry_count,
        )
        return backoff_seconds

    def acquire_permission_sync(self, agent_name: str = "Agent", estimated_tokens: int = 1500) -> None:
        """Synchronous non-blocking check and delay enforcement."""
        with self._sync_lock:
            while True:
                now = time.monotonic()

                # 1. Check if global pause is active
                if now < self._global_pause_until:
                    sleep_time = self._global_pause_until - now
                    logger.info("[%s] Global queue paused. Waiting %.2fs...", agent_name, sleep_time)
                    time.sleep(sleep_time)
                    continue

                # 2. Check rolling Token Budget
                current_tokens = self.get_current_window_tokens()
                if (current_tokens + estimated_tokens) >= self.tpm_safety_threshold:
                    logger.warning(
                        "[%s] Approaching TPM ceiling (%d / %d TPM). Pausing for 60s window reset...",
                        agent_name,
                        current_tokens,
                        self.tpm_safety_threshold,
                    )
                    self._global_pause_until = time.monotonic() + self.window_seconds
                    time.sleep(self.window_seconds)
                    continue

                # 3. Enforce 4.5s - 5.0s delay spacer between calls
                if self._last_request_time > 0:
                    elapsed = now - self._last_request_time
                    spacer = random.uniform(self.min_delay, self.max_delay)
                    if elapsed < spacer:
                        wait = spacer - elapsed
                        logger.debug("[%s] Enforcing spacer delay of %.2fs...", agent_name, wait)
                        time.sleep(wait)

                self._last_request_time = time.monotonic()
                break

    async def acquire_permission_async(self, agent_name: str = "Agent", estimated_tokens: int = 1500) -> None:
        """Asynchronous non-blocking check and delay enforcement."""
        async with self._async_lock:
            while True:
                now = time.monotonic()

                # 1. Check global pause
                if now < self._global_pause_until:
                    sleep_time = self._global_pause_until - now
                    logger.info("[%s] Global queue paused. Waiting %.2fs...", agent_name, sleep_time)
                    await asyncio.sleep(sleep_time)
                    continue

                # 2. Check rolling Token Budget
                current_tokens = self.get_current_window_tokens()
                if (current_tokens + estimated_tokens) >= self.tpm_safety_threshold:
                    logger.warning(
                        "[%s] Approaching TPM ceiling (%d / %d TPM). Pausing for 60s window reset...",
                        agent_name,
                        current_tokens,
                        self.tpm_safety_threshold,
                    )
                    self._global_pause_until = time.monotonic() + self.window_seconds
                    await asyncio.sleep(self.window_seconds)
                    continue

                # 3. Enforce delay spacer
                if self._last_request_time > 0:
                    elapsed = now - self._last_request_time
                    spacer = random.uniform(self.min_delay, self.max_delay)
                    if elapsed < spacer:
                        wait = spacer - elapsed
                        logger.debug("[%s] Enforcing spacer delay of %.2fs...", agent_name, wait)
                        await asyncio.sleep(wait)

                self._last_request_time = time.monotonic()
                break


# Global Coordinator Singleton
rate_coordinator = GlobalRateCoordinator()


def execute_with_rate_limit(
    agent_name: str,
    target_func: Callable[..., Any],
    *args: Any,
    estimated_tokens: int = 1500,
    max_retries: int = 4,
    **kwargs: Any,
) -> Any:
    """Execute a synchronous function with global rate coordination and retry backoff."""
    for attempt in range(1, max_retries + 1):
        rate_coordinator.acquire_permission_sync(agent_name=agent_name, estimated_tokens=estimated_tokens)
        try:
            result = target_func(*args, **kwargs)
            # Estimate tokens: ~4 characters per token
            token_est = len(str(result)) // 4 + estimated_tokens
            rate_coordinator.record_token_usage(token_est)
            return result
        except Exception as exc:
            err_msg = str(exc).lower()
            if "429" in err_msg or "too many requests" in err_msg or "resource_exhausted" in err_msg:
                backoff = rate_coordinator.trigger_rate_limit_backoff(base_wait=6.0, retry_count=attempt)
                if attempt == max_retries:
                    raise RuntimeError(f"[{agent_name}] Maximum rate-limit retries exceeded: {exc}") from exc
                time.sleep(backoff)
            else:
                raise exc
