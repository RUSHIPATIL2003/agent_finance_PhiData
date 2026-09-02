"""Unit tests for GlobalRateCoordinator and rate limiting execution."""

import time
import pytest
from unittest.mock import MagicMock
from src.rag_agent.rate_limiter import GlobalRateCoordinator, execute_with_rate_limit


def test_rate_coordinator_initialization():
    """Verify coordinator initial parameters."""
    coordinator = GlobalRateCoordinator(
        min_delay_seconds=0.1,
        max_delay_seconds=0.2,
        tpm_hard_cap=250_000,
        tpm_safety_threshold=200_000,
    )
    assert coordinator.min_delay == 0.1
    assert coordinator.max_delay == 0.2
    assert coordinator.tpm_safety_threshold == 200_000
    assert coordinator.get_current_window_tokens() == 0


def test_token_tracking_and_purge():
    """Verify token usage recording and rolling window calculation."""
    coordinator = GlobalRateCoordinator(rolling_window_seconds=1.0)
    coordinator.record_token_usage(5000)
    coordinator.record_token_usage(10000)
    assert coordinator.get_current_window_tokens() == 15000

    # Wait for rolling window to expire
    time.sleep(1.1)
    assert coordinator.get_current_window_tokens() == 0


def test_jittered_backoff_calculation():
    """Verify exponential backoff calculation with jitter."""
    coordinator = GlobalRateCoordinator()
    b1 = coordinator.trigger_rate_limit_backoff(base_wait=4.0, retry_count=1)
    # 4.0 * 2^0 + (0.1..1.0) = ~4.1 - 5.0
    assert 4.1 <= b1 <= 5.1

    b2 = coordinator.trigger_rate_limit_backoff(base_wait=4.0, retry_count=2)
    # 4.0 * 2^1 + (0.1..1.0) = ~8.1 - 9.1
    assert 8.1 <= b2 <= 9.1


def test_execute_with_rate_limit_success():
    """Verify standard successful function execution through wrapper."""
    mock_func = MagicMock(return_value="Market Analysis Result")
    result = execute_with_rate_limit("TestAgent", mock_func, "NVDA", estimated_tokens=500)
    assert result == "Market Analysis Result"
    assert mock_func.call_count == 1


def test_execute_with_rate_limit_429_retry():
    """Verify retry handling on 429 rate limit error."""
    attempts = [0]

    def flaky_func():
        attempts[0] += 1
        if attempts[0] == 1:
            raise RuntimeError("HTTP 429: Too Many Requests")
        return "Success on retry"

    coordinator = GlobalRateCoordinator(min_delay_seconds=0.01, max_delay_seconds=0.02)
    # Patch base wait for fast unit testing
    coordinator.trigger_rate_limit_backoff = MagicMock(return_value=0.01)

    result = execute_with_rate_limit("RetryAgent", flaky_func, max_retries=3)
    assert result == "Success on retry"
    assert attempts[0] == 2
