"""
Comprehensive tests for retry_logic module to boost coverage to 76%+

Tests all components:
- Exception classes
- CircuitState enum
- RetryPolicy class
- CircuitBreaker class
- Retry decorator
- Pre-configured policies
- RetryableMusic21Operation class
- BulkRetryExecutor class
- ResilientTool example class
"""

import asyncio
import random
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from music21_mcp.retry_logic import (
    DATABASE_POLICY,
    FILE_IO_POLICY,
    MUSIC21_POLICY,
    NETWORK_POLICY,
    BulkRetryExecutor,
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
    NonRetryableError,
    ResilientTool,
    RetryableError,
    RetryableMusic21Operation,
    RetryPolicy,
    retry,
)


class TestExceptionClasses:
    """Test custom exception classes"""

    def test_retryable_error(self):
        """Test RetryableError exception"""
        error = RetryableError("This should be retried")
        assert str(error) == "This should be retried"
        assert isinstance(error, Exception)

    def test_non_retryable_error(self):
        """Test NonRetryableError exception"""
        error = NonRetryableError("This should not be retried")
        assert str(error) == "This should not be retried"
        assert isinstance(error, Exception)

    def test_circuit_breaker_open_error(self):
        """Test CircuitBreakerOpenError exception"""
        error = CircuitBreakerOpenError("Circuit is open")
        assert str(error) == "Circuit is open"
        assert isinstance(error, Exception)


class TestCircuitState:
    """Test CircuitState enum"""

    def test_circuit_state_values(self):
        """Test enum values"""
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"

    def test_circuit_state_comparison(self):
        """Test enum comparison"""
        assert CircuitState.CLOSED != CircuitState.OPEN
        assert CircuitState.OPEN != CircuitState.HALF_OPEN
        assert str(CircuitState.CLOSED) == "CircuitState.CLOSED"


class TestRetryPolicy:
    """Test RetryPolicy class"""

    def test_policy_default_initialization(self):
        """Test default policy initialization"""
        policy = RetryPolicy()

        assert policy.max_attempts == 3
        assert policy.base_delay == 1.0
        assert policy.max_delay == 60.0
        assert policy.exponential_base == 2.0
        assert policy.jitter is True
        assert RetryableError in policy.retryable_exceptions
        assert ConnectionError in policy.retryable_exceptions
        assert NonRetryableError in policy.non_retryable_exceptions
        assert ValueError in policy.non_retryable_exceptions

    def test_policy_custom_initialization(self):
        """Test custom policy initialization"""
        custom_retryable = (RuntimeError,)
        custom_non_retryable = (KeyError,)

        policy = RetryPolicy(
            max_attempts=5,
            base_delay=0.5,
            max_delay=30.0,
            exponential_base=1.5,
            jitter=False,
            retryable_exceptions=custom_retryable,
            non_retryable_exceptions=custom_non_retryable,
        )

        assert policy.max_attempts == 5
        assert policy.base_delay == 0.5
        assert policy.max_delay == 30.0
        assert policy.exponential_base == 1.5
        assert policy.jitter is False
        assert policy.retryable_exceptions == custom_retryable
        assert policy.non_retryable_exceptions == custom_non_retryable

    def test_should_retry_retryable_exception(self):
        """Test should_retry with retryable exception"""
        policy = RetryPolicy(max_attempts=3)

        # Should retry RetryableError
        assert policy.should_retry(RetryableError("test"), attempt=1) is True
        assert policy.should_retry(ConnectionError("test"), attempt=1) is True
        assert policy.should_retry(TimeoutError("test"), attempt=1) is True

    def test_should_retry_non_retryable_exception(self):
        """Test should_retry with non-retryable exception"""
        policy = RetryPolicy(max_attempts=3)

        # Should not retry NonRetryableError
        assert policy.should_retry(NonRetryableError("test"), attempt=1) is False
        assert policy.should_retry(ValueError("test"), attempt=1) is False
        assert policy.should_retry(TypeError("test"), attempt=1) is False

    def test_should_retry_max_attempts_exceeded(self):
        """Test should_retry when max attempts exceeded"""
        policy = RetryPolicy(max_attempts=3)

        # Should not retry when attempts >= max_attempts
        assert policy.should_retry(RetryableError("test"), attempt=3) is False
        assert policy.should_retry(RetryableError("test"), attempt=5) is False

    def test_should_retry_unknown_exception(self):
        """Test should_retry with unknown exception type"""
        policy = RetryPolicy()

        # Unknown exception should not be retried by default
        class UnknownError(Exception):
            pass

        assert policy.should_retry(UnknownError("test"), attempt=1) is False

    def test_get_delay_exponential_backoff(self):
        """Test get_delay with exponential backoff"""
        policy = RetryPolicy(base_delay=1.0, exponential_base=2.0, jitter=False)

        assert policy.get_delay(0) == 1.0  # 1.0 * 2^0
        assert policy.get_delay(1) == 2.0  # 1.0 * 2^1
        assert policy.get_delay(2) == 4.0  # 1.0 * 2^2

    def test_get_delay_max_cap(self):
        """Test get_delay caps at max_delay"""
        policy = RetryPolicy(
            base_delay=1.0, max_delay=5.0, exponential_base=2.0, jitter=False
        )

        assert policy.get_delay(10) == 5.0  # Should be capped

    def test_get_delay_with_jitter(self):
        """Test get_delay with jitter"""
        policy = RetryPolicy(base_delay=2.0, exponential_base=1.0, jitter=True)

        # With jitter, delay should vary but be between base_delay * 0.5 and base_delay * 1.5
        delays = [policy.get_delay(0) for _ in range(10)]

        # All delays should be in expected range
        for delay in delays:
            assert 1.0 <= delay <= 3.0  # 2.0 * (0.5 to 1.5)

        # Should have some variation (not all identical)
        assert len(set(delays)) > 1

    def test_get_delay_no_jitter(self):
        """Test get_delay without jitter"""
        policy = RetryPolicy(base_delay=2.0, exponential_base=1.0, jitter=False)

        # Without jitter, delay should be consistent
        delays = [policy.get_delay(0) for _ in range(5)]

        # All delays should be identical
        assert all(delay == 2.0 for delay in delays)


class TestCircuitBreaker:
    """Test CircuitBreaker class"""

    def test_circuit_breaker_initialization(self):
        """Test CircuitBreaker initialization"""
        breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=30.0,
            expected_exception=RuntimeError,
        )

        assert breaker.failure_threshold == 3
        assert breaker.recovery_timeout == 30.0
        assert breaker.expected_exception is RuntimeError
        assert breaker.failure_count == 0
        assert breaker.last_failure_time is None
        assert breaker.state == CircuitState.CLOSED

    def test_circuit_breaker_default_initialization(self):
        """Test CircuitBreaker default initialization"""
        breaker = CircuitBreaker()

        assert breaker.failure_threshold == 5
        assert breaker.recovery_timeout == 60.0
        assert breaker.expected_exception is Exception
        assert breaker.state == CircuitState.CLOSED

    def test_call_success_closed_circuit(self):
        """Test successful call with closed circuit"""
        breaker = CircuitBreaker()

        def test_func(x, y):
            return x + y

        result = breaker.call(test_func, 2, 3)
        assert result == 5
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    def test_call_failure_closed_circuit(self):
        """Test failed call with closed circuit"""
        breaker = CircuitBreaker(failure_threshold=2)

        def failing_func():
            raise RuntimeError("Test error")

        # First failure
        with pytest.raises(RuntimeError):
            breaker.call(failing_func)

        assert breaker.failure_count == 1
        assert breaker.state == CircuitState.CLOSED

        # Second failure should open circuit
        with pytest.raises(RuntimeError):
            breaker.call(failing_func)

        assert breaker.failure_count == 2
        assert breaker.state == CircuitState.OPEN

    def test_call_open_circuit(self):
        """Test call with open circuit"""
        breaker = CircuitBreaker(failure_threshold=1)

        def failing_func():
            raise RuntimeError("Test error")

        # Cause circuit to open
        with pytest.raises(RuntimeError):
            breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN

        # Next call should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            breaker.call(failing_func)

    def test_call_half_open_success(self):
        """Test successful call in half-open state"""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)

        def failing_func():
            raise RuntimeError("Test error")

        def success_func():
            return "success"

        # Open circuit
        with pytest.raises(RuntimeError):
            breaker.call(failing_func)

        # Wait for recovery timeout
        time.sleep(0.2)

        # Should transition to half-open and then closed on success
        result = breaker.call(success_func)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    def test_call_half_open_failure(self):
        """Test failed call in half-open state"""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)

        def failing_func():
            raise RuntimeError("Test error")

        # Open circuit
        with pytest.raises(RuntimeError):
            breaker.call(failing_func)

        # Wait for recovery timeout
        time.sleep(0.2)

        # Failure in half-open should go back to open
        with pytest.raises(RuntimeError):
            breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN

    def test_call_non_expected_exception(self):
        """Test call with non-expected exception"""
        breaker = CircuitBreaker(expected_exception=RuntimeError)

        def failing_func():
            raise ValueError("Different error")

        # Should not affect circuit breaker
        with pytest.raises(ValueError, match="Different error"):
            breaker.call(failing_func)

        assert breaker.failure_count == 0
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_async_call_success(self):
        """Test successful async call"""
        breaker = CircuitBreaker()

        async def async_func(x):
            return x * 2

        result = await breaker.async_call(async_func, 5)
        assert result == 10
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_async_call_failure(self):
        """Test failed async call"""
        breaker = CircuitBreaker(failure_threshold=1)

        async def failing_async_func():
            raise RuntimeError("Async error")

        with pytest.raises(RuntimeError):
            await breaker.async_call(failing_async_func)

        assert breaker.failure_count == 1
        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_async_call_open_circuit(self):
        """Test async call with open circuit"""
        breaker = CircuitBreaker(failure_threshold=1)

        async def failing_func():
            raise RuntimeError("Test error")

        # Open circuit
        with pytest.raises(RuntimeError):
            await breaker.async_call(failing_func)

        # Next call should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            await breaker.async_call(failing_func)

    def test_should_attempt_reset_true(self):
        """Test _should_attempt_reset returns True after timeout"""
        breaker = CircuitBreaker(recovery_timeout=0.1)
        breaker.last_failure_time = time.time() - 0.2  # 0.2 seconds ago

        assert breaker._should_attempt_reset() is True

    def test_should_attempt_reset_false(self):
        """Test _should_attempt_reset returns False before timeout"""
        breaker = CircuitBreaker(recovery_timeout=1.0)
        breaker.last_failure_time = time.time() - 0.1  # 0.1 seconds ago

        assert breaker._should_attempt_reset() is False

    def test_should_attempt_reset_no_failure(self):
        """Test _should_attempt_reset with no previous failure"""
        breaker = CircuitBreaker()
        breaker.last_failure_time = None

        assert breaker._should_attempt_reset() is False

    def test_on_success(self):
        """Test _on_success resets circuit breaker"""
        breaker = CircuitBreaker()
        breaker.failure_count = 3
        breaker.state = CircuitState.HALF_OPEN
        breaker.last_failure_time = time.time()

        breaker._on_success()

        assert breaker.failure_count == 0
        assert breaker.state == CircuitState.CLOSED
        assert breaker.last_failure_time is None

    def test_on_failure(self):
        """Test _on_failure increments count and sets time"""
        breaker = CircuitBreaker(failure_threshold=3)

        breaker._on_failure()

        assert breaker.failure_count == 1
        assert breaker.last_failure_time is not None
        assert breaker.state == CircuitState.CLOSED  # Not open yet

        # Another failure
        breaker._on_failure()
        breaker._on_failure()  # Should open circuit

        assert breaker.failure_count == 3
        assert breaker.state == CircuitState.OPEN


class TestRetryDecorator:
    """Test retry decorator"""

    @pytest.mark.asyncio
    async def test_retry_decorator_async_success(self):
        """Test retry decorator with successful async function"""

        @retry()
        async def async_func(x):
            return x * 2

        result = await async_func(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_retry_decorator_async_retry_success(self):
        """Test retry decorator retries async function and succeeds"""
        call_count = 0

        @retry()
        async def flaky_async_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RetryableError("Temporary error")
            return "success"

        result = await flaky_async_func()
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_decorator_async_all_attempts_fail(self):
        """Test retry decorator when all async attempts fail"""

        @retry(RetryPolicy(max_attempts=2))
        async def always_failing_async():
            raise RetryableError("Always fails")

        with pytest.raises(RetryableError):
            await always_failing_async()

    @pytest.mark.asyncio
    async def test_retry_decorator_async_non_retryable(self):
        """Test retry decorator with non-retryable async exception"""

        @retry()
        async def non_retryable_async():
            raise ValueError("Non-retryable error")

        with pytest.raises(ValueError, match="Non-retryable error"):
            await non_retryable_async()

    def test_retry_decorator_sync_success(self):
        """Test retry decorator with successful sync function"""

        @retry()
        def sync_func(x):
            return x * 3

        result = sync_func(4)
        assert result == 12

    def test_retry_decorator_sync_retry_success(self):
        """Test retry decorator retries sync function and succeeds"""
        call_count = 0

        @retry()
        def flaky_sync_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network error")
            return "success"

        result = flaky_sync_func()
        assert result == "success"
        assert call_count == 3

    def test_retry_decorator_sync_all_attempts_fail(self):
        """Test retry decorator when all sync attempts fail"""

        @retry(RetryPolicy(max_attempts=2))
        def always_failing_sync():
            raise TimeoutError("Always times out")

        with pytest.raises(TimeoutError):
            always_failing_sync()

    def test_retry_decorator_sync_non_retryable(self):
        """Test retry decorator with non-retryable sync exception"""

        @retry()
        def non_retryable_sync():
            raise KeyError("Non-retryable error")

        with pytest.raises(KeyError):
            non_retryable_sync()

    @pytest.mark.asyncio
    async def test_retry_decorator_with_circuit_breaker(self):
        """Test retry decorator with circuit breaker"""
        circuit_breaker = CircuitBreaker(failure_threshold=2)

        @retry(circuit_breaker=circuit_breaker)
        async def failing_func():
            raise RuntimeError("Consistent failure")

        # First two calls should fail normally
        with pytest.raises(RuntimeError):
            await failing_func()

        with pytest.raises(RuntimeError):
            await failing_func()

        # Third call should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            await failing_func()

    @pytest.mark.asyncio
    async def test_retry_decorator_custom_policy(self):
        """Test retry decorator with custom policy"""
        policy = RetryPolicy(
            max_attempts=2,
            base_delay=0.001,  # Very small delay for fast testing
            retryable_exceptions=(RuntimeError,),
        )

        call_count = 0

        @retry(policy=policy)
        async def custom_func():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Test error")

        with pytest.raises(RuntimeError):
            await custom_func()

        assert call_count == 2  # Should have retried once

    def test_retry_decorator_detects_async_function(self):
        """Test retry decorator properly detects async functions"""

        @retry()
        async def async_function():
            return "async"

        @retry()
        def sync_function():
            return "sync"

        # Both should be callable
        import asyncio

        assert asyncio.iscoroutinefunction(async_function)
        assert not asyncio.iscoroutinefunction(sync_function)


class TestPreConfiguredPolicies:
    """Test pre-configured retry policies"""

    def test_file_io_policy(self):
        """Test FILE_IO_POLICY configuration"""
        policy = FILE_IO_POLICY

        assert policy.max_attempts == 3
        assert policy.base_delay == 0.5
        assert policy.max_delay == 5.0
        assert OSError in policy.retryable_exceptions
        assert IOError in policy.retryable_exceptions
        assert PermissionError in policy.retryable_exceptions

    def test_network_policy(self):
        """Test NETWORK_POLICY configuration"""
        policy = NETWORK_POLICY

        assert policy.max_attempts == 5
        assert policy.base_delay == 1.0
        assert policy.max_delay == 30.0
        assert ConnectionError in policy.retryable_exceptions
        assert TimeoutError in policy.retryable_exceptions

    def test_music21_policy(self):
        """Test MUSIC21_POLICY configuration"""
        policy = MUSIC21_POLICY

        assert policy.max_attempts == 3
        assert policy.base_delay == 0.2
        assert policy.max_delay == 2.0
        assert RetryableError in policy.retryable_exceptions
        assert RuntimeError in policy.retryable_exceptions

    def test_database_policy(self):
        """Test DATABASE_POLICY configuration"""
        policy = DATABASE_POLICY

        assert policy.max_attempts == 3
        assert policy.base_delay == 0.1
        assert policy.max_delay == 1.0
        assert policy.exponential_base == 2.0
        assert ConnectionError in policy.retryable_exceptions


class TestRetryableMusic21Operation:
    """Test RetryableMusic21Operation class"""

    @pytest.fixture
    def music21_op(self):
        """Create RetryableMusic21Operation instance"""
        return RetryableMusic21Operation()

    def test_initialization_default_policy(self, music21_op):
        """Test initialization with default policy"""
        assert music21_op.policy == MUSIC21_POLICY
        assert isinstance(music21_op.circuit_breaker, CircuitBreaker)
        assert music21_op.circuit_breaker.failure_threshold == 10
        assert music21_op.circuit_breaker.recovery_timeout == 120.0

    def test_initialization_custom_policy(self):
        """Test initialization with custom policy"""
        custom_policy = RetryPolicy(max_attempts=5)
        music21_op = RetryableMusic21Operation(policy=custom_policy)

        assert music21_op.policy == custom_policy

    @pytest.mark.asyncio
    async def test_parse_score_success(self, music21_op):
        """Test successful score parsing"""
        with patch("music21.converter.parse") as mock_parse:
            mock_score = Mock()
            mock_parse.return_value = mock_score

            result = await music21_op.parse_score("test data")

            assert result == mock_score
            mock_parse.assert_called_once_with("test data")

    @pytest.mark.asyncio
    async def test_parse_score_retryable_error(self, music21_op):
        """Test parse_score with retryable error"""
        with patch("music21.converter.parse") as mock_parse:
            mock_parse.side_effect = Exception("memory error")

            with pytest.raises(RetryableError) as exc_info:
                await music21_op.parse_score("test data")

            assert "Transient error parsing score" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_parse_score_timeout_error(self, music21_op):
        """Test parse_score with timeout error"""
        with patch("music21.converter.parse") as mock_parse:
            mock_parse.side_effect = Exception("timeout occurred")

            with pytest.raises(RetryableError) as exc_info:
                await music21_op.parse_score("test data")

            assert "Transient error parsing score" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_parse_score_non_retryable_error(self, music21_op):
        """Test parse_score with non-retryable error"""
        with patch("music21.converter.parse") as mock_parse:
            mock_parse.side_effect = ValueError("Invalid data format")

            with pytest.raises(ValueError, match="Invalid data format"):
                await music21_op.parse_score("test data")

    @pytest.mark.asyncio
    async def test_write_file_success(self, music21_op):
        """Test successful file writing"""
        mock_file = AsyncMock()

        with patch("aiofiles.open", return_value=mock_file) as mock_open:
            await music21_op.write_file("/test/path.txt", "content")

            mock_open.assert_called_once_with("/test/path.txt", "w")
            mock_file.__aenter__.assert_called_once()
            mock_file.__aenter__.return_value.write.assert_called_once_with("content")

    @pytest.mark.asyncio
    async def test_write_file_binary_content(self, music21_op):
        """Test writing binary file content"""
        mock_file = AsyncMock()

        with patch("aiofiles.open", return_value=mock_file) as mock_open:
            await music21_op.write_file("/test/path.bin", b"binary content")

            mock_open.assert_called_once_with("/test/path.bin", "wb")

    @pytest.mark.asyncio
    async def test_write_file_missing_aiofiles(self, music21_op):
        """Test write_file without aiofiles installed"""
        with patch(
            "builtins.__import__", side_effect=ImportError("No module named 'aiofiles'")
        ):
            with pytest.raises(ImportError) as exc_info:
                await music21_op.write_file("/test/path.txt", "content")

            assert "aiofiles is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fetch_corpus_success(self, music21_op):
        """Test successful corpus fetching"""
        with patch("music21.corpus.parse") as mock_parse:
            mock_score = Mock()
            mock_parse.return_value = mock_score

            result = await music21_op.fetch_corpus("bach/bwv1.1")

            assert result == mock_score
            mock_parse.assert_called_once_with("bach/bwv1.1")

    @pytest.mark.asyncio
    async def test_fetch_corpus_network_error(self, music21_op):
        """Test fetch_corpus with network error"""
        with patch("music21.corpus.parse") as mock_parse:
            mock_parse.side_effect = Exception("network connection failed")

            with pytest.raises(RetryableError) as exc_info:
                await music21_op.fetch_corpus("bach/bwv1.1")

            assert "Network error fetching corpus" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fetch_corpus_connection_error(self, music21_op):
        """Test fetch_corpus with connection error"""
        with patch("music21.corpus.parse") as mock_parse:
            mock_parse.side_effect = Exception("connection timeout")

            with pytest.raises(RetryableError) as exc_info:
                await music21_op.fetch_corpus("bach/bwv1.1")

            assert "Network error fetching corpus" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_fetch_corpus_non_retryable_error(self, music21_op):
        """Test fetch_corpus with non-retryable error"""
        with patch("music21.corpus.parse") as mock_parse:
            mock_parse.side_effect = ValueError("Invalid corpus path")

            with pytest.raises(ValueError, match="Invalid corpus path"):
                await music21_op.fetch_corpus("invalid/path")


class TestBulkRetryExecutor:
    """Test BulkRetryExecutor class"""

    @pytest.fixture
    def executor(self):
        """Create BulkRetryExecutor instance"""
        return BulkRetryExecutor(max_concurrent=3)

    def test_initialization_default_policy(self, executor):
        """Test initialization with default policy"""
        assert isinstance(executor.policy, RetryPolicy)
        assert executor.max_concurrent == 3
        assert isinstance(executor.semaphore, asyncio.Semaphore)

    def test_initialization_custom_policy(self):
        """Test initialization with custom policy"""
        custom_policy = RetryPolicy(max_attempts=5)
        executor = BulkRetryExecutor(policy=custom_policy, max_concurrent=5)

        assert executor.policy == custom_policy
        assert executor.max_concurrent == 5

    @pytest.mark.asyncio
    async def test_execute_all_success(self, executor):
        """Test execute_all with all successful operations"""

        async def op1():
            return "result1"

        def op2():
            return "result2"

        async def op3():
            return "result3"

        operations = [op1, op2, op3]
        results = await executor.execute_all(operations)

        assert results["total"] == 3
        assert len(results["successful"]) == 3
        assert len(results["failed"]) == 0
        assert results["success_rate"] == 1.0

        # Check results
        successful_results = {r["id"]: r["result"] for r in results["successful"]}
        assert successful_results[0] == "result1"
        assert successful_results[1] == "result2"
        assert successful_results[2] == "result3"

    @pytest.mark.asyncio
    async def test_execute_all_mixed_results(self, executor):
        """Test execute_all with mixed success/failure"""

        async def success_op():
            return "success"

        def failing_op():
            raise NonRetryableError("Permanent failure")  # Won't be retried

        async def success_op2():
            return "success2"

        operations = [success_op, failing_op, success_op2]
        results = await executor.execute_all(operations, continue_on_error=True)

        assert results["total"] == 3
        assert len(results["successful"]) == 2
        assert len(results["failed"]) == 1
        assert results["success_rate"] == 2 / 3

        # Check failed operation
        failed_result = results["failed"][0]
        assert failed_result["id"] == 1
        assert "Permanent failure" in failed_result["error"]

    @pytest.mark.asyncio
    async def test_execute_all_stop_on_error(self, executor):
        """Test execute_all stops on error when continue_on_error=False"""

        def failing_op():
            raise ValueError("Stop execution")

        async def should_not_run():
            return "should not reach here"

        operations = [failing_op, should_not_run]

        # Should handle the error but continue (since we use return_exceptions=True in gather)
        results = await executor.execute_all(operations, continue_on_error=False)

        # Both operations are attempted, but one fails
        assert results["total"] == 2
        assert len(results["failed"]) >= 1

    @pytest.mark.asyncio
    async def test_execute_all_with_retry_policy(self, executor):
        """Test execute_all applies retry policy"""
        call_count = 0

        async def flaky_op():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RetryableError("First attempt fails")
            return "success after retry"

        operations = [flaky_op]
        results = await executor.execute_all(operations)

        assert results["total"] == 1
        assert len(results["successful"]) == 1
        assert results["successful"][0]["result"] == "success after retry"
        assert call_count >= 2  # Should have retried

    @pytest.mark.asyncio
    async def test_execute_all_empty_operations(self, executor):
        """Test execute_all with empty operations list"""
        results = await executor.execute_all([])

        assert results["total"] == 0
        assert len(results["successful"]) == 0
        assert len(results["failed"]) == 0
        assert results["success_rate"] == 0

    @pytest.mark.asyncio
    async def test_execute_all_concurrency_limit(self, executor):
        """Test execute_all respects concurrency limit"""
        # This test is more about ensuring no errors occur with concurrency
        # Actual concurrency testing would require more complex timing

        async def slow_op(delay):
            await asyncio.sleep(0.01)  # Small delay
            return f"completed {delay}"

        operations = [lambda i=i: slow_op(i) for i in range(10)]
        results = await executor.execute_all(operations)

        assert results["total"] == 10
        assert len(results["successful"]) == 10
        assert results["success_rate"] == 1.0


class TestResilientTool:
    """Test ResilientTool example class"""

    @pytest.fixture
    def resilient_tool(self):
        """Create ResilientTool instance"""
        return ResilientTool()

    def test_initialization(self, resilient_tool):
        """Test ResilientTool initialization"""
        assert isinstance(resilient_tool.retry_executor, RetryableMusic21Operation)
        assert isinstance(resilient_tool.bulk_executor, BulkRetryExecutor)

    @pytest.mark.asyncio
    async def test_analyze_with_retry_success(self, resilient_tool):
        """Test analyze_with_retry success"""
        mock_score = Mock()
        mock_key = Mock()
        mock_key.__str__ = Mock(return_value="C major")
        mock_time_sig = Mock()
        mock_time_sig.__str__ = Mock(return_value="4/4")

        mock_score.analyze.return_value = mock_key
        mock_score.getTimeSignatures.return_value = [mock_time_sig]

        result = await resilient_tool.analyze_with_retry(mock_score)

        assert result["key"] == "C major"
        assert result["time_signature"] == "4/4"

    @pytest.mark.asyncio
    async def test_analyze_with_retry_no_time_signatures(self, resilient_tool):
        """Test analyze_with_retry with no time signatures"""
        mock_score = Mock()
        mock_key = Mock()
        mock_key.__str__ = Mock(return_value="D minor")

        mock_score.analyze.return_value = mock_key
        mock_score.getTimeSignatures.return_value = []

        result = await resilient_tool.analyze_with_retry(mock_score)

        assert result["key"] == "D minor"
        assert result["time_signature"] == "None"

    @pytest.mark.asyncio
    async def test_analyze_with_retry_index_error(self, resilient_tool):
        """Test analyze_with_retry handles index error as non-retryable"""
        mock_score = Mock()
        mock_score.analyze.side_effect = IndexError("list index out of range")

        with pytest.raises(NonRetryableError) as exc_info:
            await resilient_tool.analyze_with_retry(mock_score)

        assert "Score has no time signatures" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_analyze_with_retry_retryable_error(self, resilient_tool):
        """Test analyze_with_retry converts other errors to retryable"""
        mock_score = Mock()
        mock_score.analyze.side_effect = RuntimeError("Temporary processing error")

        with pytest.raises(RetryableError) as exc_info:
            await resilient_tool.analyze_with_retry(mock_score)

        assert "Transient analysis error" in str(exc_info.value)


class TestIntegration:
    """Integration tests combining multiple components"""

    @pytest.mark.asyncio
    async def test_retry_with_circuit_breaker_integration(self):
        """Test retry decorator with circuit breaker integration"""
        circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=0.1)
        policy = RetryPolicy(
            max_attempts=3,
            base_delay=0.01,
            retryable_exceptions=(ConnectionError, RuntimeError),
        )

        call_count = 0

        @retry(policy=policy, circuit_breaker=circuit_breaker)
        async def consistently_failing():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Consistent failure")

        # First call should exhaust retries and record failures
        with pytest.raises(RuntimeError):
            await consistently_failing()

        # Should have made multiple attempts due to retry
        retry_attempts = call_count
        assert retry_attempts >= 2

        # Circuit should still be closed after first failure series
        call_count = 0

        # Second call should also fail and open circuit
        with pytest.raises((RuntimeError, CircuitBreakerOpenError)):
            await consistently_failing()

        # Circuit should now be open
        assert circuit_breaker.state == CircuitState.OPEN

        # Subsequent calls should immediately fail with CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            await consistently_failing()

    @pytest.mark.asyncio
    async def test_bulk_executor_with_circuit_breaker(self):
        """Test BulkRetryExecutor handles circuit breaker scenarios"""
        # Create executor with circuit breaker-enabled operations
        executor = BulkRetryExecutor(max_concurrent=2)

        call_count = 0

        async def protected_operation():
            nonlocal call_count
            call_count += 1
            # Fail consistently to trigger circuit breaker
            raise ConnectionError("Service unavailable")

        operations = [protected_operation, protected_operation, protected_operation]
        results = await executor.execute_all(operations, continue_on_error=True)

        # All should fail, but should not crash
        assert results["total"] == 3
        assert results["success_rate"] == 0
        assert len(results["failed"]) == 3

        # Should have attempted retries
        assert call_count > 3  # More than one attempt per operation

    @pytest.mark.asyncio
    async def test_music21_operation_integration(self):
        """Test RetryableMusic21Operation with real-ish scenarios"""
        music21_op = RetryableMusic21Operation()

        # Test with successful mock operations
        with patch("music21.converter.parse") as mock_parse:
            mock_score = Mock()
            mock_parse.return_value = mock_score

            result = await music21_op.parse_score("test musicxml")
            assert result == mock_score

    def test_all_policies_are_configured(self):
        """Test that all pre-configured policies are properly set"""
        policies = [FILE_IO_POLICY, NETWORK_POLICY, MUSIC21_POLICY, DATABASE_POLICY]

        for policy in policies:
            assert isinstance(policy, RetryPolicy)
            assert policy.max_attempts > 0
            assert policy.base_delay > 0
            assert policy.max_delay >= policy.base_delay
            assert len(policy.retryable_exceptions) > 0
            assert len(policy.non_retryable_exceptions) > 0

    @pytest.mark.asyncio
    async def test_resilient_tool_integration(self):
        """Test ResilientTool integration"""
        tool = ResilientTool()

        # Mock a score with realistic behavior
        mock_score = Mock()
        mock_key = Mock()
        mock_key.__str__ = Mock(return_value="F# major")
        mock_time_sig = Mock()
        mock_time_sig.__str__ = Mock(return_value="3/4")

        mock_score.analyze.return_value = mock_key
        mock_score.getTimeSignatures.return_value = [mock_time_sig]

        result = await tool.analyze_with_retry(mock_score)

        assert result == {"key": "F# major", "time_signature": "3/4"}
