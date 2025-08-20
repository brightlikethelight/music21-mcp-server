"""
Retry Logic and Resilience Patterns for Music21 MCP Server

Implements exponential backoff, circuit breaker, and intelligent retry policies
for handling transient failures in production environments.
"""

import asyncio
import functools
import logging
import random
import time
from collections.abc import Callable
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class RetryableError(Exception):
    """Base class for errors that should trigger retries"""

    pass


class NonRetryableError(Exception):
    """Base class for errors that should NOT trigger retries"""

    pass


class CircuitState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class RetryPolicy:
    """Defines retry behavior for different operation types"""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: tuple[type[Exception], ...] = (
            RetryableError,
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
            OSError,
        ),
        non_retryable_exceptions: tuple[type[Exception], ...] = (
            NonRetryableError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
        ),
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions
        self.non_retryable_exceptions = non_retryable_exceptions

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if an operation should be retried"""
        if attempt >= self.max_attempts:
            return False

        # Don't retry non-retryable exceptions
        if isinstance(exception, self.non_retryable_exceptions):
            return False

        # Retry known retryable exceptions
        return isinstance(exception, self.retryable_exceptions)

    def get_delay(self, attempt: int) -> float:
        """Calculate delay before next retry with exponential backoff"""
        delay = min(self.base_delay * (self.exponential_base**attempt), self.max_delay)

        if self.jitter:
            # Add random jitter to prevent thundering herd
            delay = delay * (0.5 + random.random())

        return delay


class CircuitBreaker:
    """Circuit breaker to prevent cascading failures"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type[Exception] = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time: float | None = None
        self.state = CircuitState.CLOSED

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN. Failures: {self.failure_count}"
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e

    async def async_call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN. Failures: {self.failure_count}"
                )

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        return (
            self.last_failure_time
            and time.time() - self.last_failure_time >= self.recovery_timeout
        )

    def _on_success(self):
        """Reset circuit breaker on successful call"""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = None

    def _on_failure(self):
        """Record failure and potentially open circuit"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""

    pass


def retry(
    policy: RetryPolicy | None = None,
    circuit_breaker: CircuitBreaker | None = None,
):
    """Decorator for adding retry logic to functions"""
    if policy is None:
        policy = RetryPolicy()

    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(policy.max_attempts):
                try:
                    # Use circuit breaker if provided
                    if circuit_breaker:
                        return await circuit_breaker.async_call(func, *args, **kwargs)
                    return await func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    if not policy.should_retry(e, attempt + 1):
                        logger.error(f"Non-retryable error in {func.__name__}: {e}")
                        raise

                    delay = policy.get_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1}/{policy.max_attempts} failed "
                        f"for {func.__name__}: {e}. Retrying in {delay:.1f}s..."
                    )

                    await asyncio.sleep(delay)

            logger.error(
                f"All {policy.max_attempts} attempts failed for {func.__name__}"
            )
            raise last_exception

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(policy.max_attempts):
                try:
                    # Use circuit breaker if provided
                    if circuit_breaker:
                        return circuit_breaker.call(func, *args, **kwargs)
                    return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    if not policy.should_retry(e, attempt + 1):
                        logger.error(f"Non-retryable error in {func.__name__}: {e}")
                        raise

                    delay = policy.get_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1}/{policy.max_attempts} failed "
                        f"for {func.__name__}: {e}. Retrying in {delay:.1f}s..."
                    )

                    time.sleep(delay)

            logger.error(
                f"All {policy.max_attempts} attempts failed for {func.__name__}"
            )
            raise last_exception

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# Pre-configured retry policies for different operation types

# For file I/O operations
FILE_IO_POLICY = RetryPolicy(
    max_attempts=3,
    base_delay=0.5,
    max_delay=5.0,
    retryable_exceptions=(
        OSError,
        IOError,
        PermissionError,
        TimeoutError,
    ),
)

# For network operations (API calls, HTTP requests)
NETWORK_POLICY = RetryPolicy(
    max_attempts=5,
    base_delay=1.0,
    max_delay=30.0,
    retryable_exceptions=(
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError,
        OSError,
    ),
)

# For music21 operations that might fail transiently
MUSIC21_POLICY = RetryPolicy(
    max_attempts=3,
    base_delay=0.2,
    max_delay=2.0,
    retryable_exceptions=(
        RetryableError,
        RuntimeError,
        TimeoutError,
    ),
)

# For database operations
DATABASE_POLICY = RetryPolicy(
    max_attempts=3,
    base_delay=0.1,
    max_delay=1.0,
    exponential_base=2.0,
    retryable_exceptions=(
        ConnectionError,
        TimeoutError,
    ),
)


class RetryableMusic21Operation:
    """Wrapper for music21 operations with built-in retry logic"""

    def __init__(self, policy: RetryPolicy | None = None):
        self.policy = policy or MUSIC21_POLICY
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=10,
            recovery_timeout=120.0,
        )

    @retry(policy=MUSIC21_POLICY)
    async def parse_score(self, data: str | bytes) -> Any:
        """Parse score with retry logic"""
        from music21 import converter

        try:
            # This might fail transiently due to memory or processing issues
            return converter.parse(data)
        except Exception as e:
            if "memory" in str(e).lower() or "timeout" in str(e).lower():
                raise RetryableError(f"Transient error parsing score: {e}")
            raise

    @retry(policy=FILE_IO_POLICY)
    async def write_file(self, path: str, content: str | bytes) -> None:
        """Write file with retry logic"""
        import aiofiles

        async with aiofiles.open(
            path, "wb" if isinstance(content, bytes) else "w"
        ) as f:
            await f.write(content)

    @retry(policy=NETWORK_POLICY)
    async def fetch_corpus(self, corpus_path: str) -> Any:
        """Fetch corpus with retry logic"""
        from music21 import corpus

        try:
            return corpus.parse(corpus_path)
        except Exception as e:
            if "network" in str(e).lower() or "connection" in str(e).lower():
                raise RetryableError(f"Network error fetching corpus: {e}")
            raise


class BulkRetryExecutor:
    """Execute multiple operations with retry logic and partial failure handling"""

    def __init__(
        self,
        policy: RetryPolicy | None = None,
        max_concurrent: int = 10,
    ):
        self.policy = policy or RetryPolicy()
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def execute_all(
        self,
        operations: list[Callable],
        continue_on_error: bool = True,
    ) -> dict[str, Any]:
        """Execute all operations with retry logic"""
        results = {
            "successful": [],
            "failed": [],
            "total": len(operations),
        }

        async def execute_with_retry(op_id: int, operation: Callable):
            async with self.semaphore:
                try:
                    # Apply retry logic
                    @retry(policy=self.policy)
                    async def wrapped():
                        if asyncio.iscoroutinefunction(operation):
                            return await operation()
                        return operation()

                    result = await wrapped()
                    results["successful"].append(
                        {
                            "id": op_id,
                            "result": result,
                        }
                    )

                except Exception as e:
                    logger.error(f"Operation {op_id} failed after retries: {e}")
                    results["failed"].append(
                        {
                            "id": op_id,
                            "error": str(e),
                        }
                    )

                    if not continue_on_error:
                        raise

        # Execute all operations concurrently
        tasks = [execute_with_retry(i, op) for i, op in enumerate(operations)]

        await asyncio.gather(*tasks, return_exceptions=True)

        # Calculate success rate
        results["success_rate"] = (
            len(results["successful"]) / results["total"] if results["total"] > 0 else 0
        )

        return results


# Example usage in tools
class ResilientTool:
    """Example of integrating retry logic into a tool"""

    def __init__(self):
        self.retry_executor = RetryableMusic21Operation()
        self.bulk_executor = BulkRetryExecutor()

    @retry(policy=MUSIC21_POLICY)
    async def analyze_with_retry(self, score) -> dict[str, Any]:
        """Analyze score with automatic retry on transient failures"""
        try:
            # These operations might fail transiently
            key = score.analyze("key")
            time_sig = (
                score.getTimeSignatures()[0] if score.getTimeSignatures() else None
            )

            return {
                "key": str(key),
                "time_signature": str(time_sig),
            }
        except Exception as e:
            # Determine if error is retryable
            if "list index out of range" in str(e):
                raise NonRetryableError(f"Score has no time signatures: {e}")
            raise RetryableError(f"Transient analysis error: {e}")
