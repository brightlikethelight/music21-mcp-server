"""
Production resilience features for Music21 MCP Server
Implements circuit breakers, rate limiting, resource pooling, and self-healing
"""

import asyncio
import functools
import gc
import logging
import os
import queue
import resource
import signal
import threading
import time
import traceback
import weakref
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import psutil

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""

    failure_threshold: int = 5
    recovery_timeout: int = 60
    expected_exception: type = Exception
    success_threshold: int = 2
    max_failures_per_minute: int = 20


class CircuitBreaker:
    """Circuit breaker pattern implementation"""

    def __init__(
        self, name: str, config: Optional[CircuitBreakerConfig] = None
    ) -> None:
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.next_attempt_time: Optional[float] = None
        self.failure_history: deque = deque(maxlen=100)
        self._lock = threading.Lock()

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator for protecting functions"""

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if asyncio.iscoroutinefunction(func):
                return await self._async_call(func, *args, **kwargs)
            else:
                return await asyncio.to_thread(self._sync_call, func, *args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            return self._sync_call(func, *args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    async def _async_call(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Handle async function calls"""
        if not self._can_attempt():
            raise Exception(f"Circuit breaker {self.name} is OPEN")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            if isinstance(e, self.config.expected_exception):
                self._on_failure(e)
            raise

    def _sync_call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Handle sync function calls"""
        if not self._can_attempt():
            raise Exception(f"Circuit breaker {self.name} is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            if isinstance(e, self.config.expected_exception):
                self._on_failure(e)
            raise

    def _can_attempt(self) -> bool:
        """Check if request can be attempted"""
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True

            if self.state == CircuitState.OPEN:
                if self.next_attempt_time and time.time() >= self.next_attempt_time:
                    self.state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit breaker {self.name} entering HALF_OPEN state")
                    return True
                return False

            # HALF_OPEN state
            return True

    def _on_success(self) -> None:
        """Handle successful call"""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info(
                        f"Circuit breaker {self.name} recovered to CLOSED state"
                    )
            elif self.state == CircuitState.CLOSED:
                self.failure_count = max(0, self.failure_count - 1)

    def _on_failure(self, error: Exception) -> None:
        """Handle failed call"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            self.failure_history.append({"time": datetime.now(), "error": str(error)})

            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.next_attempt_time = time.time() + self.config.recovery_timeout
                logger.warning(
                    f"Circuit breaker {self.name} failed in HALF_OPEN, returning to OPEN"
                )

            elif self.state == CircuitState.CLOSED:
                # Check failure rate
                recent_failures = sum(
                    1
                    for f in self.failure_history
                    if (datetime.now() - f["time"]).total_seconds() < 60
                )

                if (
                    self.failure_count >= self.config.failure_threshold
                    or recent_failures >= self.config.max_failures_per_minute
                ):
                    self.state = CircuitState.OPEN
                    self.next_attempt_time = time.time() + self.config.recovery_timeout
                    logger.warning(
                        f"Circuit breaker {self.name} opened due to failures"
                    )


class ResourcePool:
    """Generic resource pool with health checking"""

    def __init__(
        self,
        name: str,
        factory: Callable[[], Any],
        max_size: int = 10,
        min_size: int = 2,
        health_check: Optional[Callable[[Any], bool]] = None,
        max_idle_time: int = 300,
    ) -> None:
        self.name = name
        self.factory = factory
        self.max_size = max_size
        self.min_size = min_size
        self.health_check = health_check
        self.max_idle_time = max_idle_time

        self._pool: queue.Queue = queue.Queue(maxsize=max_size)
        self._all_resources: weakref.WeakSet = weakref.WeakSet()
        self._size = 0
        self._lock = threading.Lock()
        self._shutdown = False

        # Pre-populate pool
        for _ in range(min_size):
            self._create_resource()

    def _create_resource(self) -> None:
        """Create a new resource"""
        try:
            resource = self.factory()
            self._all_resources.add(resource)
            self._pool.put(
                {"resource": resource, "created": time.time(), "last_used": time.time()}
            )
            self._size += 1
            logger.debug(f"Created resource for pool {self.name}, size: {self._size}")
        except Exception as e:
            logger.error(f"Failed to create resource for pool {self.name}: {e}")

    @contextmanager
    def acquire(self, timeout: float = 30.0) -> Any:
        """Acquire a resource from the pool"""
        if self._shutdown:
            raise RuntimeError(f"Pool {self.name} is shut down")

        start_time = time.time()
        resource_wrapper = None

        while time.time() - start_time < timeout:
            try:
                # Try to get from pool
                resource_wrapper = self._pool.get(timeout=1.0)

                # Health check
                if self.health_check:
                    try:
                        if not self.health_check(resource_wrapper["resource"]):
                            logger.warning(
                                f"Resource failed health check in pool {self.name}"
                            )
                            self._size -= 1
                            continue
                    except Exception as e:
                        logger.error(f"Health check error in pool {self.name}: {e}")
                        self._size -= 1
                        continue

                # Check idle time
                if time.time() - resource_wrapper["last_used"] > self.max_idle_time:
                    logger.debug(f"Resource expired in pool {self.name}")
                    self._size -= 1
                    continue

                # Resource is good
                resource_wrapper["last_used"] = time.time()
                break

            except queue.Empty:
                # Create new resource if under limit
                with self._lock:
                    if self._size < self.max_size:
                        self._create_resource()

        if not resource_wrapper:
            raise TimeoutError(f"Could not acquire resource from pool {self.name}")

        try:
            yield resource_wrapper["resource"]
        finally:
            # Return to pool
            if not self._shutdown:
                resource_wrapper["last_used"] = time.time()
                try:
                    self._pool.put(resource_wrapper, timeout=1.0)
                except queue.Full:
                    self._size -= 1

    def shutdown(self) -> None:
        """Shutdown the pool"""
        self._shutdown = True

        # Drain pool
        while not self._pool.empty():
            try:
                self._pool.get_nowait()
                self._size -= 1
            except queue.Empty:
                break

        logger.info(f"Pool {self.name} shut down")


class RateLimiter:
    """Token bucket rate limiter"""

    def __init__(self, rate: float, burst: int):
        self.rate = rate  # Tokens per second
        self.burst = burst  # Max tokens
        self.tokens: float = burst
        self.last_update = time.time()
        self._lock = threading.Lock()

    def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens"""
        with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            self.last_update = now

            # Add tokens based on elapsed time
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    async def acquire_async(
        self, tokens: int = 1, timeout: Optional[float] = None
    ) -> bool:
        """Async version with optional wait"""
        start_time = time.time()

        while True:
            if self.acquire(tokens):
                return True

            if timeout and time.time() - start_time >= timeout:
                return False

            # Calculate wait time
            needed = tokens - self.tokens
            wait_time = needed / self.rate

            if timeout:
                wait_time = min(wait_time, timeout - (time.time() - start_time))

            await asyncio.sleep(min(wait_time, 0.1))


class MemoryGuard:
    """Monitor and control memory usage"""

    def __init__(
        self,
        soft_limit_mb: int = 2048,
        hard_limit_mb: int = 4096,
        check_interval: int = 5,
    ):
        self.soft_limit_mb = soft_limit_mb
        self.hard_limit_mb = hard_limit_mb
        self.check_interval = check_interval
        self.process = psutil.Process()
        self._monitoring = False
        self._callbacks: List[Callable[[float], None]] = []

    def add_callback(self, callback: Callable[[float], None]) -> None:
        """Add callback for memory warnings"""
        self._callbacks.append(callback)

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return float(self.process.memory_info().rss / 1024 / 1024)

    async def start_monitoring(self) -> None:
        """Start memory monitoring"""
        self._monitoring = True

        while self._monitoring:
            try:
                current_mb = self.get_memory_usage()

                if current_mb > self.hard_limit_mb:
                    logger.critical(
                        f"Memory usage critical: {current_mb:.0f}MB > {self.hard_limit_mb}MB"
                    )
                    await self._emergency_cleanup()

                elif current_mb > self.soft_limit_mb:
                    logger.warning(
                        f"Memory usage high: {current_mb:.0f}MB > {self.soft_limit_mb}MB"
                    )
                    for callback in self._callbacks:
                        try:
                            callback(current_mb)
                        except Exception as e:
                            logger.error(f"Memory callback error: {e}")

                await asyncio.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                await asyncio.sleep(self.check_interval)

    async def _emergency_cleanup(self) -> None:
        """Emergency memory cleanup"""
        logger.warning("Performing emergency memory cleanup")

        # Force garbage collection
        gc.collect(2)

        # Clear caches (implement cache clearing in actual server)

        # If still over limit, may need to reject new requests
        current_mb = self.get_memory_usage()
        if current_mb > self.hard_limit_mb:
            logger.critical("Memory still critical after cleanup")

    def stop_monitoring(self) -> None:
        """Stop memory monitoring"""
        self._monitoring = False


class GracefulShutdown:
    """Handle graceful shutdown with timeout"""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.shutdown_event = asyncio.Event()
        self.tasks: Set[asyncio.Task] = set()
        self._original_handlers: Dict[int, Any] = {}

    def __enter__(self) -> "GracefulShutdown":
        """Install signal handlers"""
        for sig in [signal.SIGTERM, signal.SIGINT]:
            self._original_handlers[sig] = signal.signal(sig, self._signal_handler)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Restore original handlers"""
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signal"""
        logger.info(f"Received signal {signum}, starting graceful shutdown")
        self.shutdown_event.set()

    def track_task(self, task: asyncio.Task) -> None:
        """Track a task for graceful shutdown"""
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal"""
        await self.shutdown_event.wait()

        logger.info("Starting graceful shutdown sequence")

        # Cancel all tracked tasks
        for task in self.tasks:
            task.cancel()

        # Wait for tasks to complete with timeout
        if self.tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.tasks, return_exceptions=True),
                    timeout=self.timeout,
                )
                logger.info("All tasks completed gracefully")
            except asyncio.TimeoutError:
                logger.warning(f"Shutdown timeout after {self.timeout}s, forcing exit")


class RequestContext:
    """Context for request tracking and resource management"""

    def __init__(self, request_id: str, timeout: int = 30):
        self.request_id = request_id
        self.timeout = timeout
        self.start_time = time.time()
        self.resources: List[Any] = []
        self.metadata: Dict[str, Any] = {}

    def add_resource(self, resource: Any) -> None:
        """Track a resource for cleanup"""
        self.resources.append(resource)

    def set_metadata(self, key: str, value: Any) -> None:
        """Set request metadata"""
        self.metadata[key] = value

    def get_elapsed_time(self) -> float:
        """Get elapsed time"""
        return time.time() - self.start_time

    def is_expired(self) -> bool:
        """Check if request has expired"""
        return self.get_elapsed_time() > self.timeout

    async def cleanup(self) -> None:
        """Clean up request resources"""
        for resource in self.resources:
            try:
                if hasattr(resource, "close"):
                    if asyncio.iscoroutinefunction(resource.close):
                        await resource.close()
                    else:
                        resource.close()
                elif hasattr(resource, "cleanup"):
                    if asyncio.iscoroutinefunction(resource.cleanup):
                        await resource.cleanup()
                    else:
                        resource.cleanup()
            except Exception as e:
                logger.error(
                    f"Error cleaning up resource in request {self.request_id}: {e}"
                )


class HealthCheck:
    """System health checking"""

    def __init__(self) -> None:
        self.checks: Dict[str, Dict[str, Any]] = {}
        self.last_check_time: Dict[str, float] = {}
        self.check_results: Dict[str, Dict[str, Any]] = {}

    def register_check(
        self, name: str, check_func: Callable, interval: int = 30
    ) -> None:
        """Register a health check"""
        self.checks[name] = {"func": check_func, "interval": interval}

    async def run_checks(self) -> Dict[str, Any]:
        """Run all due health checks"""
        results = {}
        now = time.time()

        for name, check in self.checks.items():
            last_check = self.last_check_time.get(name, 0)

            if now - last_check >= check["interval"]:
                try:
                    if asyncio.iscoroutinefunction(check["func"]):
                        result = await check["func"]()
                    else:
                        result = await asyncio.to_thread(check["func"])

                    results[name] = {
                        "status": "healthy" if result else "unhealthy",
                        "timestamp": datetime.now().isoformat(),
                    }

                except Exception as e:
                    results[name] = {
                        "status": "error",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }

                self.last_check_time[name] = now
                result = results.get(name)
                if result is not None:
                    self.check_results[name] = result

        return results

    def get_overall_health(self) -> str:
        """Get overall system health"""
        if not self.check_results:
            return "unknown"

        statuses = [r.get("status", "unknown") for r in self.check_results.values()]

        if "error" in statuses or "unhealthy" in statuses:
            return "unhealthy"
        elif all(s == "healthy" for s in statuses):
            return "healthy"
        else:
            return "degraded"


class AutoRecovery:
    """Automatic recovery mechanisms"""

    def __init__(self) -> None:
        self.recovery_strategies: Dict[type, Dict[str, Any]] = {}
        self.recovery_history: deque = deque(maxlen=100)

    def register_strategy(
        self, error_type: type, strategy: Callable, max_retries: int = 3
    ) -> None:
        """Register a recovery strategy for an error type"""
        self.recovery_strategies[error_type] = {
            "strategy": strategy,
            "max_retries": max_retries,
        }

    async def attempt_recovery(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Attempt to recover from an error"""
        error_type = type(error)

        # Find matching strategy
        strategy_info = None
        for registered_type, info in self.recovery_strategies.items():
            if issubclass(error_type, registered_type):
                strategy_info = info
                break

        if not strategy_info:
            return False

        # Check retry limit
        recent_recoveries = [
            r
            for r in self.recovery_history
            if r["error_type"] == error_type
            and (datetime.now() - r["timestamp"]).total_seconds() < 300
        ]

        if len(recent_recoveries) >= strategy_info["max_retries"]:
            logger.warning(f"Recovery retry limit reached for {error_type.__name__}")
            return False

        # Attempt recovery
        try:
            logger.info(f"Attempting recovery for {error_type.__name__}")

            if asyncio.iscoroutinefunction(strategy_info["strategy"]):
                success = await strategy_info["strategy"](error, context)
            else:
                success = await asyncio.to_thread(
                    strategy_info["strategy"], error, context
                )

            self.recovery_history.append(
                {
                    "error_type": error_type,
                    "timestamp": datetime.now(),
                    "success": success,
                }
            )

            return bool(success)

        except Exception as e:
            logger.error(f"Recovery strategy failed: {e}")
            return False


# Global instances for easy access
circuit_breakers = {}
resource_pools = {}
rate_limiters = {}
memory_guard = MemoryGuard()
health_checker = HealthCheck()
auto_recovery = AutoRecovery()


def get_circuit_breaker(
    name: str, config: Optional[CircuitBreakerConfig] = None
) -> CircuitBreaker:
    """Get or create a circuit breaker"""
    if name not in circuit_breakers:
        circuit_breakers[name] = CircuitBreaker(name, config)
    return circuit_breakers[name]


def get_resource_pool(name: str, **kwargs: Any) -> ResourcePool:
    """Get or create a resource pool"""
    if name not in resource_pools:
        factory = kwargs.pop("factory")
        resource_pools[name] = ResourcePool(name, factory, **kwargs)
    return resource_pools[name]


def get_rate_limiter(name: str, rate: float, burst: int) -> RateLimiter:
    """Get or create a rate limiter"""
    if name not in rate_limiters:
        rate_limiters[name] = RateLimiter(rate, burst)
    return rate_limiters[name]


# Decorators for easy use
def with_circuit_breaker(name: str, **config_kwargs: Any) -> Callable:
    """Decorator to add circuit breaker to a function"""

    def decorator(func: Callable) -> Callable:
        config = CircuitBreakerConfig(**config_kwargs)
        breaker = get_circuit_breaker(name, config)
        return breaker(func)

    return decorator


def with_rate_limit(name: str, rate: float, burst: int) -> Callable:
    """Decorator to add rate limiting to a function"""

    def decorator(func: Callable) -> Callable:
        limiter = get_rate_limiter(name, rate, burst)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not await limiter.acquire_async(timeout=5.0):
                raise Exception(f"Rate limit exceeded for {name}")
            return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not limiter.acquire():
                raise Exception(f"Rate limit exceeded for {name}")
            return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def with_timeout(seconds: int) -> Callable:
    """Decorator to add timeout to async functions"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)

        return wrapper

    return decorator


# Recovery strategies
async def recover_from_connection_error(
    error: Exception, context: Dict[str, Any]
) -> bool:
    """Recover from connection errors"""
    logger.info("Attempting to recover from connection error")

    # Clear connection pools
    for pool in resource_pools.values():
        pool.shutdown()
    resource_pools.clear()

    # Wait a bit
    await asyncio.sleep(5)

    # Reinitialize connections
    return True


async def recover_from_memory_error(error: Exception, context: Dict[str, Any]) -> bool:
    """Recover from memory errors"""
    logger.info("Attempting to recover from memory error")

    # Force aggressive garbage collection
    gc.collect(2)

    # Clear all caches
    for attr_name in dir(context.get("server", {})):
        attr = getattr(context.get("server", {}), attr_name)
        if hasattr(attr, "clear") and callable(attr.clear):
            try:
                attr.clear()
            except:
                pass

    # Check if recovery worked
    memory_mb = memory_guard.get_memory_usage()
    return memory_mb < memory_guard.soft_limit_mb


# Register default recovery strategies
auto_recovery.register_strategy(ConnectionError, recover_from_connection_error)
auto_recovery.register_strategy(MemoryError, recover_from_memory_error)
auto_recovery.register_strategy(
    OSError, recover_from_connection_error
)  # File descriptor exhaustion


# Health check functions
def check_memory_health() -> bool:
    """Check if memory usage is healthy"""
    memory_mb = memory_guard.get_memory_usage()
    return memory_mb < memory_guard.soft_limit_mb


def check_file_descriptors() -> bool:
    """Check if file descriptors are available"""
    try:
        # Try to open a file
        with open("/dev/null", "r") as f:
            pass
        return True
    except OSError:
        return False


async def check_response_time() -> bool:
    """Check if system is responsive"""
    start = time.time()
    await asyncio.sleep(0.001)  # Minimal async operation
    return time.time() - start < 0.1  # Should be very fast


# Register default health checks
health_checker.register_check("memory", check_memory_health, interval=30)
health_checker.register_check("file_descriptors", check_file_descriptors, interval=60)
health_checker.register_check("response_time", check_response_time, interval=10)
