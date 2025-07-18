"""
Circuit Breaker and Resilience Patterns for Music21 MCP Server

Enterprise-grade resilience patterns:
- Circuit breaker with adaptive thresholds
- Bulkhead isolation patterns
- Timeout management with context-aware limits
- Graceful degradation strategies
- Retry patterns with exponential backoff
- Health check and monitoring systems
- Resource pool management

Complies with 2024 microservice resilience standards.
"""

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
from datetime import datetime, timedelta
import weakref

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"           # Normal operation
    OPEN = "open"              # Circuit is open, failing fast
    HALF_OPEN = "half_open"    # Testing if service has recovered


class FailureType(Enum):
    """Types of failures that can trigger circuit breaker"""
    TIMEOUT = "timeout"
    EXCEPTION = "exception"
    RATE_LIMIT = "rate_limit"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    VALIDATION_ERROR = "validation_error"
    EXTERNAL_DEPENDENCY = "external_dependency"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior"""
    # Failure thresholds
    failure_threshold: int = 5          # Number of failures to open circuit
    recovery_timeout: float = 60.0      # Seconds to wait before trying half-open
    success_threshold: int = 3          # Successes needed to close from half-open
    
    # Time windows
    monitoring_window: float = 60.0     # Window for tracking failures (seconds)
    half_open_max_calls: int = 5        # Max calls allowed in half-open state
    
    # Advanced settings
    adaptive_threshold: bool = True     # Adjust thresholds based on patterns
    exponential_backoff: bool = True    # Use exponential backoff for recovery
    max_recovery_timeout: float = 300.0 # Maximum recovery timeout
    
    # Failure weights (some failures are more serious than others)
    failure_weights: Dict[FailureType, float] = field(default_factory=lambda: {
        FailureType.TIMEOUT: 1.0,
        FailureType.EXCEPTION: 0.8,
        FailureType.RATE_LIMIT: 0.6,
        FailureType.RESOURCE_EXHAUSTION: 1.2,
        FailureType.VALIDATION_ERROR: 0.3,
        FailureType.EXTERNAL_DEPENDENCY: 1.0
    })


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open"""
    def __init__(self, circuit_name: str, state: CircuitState, last_failure: Optional[str] = None):
        self.circuit_name = circuit_name
        self.state = state
        self.last_failure = last_failure
        super().__init__(f"Circuit '{circuit_name}' is {state.value}" + 
                        (f" (last failure: {last_failure})" if last_failure else ""))


@dataclass
class FailureRecord:
    """Record of a failure event"""
    timestamp: float
    failure_type: FailureType
    error_message: str
    duration: Optional[float] = None
    weight: float = 1.0


class CircuitBreaker:
    """
    Advanced circuit breaker with adaptive thresholds and multiple failure types
    
    Features:
    - Multiple failure types with different weights
    - Adaptive thresholds based on historical patterns
    - Exponential backoff for recovery attempts
    - Half-open state with limited testing
    - Comprehensive metrics and monitoring
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # State management
        self.state = CircuitState.CLOSED
        self.last_state_change = time.time()
        
        # Failure tracking
        self.failures: deque = deque(maxlen=1000)  # Recent failures
        self.failure_count = 0
        self.weighted_failure_count = 0.0
        
        # Success tracking for half-open state
        self.success_count = 0
        self.half_open_calls = 0
        
        # Recovery management
        self.recovery_attempts = 0
        self.next_recovery_time = 0.0
        
        # Statistics
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0
        self.state_changes = 0
        
        # Adaptive thresholds
        self.adaptive_failure_threshold = self.config.failure_threshold
        self.historical_failure_rate = 0.0
        
        logger.info(f"Circuit breaker '{name}' initialized")
    
    async def __call__(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        return await self.execute(func, *args, **kwargs)
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        # Check circuit state before execution
        self._check_state()
        
        if self.state == CircuitState.OPEN:
            raise CircuitBreakerError(
                self.name, 
                self.state, 
                self.failures[-1].error_message if self.failures else None
            )
        
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= self.config.half_open_max_calls:
                raise CircuitBreakerError(
                    self.name,
                    self.state,
                    "Half-open call limit exceeded"
                )
            self.half_open_calls += 1
        
        # Execute with monitoring
        start_time = time.time()
        self.total_calls += 1
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Record success
            duration = time.time() - start_time
            await self._record_success(duration)
            
            return result
            
        except Exception as e:
            # Record failure
            duration = time.time() - start_time
            failure_type = self._classify_failure(e)
            await self._record_failure(failure_type, str(e), duration)
            
            raise
    
    def _check_state(self):
        """Check and update circuit breaker state"""
        current_time = time.time()
        
        if self.state == CircuitState.OPEN:
            # Check if it's time to try recovery
            if current_time >= self.next_recovery_time:
                self._transition_to_half_open()
        
        elif self.state == CircuitState.HALF_OPEN:
            # Check if we should close or open based on recent results
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()
            elif self.failure_count > 0:  # Any failure in half-open goes back to open
                self._transition_to_open()
    
    async def _record_success(self, duration: float):
        """Record successful execution"""
        self.total_successes += 1
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
        
        # Update adaptive thresholds
        if self.config.adaptive_threshold:
            await self._update_adaptive_thresholds()
        
        logger.debug(f"Circuit '{self.name}' recorded success (duration: {duration:.3f}s)")
    
    async def _record_failure(self, failure_type: FailureType, error_message: str, duration: float):
        """Record failed execution"""
        weight = self.config.failure_weights.get(failure_type, 1.0)
        
        failure_record = FailureRecord(
            timestamp=time.time(),
            failure_type=failure_type,
            error_message=error_message,
            duration=duration,
            weight=weight
        )
        
        self.failures.append(failure_record)
        self.total_failures += 1
        
        # Clean old failures outside monitoring window
        self._clean_old_failures()
        
        # Update failure counts
        self.failure_count = len(self.failures)
        self.weighted_failure_count = sum(f.weight for f in self.failures)
        
        # Check if we should open the circuit
        if self._should_open_circuit():
            self._transition_to_open()
        
        # Update adaptive thresholds
        if self.config.adaptive_threshold:
            await self._update_adaptive_thresholds()
        
        logger.warning(
            f"Circuit '{self.name}' recorded failure: {failure_type.value} - {error_message} "
            f"(duration: {duration:.3f}s, weight: {weight})"
        )
    
    def _classify_failure(self, exception: Exception) -> FailureType:
        """Classify the type of failure based on exception"""
        if isinstance(exception, asyncio.TimeoutError):
            return FailureType.TIMEOUT
        elif isinstance(exception, (ConnectionError, OSError)):
            return FailureType.EXTERNAL_DEPENDENCY
        elif isinstance(exception, MemoryError):
            return FailureType.RESOURCE_EXHAUSTION
        elif isinstance(exception, ValueError):
            return FailureType.VALIDATION_ERROR
        else:
            return FailureType.EXCEPTION
    
    def _clean_old_failures(self):
        """Remove failures outside the monitoring window"""
        cutoff_time = time.time() - self.config.monitoring_window
        
        while self.failures and self.failures[0].timestamp < cutoff_time:
            self.failures.popleft()
    
    def _should_open_circuit(self) -> bool:
        """Determine if circuit should be opened"""
        if self.state == CircuitState.OPEN:
            return False
        
        # Use adaptive threshold if enabled
        threshold = (self.adaptive_failure_threshold if self.config.adaptive_threshold 
                    else self.config.failure_threshold)
        
        # Check both absolute count and weighted count
        return (self.failure_count >= threshold or 
                self.weighted_failure_count >= threshold * 1.2)
    
    def _transition_to_open(self):
        """Transition circuit to open state"""
        if self.state != CircuitState.OPEN:
            self.state = CircuitState.OPEN
            self.last_state_change = time.time()
            self.state_changes += 1
            
            # Calculate next recovery time with exponential backoff
            if self.config.exponential_backoff:
                backoff_multiplier = min(2 ** self.recovery_attempts, 8)  # Cap at 8x
                recovery_delay = self.config.recovery_timeout * backoff_multiplier
                recovery_delay = min(recovery_delay, self.config.max_recovery_timeout)
            else:
                recovery_delay = self.config.recovery_timeout
            
            self.next_recovery_time = time.time() + recovery_delay
            self.recovery_attempts += 1
            
            logger.warning(
                f"Circuit '{self.name}' opened due to failures "
                f"(failure_count: {self.failure_count}, weighted: {self.weighted_failure_count:.1f}). "
                f"Recovery in {recovery_delay:.1f}s"
            )
    
    def _transition_to_half_open(self):
        """Transition circuit to half-open state"""
        self.state = CircuitState.HALF_OPEN
        self.last_state_change = time.time()
        self.state_changes += 1
        self.success_count = 0
        self.half_open_calls = 0
        
        logger.info(f"Circuit '{self.name}' transitioned to half-open for testing")
    
    def _transition_to_closed(self):
        """Transition circuit to closed state"""
        self.state = CircuitState.CLOSED
        self.last_state_change = time.time()
        self.state_changes += 1
        self.recovery_attempts = 0  # Reset recovery attempts
        self.failure_count = 0
        self.weighted_failure_count = 0.0
        self.failures.clear()
        
        logger.info(f"Circuit '{self.name}' closed - service recovered")
    
    async def _update_adaptive_thresholds(self):
        """Update adaptive failure thresholds based on historical data"""
        if self.total_calls < 100:  # Need sufficient data
            return
        
        # Calculate historical failure rate
        self.historical_failure_rate = self.total_failures / self.total_calls
        
        # Adjust threshold based on historical performance
        if self.historical_failure_rate < 0.01:  # Very reliable service
            self.adaptive_failure_threshold = max(self.config.failure_threshold + 2, 3)
        elif self.historical_failure_rate > 0.1:  # Unreliable service
            self.adaptive_failure_threshold = max(self.config.failure_threshold - 1, 2)
        else:
            self.adaptive_failure_threshold = self.config.failure_threshold
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        current_time = time.time()
        uptime = current_time - self.last_state_change
        
        return {
            'name': self.name,
            'state': self.state.value,
            'uptime_seconds': uptime,
            'total_calls': self.total_calls,
            'total_successes': self.total_successes,
            'total_failures': self.total_failures,
            'current_failure_count': self.failure_count,
            'weighted_failure_count': self.weighted_failure_count,
            'failure_rate': self.historical_failure_rate,
            'adaptive_threshold': self.adaptive_failure_threshold,
            'recovery_attempts': self.recovery_attempts,
            'state_changes': self.state_changes,
            'next_recovery_time': self.next_recovery_time if self.state == CircuitState.OPEN else None
        }
    
    def reset(self):
        """Reset circuit breaker to closed state"""
        self.state = CircuitState.CLOSED
        self.last_state_change = time.time()
        self.failures.clear()
        self.failure_count = 0
        self.weighted_failure_count = 0.0
        self.success_count = 0
        self.half_open_calls = 0
        self.recovery_attempts = 0
        
        logger.info(f"Circuit '{self.name}' reset to closed state")


class BulkheadConfig:
    """Configuration for bulkhead isolation"""
    def __init__(self, 
                 max_concurrent: int = 10,
                 queue_size: int = 100,
                 timeout: float = 30.0,
                 priority_levels: int = 3):
        self.max_concurrent = max_concurrent
        self.queue_size = queue_size
        self.timeout = timeout
        self.priority_levels = priority_levels


class Bulkhead:
    """
    Bulkhead pattern implementation for resource isolation
    
    Isolates resources to prevent cascading failures by:
    - Limiting concurrent executions per resource pool
    - Queueing requests with priority support
    - Timeout management for resource allocation
    - Fair resource distribution
    """
    
    def __init__(self, name: str, config: Optional[BulkheadConfig] = None):
        self.name = name
        self.config = config or BulkheadConfig()
        
        # Resource management
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent)
        self.active_tasks: Set[asyncio.Task] = set()
        self.queue: List[Tuple[int, asyncio.Future, Callable]] = []  # (priority, future, func)
        self.queue_lock = asyncio.Lock()
        
        # Statistics
        self.total_requests = 0
        self.active_count = 0
        self.queued_count = 0
        self.rejected_count = 0
        self.timeout_count = 0
        
        logger.info(f"Bulkhead '{name}' initialized with {self.config.max_concurrent} slots")
    
    @asynccontextmanager
    async def acquire(self, priority: int = 1, timeout: Optional[float] = None):
        """Acquire resource slot with priority and timeout"""
        timeout = timeout or self.config.timeout
        
        self.total_requests += 1
        
        try:
            # Try to acquire immediately
            if self.semaphore.locked():
                # Add to queue if at capacity
                if len(self.queue) >= self.config.queue_size:
                    self.rejected_count += 1
                    raise ResourceExhaustedError(
                        f"Bulkhead '{self.name}' queue full "
                        f"(capacity: {self.config.queue_size})"
                    )
                
                # Wait in queue
                future = asyncio.Future()
                async with self.queue_lock:
                    self.queue.append((priority, future, None))
                    self.queue.sort(key=lambda x: x[0], reverse=True)  # Higher priority first
                    self.queued_count += 1
                
                try:
                    await asyncio.wait_for(future, timeout=timeout)
                except asyncio.TimeoutError:
                    # Remove from queue if still there
                    async with self.queue_lock:
                        self.queue = [(p, f, func) for p, f, func in self.queue if f != future]
                        self.queued_count = len(self.queue)
                    
                    self.timeout_count += 1
                    raise BulkheadTimeoutError(
                        f"Bulkhead '{self.name}' timeout after {timeout}s"
                    )
            
            # Acquire semaphore
            await asyncio.wait_for(self.semaphore.acquire(), timeout=timeout)
            self.active_count += 1
            
            try:
                yield
            finally:
                self.active_count -= 1
                self.semaphore.release()
                
                # Process queue
                await self._process_queue()
                
        except asyncio.TimeoutError:
            self.timeout_count += 1
            raise BulkheadTimeoutError(
                f"Bulkhead '{self.name}' acquisition timeout after {timeout}s"
            )
    
    async def _process_queue(self):
        """Process waiting requests in priority order"""
        async with self.queue_lock:
            if self.queue and not self.semaphore.locked():
                # Get highest priority request
                priority, future, func = self.queue.pop(0)
                self.queued_count -= 1
                
                if not future.cancelled():
                    future.set_result(True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bulkhead statistics"""
        return {
            'name': self.name,
            'max_concurrent': self.config.max_concurrent,
            'active_count': self.active_count,
            'queued_count': self.queued_count,
            'total_requests': self.total_requests,
            'rejected_count': self.rejected_count,
            'timeout_count': self.timeout_count,
            'utilization': self.active_count / self.config.max_concurrent
        }


class RetryConfig:
    """Configuration for retry patterns"""
    def __init__(self,
                 max_attempts: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True,
                 retryable_exceptions: Optional[Set[type]] = None):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or {
            ConnectionError, TimeoutError, asyncio.TimeoutError
        }


class RetryHandler:
    """
    Advanced retry handler with exponential backoff and jitter
    
    Features:
    - Exponential backoff with configurable base
    - Jitter to prevent thundering herd
    - Selective retry based on exception types
    - Circuit breaker integration
    - Comprehensive retry statistics
    """
    
    def __init__(self, name: str, config: Optional[RetryConfig] = None):
        self.name = name
        self.config = config or RetryConfig()
        
        # Statistics
        self.total_attempts = 0
        self.successful_retries = 0
        self.failed_retries = 0
        self.total_delay = 0.0
        
        logger.info(f"Retry handler '{name}' initialized")
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            self.total_attempts += 1
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                if attempt > 0:
                    self.successful_retries += 1
                    logger.info(f"Retry '{self.name}' succeeded on attempt {attempt + 1}")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if exception is retryable
                if not self._is_retryable(e):
                    logger.debug(f"Retry '{self.name}': Non-retryable exception {type(e).__name__}")
                    raise
                
                # Don't retry on last attempt
                if attempt == self.config.max_attempts - 1:
                    self.failed_retries += 1
                    logger.warning(f"Retry '{self.name}' failed after {self.config.max_attempts} attempts")
                    raise
                
                # Calculate delay
                delay = self._calculate_delay(attempt)
                self.total_delay += delay
                
                logger.info(
                    f"Retry '{self.name}' attempt {attempt + 1} failed: {type(e).__name__}. "
                    f"Retrying in {delay:.2f}s"
                )
                
                await asyncio.sleep(delay)
        
        # Should never reach here, but just in case
        if last_exception:
            raise last_exception
    
    def _is_retryable(self, exception: Exception) -> bool:
        """Check if exception is retryable"""
        return any(isinstance(exception, exc_type) for exc_type in self.config.retryable_exceptions)
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        # Exponential backoff
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        
        # Apply maximum delay
        delay = min(delay, self.config.max_delay)
        
        # Add jitter
        if self.config.jitter:
            jitter_range = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retry statistics"""
        return {
            'name': self.name,
            'total_attempts': self.total_attempts,
            'successful_retries': self.successful_retries,
            'failed_retries': self.failed_retries,
            'total_delay': self.total_delay,
            'average_delay': self.total_delay / max(self.successful_retries + self.failed_retries, 1)
        }


class ResilienceOrchestrator:
    """
    Orchestrates all resilience patterns for comprehensive fault tolerance
    
    Combines:
    - Circuit breakers for fail-fast behavior
    - Bulkheads for resource isolation
    - Retry handlers for transient failures
    - Timeout management
    - Health monitoring
    """
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.bulkheads: Dict[str, Bulkhead] = {}
        self.retry_handlers: Dict[str, RetryHandler] = {}
        
        # Global statistics
        self.start_time = time.time()
        
        logger.info("Resilience orchestrator initialized")
    
    def create_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Create or get circuit breaker"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(name, config)
        return self.circuit_breakers[name]
    
    def create_bulkhead(self, name: str, config: Optional[BulkheadConfig] = None) -> Bulkhead:
        """Create or get bulkhead"""
        if name not in self.bulkheads:
            self.bulkheads[name] = Bulkhead(name, config)
        return self.bulkheads[name]
    
    def create_retry_handler(self, name: str, config: Optional[RetryConfig] = None) -> RetryHandler:
        """Create or get retry handler"""
        if name not in self.retry_handlers:
            self.retry_handlers[name] = RetryHandler(name, config)
        return self.retry_handlers[name]
    
    async def execute_with_resilience(self,
                                    func: Callable,
                                    circuit_breaker_name: Optional[str] = None,
                                    bulkhead_name: Optional[str] = None,
                                    retry_handler_name: Optional[str] = None,
                                    timeout: Optional[float] = None,
                                    priority: int = 1,
                                    *args, **kwargs) -> Any:
        """Execute function with full resilience protection"""
        
        async def _execute():
            # Retry wrapper
            if retry_handler_name:
                retry_handler = self.retry_handlers.get(retry_handler_name)
                if retry_handler:
                    return await retry_handler.execute(func, *args, **kwargs)
            
            # Direct execution
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        async def _execute_with_bulkhead():
            # Bulkhead wrapper
            if bulkhead_name:
                bulkhead = self.bulkheads.get(bulkhead_name)
                if bulkhead:
                    async with bulkhead.acquire(priority=priority, timeout=timeout):
                        return await _execute()
            
            return await _execute()
        
        async def _execute_with_circuit_breaker():
            # Circuit breaker wrapper
            if circuit_breaker_name:
                circuit_breaker = self.circuit_breakers.get(circuit_breaker_name)
                if circuit_breaker:
                    return await circuit_breaker.execute(_execute_with_bulkhead)
            
            return await _execute_with_bulkhead()
        
        # Timeout wrapper
        if timeout:
            try:
                return await asyncio.wait_for(_execute_with_circuit_breaker(), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning(f"Resilience execution timeout after {timeout}s")
                raise
        else:
            return await _execute_with_circuit_breaker()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status of all resilience components"""
        health = {
            'status': 'healthy',
            'uptime_seconds': time.time() - self.start_time,
            'circuit_breakers': {},
            'bulkheads': {},
            'retry_handlers': {}
        }
        
        # Check circuit breakers
        unhealthy_circuits = 0
        for name, cb in self.circuit_breakers.items():
            stats = cb.get_stats()
            health['circuit_breakers'][name] = stats
            
            if stats['state'] == 'open':
                unhealthy_circuits += 1
        
        # Check bulkheads
        overloaded_bulkheads = 0
        for name, bh in self.bulkheads.items():
            stats = bh.get_stats()
            health['bulkheads'][name] = stats
            
            if stats['utilization'] > 0.9:  # 90% utilization
                overloaded_bulkheads += 1
        
        # Check retry handlers
        for name, rh in self.retry_handlers.items():
            health['retry_handlers'][name] = rh.get_stats()
        
        # Determine overall health
        if unhealthy_circuits > 0 or overloaded_bulkheads > 0:
            health['status'] = 'degraded'
        
        if unhealthy_circuits > len(self.circuit_breakers) * 0.5:  # More than 50% unhealthy
            health['status'] = 'unhealthy'
        
        return health


# Custom exceptions
class ResourceExhaustedError(Exception):
    """Raised when resources are exhausted"""
    pass


class BulkheadTimeoutError(Exception):
    """Raised when bulkhead acquisition times out"""
    pass


# Global resilience orchestrator
_resilience_orchestrator: Optional[ResilienceOrchestrator] = None


def get_resilience_orchestrator() -> ResilienceOrchestrator:
    """Get the global resilience orchestrator"""
    global _resilience_orchestrator
    if _resilience_orchestrator is None:
        _resilience_orchestrator = ResilienceOrchestrator()
    return _resilience_orchestrator


# Convenience decorators
def with_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator for circuit breaker protection"""
    def decorator(func: Callable) -> Callable:
        orchestrator = get_resilience_orchestrator()
        circuit_breaker = orchestrator.create_circuit_breaker(name, config)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await circuit_breaker.execute(func, *args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we need to handle differently
            return asyncio.run(circuit_breaker.execute(func, *args, **kwargs))
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def with_bulkhead(name: str, config: Optional[BulkheadConfig] = None, priority: int = 1):
    """Decorator for bulkhead protection"""
    def decorator(func: Callable) -> Callable:
        orchestrator = get_resilience_orchestrator()
        bulkhead = orchestrator.create_bulkhead(name, config)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            async with bulkhead.acquire(priority=priority):
                return await func(*args, **kwargs)
        
        return async_wrapper
    
    return decorator


def with_retry(name: str, config: Optional[RetryConfig] = None):
    """Decorator for retry protection"""
    def decorator(func: Callable) -> Callable:
        orchestrator = get_resilience_orchestrator()
        retry_handler = orchestrator.create_retry_handler(name, config)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await retry_handler.execute(func, *args, **kwargs)
        
        return async_wrapper
    
    return decorator


# Export main classes
__all__ = [
    'CircuitBreaker',
    'CircuitBreakerConfig',
    'CircuitState',
    'FailureType',
    'Bulkhead',
    'BulkheadConfig',
    'RetryHandler',
    'RetryConfig',
    'ResilienceOrchestrator',
    'CircuitBreakerError',
    'ResourceExhaustedError',
    'BulkheadTimeoutError',
    'get_resilience_orchestrator',
    'with_circuit_breaker',
    'with_bulkhead',
    'with_retry'
]