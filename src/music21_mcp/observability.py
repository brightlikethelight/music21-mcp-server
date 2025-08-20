#!/usr/bin/env python3
"""
Observability System for Music21 MCP Server

Provides structured logging, metrics collection, and monitoring
for production-ready observability and debugging.
"""

import functools
import json
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from contextvars import ContextVar
from enum import Enum
from typing import Any

# Context variables for request correlation
REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="")
USER_ID: ContextVar[str] = ContextVar("user_id", default="")
OPERATION: ContextVar[str] = ContextVar("operation", default="")


class LogLevel(Enum):
    """Structured log levels"""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics we collect"""

    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    TIMER = "timer"


class StructuredLogger:
    """
    Production-ready structured logger with context and correlation IDs.

    Provides:
    - JSON-structured logging for machine readability
    - Request correlation IDs for tracing
    - Contextual information injection
    - Performance timing integration
    - Error categorization and tracking
    """

    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # Configure structured formatter if not already configured
        if not self.logger.handlers:
            self._configure_handler()

    def _configure_handler(self) -> None:
        """Configure structured JSON handler"""
        handler = logging.StreamHandler()
        formatter = StructuredFormatter()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _build_log_entry(
        self, level: LogLevel, message: str, **kwargs: Any
    ) -> dict[str, Any]:
        """Build structured log entry with context"""
        entry = {
            "timestamp": time.time(),
            "level": level.value,
            "message": message,
            "logger": self.logger.name,
        }

        # Add correlation context
        request_id = REQUEST_ID.get()
        if request_id:
            entry["request_id"] = request_id

        user_id = USER_ID.get()
        if user_id:
            entry["user_id"] = user_id

        operation = OPERATION.get()
        if operation:
            entry["operation"] = operation

        # Add additional context
        entry.update(kwargs)

        return entry

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message"""
        entry = self._build_log_entry(LogLevel.DEBUG, message, **kwargs)
        self.logger.debug(json.dumps(entry))

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message"""
        entry = self._build_log_entry(LogLevel.INFO, message, **kwargs)
        self.logger.info(json.dumps(entry))

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message"""
        entry = self._build_log_entry(LogLevel.WARNING, message, **kwargs)
        self.logger.warning(json.dumps(entry))

    def error(self, message: str, error: Exception | None = None, **kwargs: Any) -> None:
        """Log error message with optional exception details"""
        entry = self._build_log_entry(LogLevel.ERROR, message, **kwargs)

        if error:
            entry.update(
                {
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "error_module": getattr(error, "__module__", "unknown"),
                }
            )

        self.logger.error(json.dumps(entry))

    def critical(self, message: str, error: Exception | None = None, **kwargs: Any) -> None:
        """Log critical message"""
        entry = self._build_log_entry(LogLevel.CRITICAL, message, **kwargs)

        if error:
            entry.update(
                {
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "error_module": getattr(error, "__module__", "unknown"),
                }
            )

        self.logger.critical(json.dumps(entry))


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging"""

    def format(self, record: Any) -> str:
        # If the message is already JSON (from StructuredLogger), return as-is
        try:
            json.loads(record.getMessage())
            return record.getMessage()
        except (json.JSONDecodeError, ValueError):
            # Fall back to basic structured format
            return json.dumps(
                {
                    "timestamp": record.created,
                    "level": record.levelname.lower(),
                    "message": record.getMessage(),
                    "logger": record.name,
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno,
                }
            )


class MetricsCollector:
    """
    Thread-safe metrics collection system.

    Collects and aggregates performance metrics for monitoring
    and alerting in production environments.
    """

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._lock = threading.RLock()

        # Metric storage
        self._counters: dict[str, int] = defaultdict(int)
        self._histograms: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=max_history)
        )
        self._gauges: dict[str, float] = {}
        self._timers: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=max_history))

        # Metadata
        self._start_time = time.time()
        self._metric_metadata: dict[str, dict[str, Any]] = {}

    def increment_counter(self, name: str, value: int = 1, **labels: Any) -> None:
        """Increment a counter metric"""
        with self._lock:
            key = self._build_metric_key(name, labels)
            self._counters[key] += value
            self._record_metadata(key, MetricType.COUNTER, labels)

    def record_histogram(self, name: str, value: float, **labels: Any) -> None:
        """Record a histogram value"""
        with self._lock:
            key = self._build_metric_key(name, labels)
            self._histograms[key].append(value)
            self._record_metadata(key, MetricType.HISTOGRAM, labels)

    def set_gauge(self, name: str, value: float, **labels: Any) -> None:
        """Set a gauge value"""
        with self._lock:
            key = self._build_metric_key(name, labels)
            self._gauges[key] = value
            self._record_metadata(key, MetricType.GAUGE, labels)

    def record_timer(self, name: str, duration: float, **labels: Any) -> None:
        """Record a timer duration"""
        with self._lock:
            key = self._build_metric_key(name, labels)
            self._timers[key].append(duration)
            self._record_metadata(key, MetricType.TIMER, labels)

    def get_metrics(self) -> dict[str, Any]:
        """Get all collected metrics"""
        with self._lock:
            metrics = {
                "counters": dict(self._counters),
                "histograms": {
                    key: {
                        "count": len(values),
                        "mean": sum(values) / len(values) if values else 0,
                        "min": min(values) if values else 0,
                        "max": max(values) if values else 0,
                        "recent": list(values)[-10:],  # Last 10 values
                    }
                    for key, values in self._histograms.items()
                },
                "gauges": dict(self._gauges),
                "timers": {
                    key: {
                        "count": len(durations),
                        "mean_ms": (sum(durations) / len(durations)) * 1000
                        if durations
                        else 0,
                        "min_ms": min(durations) * 1000 if durations else 0,
                        "max_ms": max(durations) * 1000 if durations else 0,
                        "p95_ms": self._percentile(durations, 0.95) * 1000
                        if durations
                        else 0,
                        "recent": [
                            (d * 1000) for d in list(durations)[-10:]
                        ],  # Last 10 in ms
                    }
                    for key, durations in self._timers.items()
                },
                "metadata": {
                    "start_time": self._start_time,
                    "uptime_seconds": time.time() - self._start_time,
                    "metric_definitions": self._metric_metadata,
                },
            }
            return metrics

    def reset_metrics(self) -> None:
        """Reset all metrics (useful for testing)"""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()
            self._gauges.clear()
            self._timers.clear()
            self._metric_metadata.clear()

    def _build_metric_key(self, name: str, labels: dict[str, Any]) -> str:
        """Build metric key with labels"""
        if not labels:
            return name

        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def _record_metadata(
        self, key: str, metric_type: MetricType, labels: dict[str, Any]
    ) -> None:
        """Record metadata about a metric"""
        if key not in self._metric_metadata:
            self._metric_metadata[key] = {
                "type": metric_type.value,
                "labels": labels,
                "first_seen": time.time(),
            }

        self._metric_metadata[key]["last_seen"] = time.time()

    def _percentile(self, values: deque[float], percentile: float) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)
        return sorted_values[min(index, len(sorted_values) - 1)]


# Global instances
_metrics_collector = MetricsCollector()
_logger = StructuredLogger("music21_mcp")


def get_metrics() -> dict[str, Any]:
    """Get global metrics"""
    return _metrics_collector.get_metrics()


def get_logger(name: str = "music21_mcp") -> StructuredLogger:
    """Get structured logger instance"""
    return StructuredLogger(name)


def with_context(request_id: str | None = None, user_id: str | None = None, operation: str | None = None) -> Any:
    """Context manager for request correlation"""

    class ContextManager:
        def __init__(self, req_id: str | None, usr_id: str | None, op: str | None) -> None:
            self.request_id = req_id or str(uuid.uuid4())
            self.user_id = usr_id
            self.operation = op
            self.request_token: Any = None
            self.user_token: Any = None
            self.operation_token: Any = None

        def __enter__(self) -> Any:
            self.request_token = REQUEST_ID.set(self.request_id)
            if self.user_id:
                self.user_token = USER_ID.set(self.user_id)
            if self.operation:
                self.operation_token = OPERATION.set(self.operation)
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            # Reset tokens in reverse order
            try:
                if self.operation_token:
                    OPERATION.reset(self.operation_token)
            except LookupError:
                pass

            try:
                if self.user_token:
                    USER_ID.reset(self.user_token)
            except LookupError:
                pass

            try:
                if self.request_token:
                    REQUEST_ID.reset(self.request_token)
            except LookupError:
                pass

    return ContextManager(request_id, user_id, operation)


def monitor_performance(operation_name: str | None = None, track_errors: bool = True) -> Any:
    """Decorator for monitoring function performance"""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        op_name = operation_name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            logger = get_logger()

            logger.debug(f"Starting {op_name}", operation=op_name)

            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                # Record successful operation metrics
                _metrics_collector.increment_counter(
                    "operations_total", operation=op_name, status="success"
                )
                _metrics_collector.record_timer(
                    "operation_duration", duration, operation=op_name
                )

                logger.info(
                    f"Completed {op_name}",
                    operation=op_name,
                    duration_ms=duration * 1000,
                    status="success",
                )

                return result

            except Exception as error:
                duration = time.time() - start_time

                # Record failed operation metrics
                _metrics_collector.increment_counter(
                    "operations_total",
                    operation=op_name,
                    status="error",
                    error_type=type(error).__name__,
                )
                _metrics_collector.record_timer(
                    "operation_duration", duration, operation=op_name, status="error"
                )

                if track_errors:
                    logger.error(
                        f"Failed {op_name}",
                        error=error,
                        operation=op_name,
                        duration_ms=duration * 1000,
                        status="error",
                    )

                raise

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            logger = get_logger()

            logger.debug(f"Starting {op_name}", operation=op_name)

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                # Record successful operation metrics
                _metrics_collector.increment_counter(
                    "operations_total", operation=op_name, status="success"
                )
                _metrics_collector.record_timer(
                    "operation_duration", duration, operation=op_name
                )

                logger.info(
                    f"Completed {op_name}",
                    operation=op_name,
                    duration_ms=duration * 1000,
                    status="success",
                )

                return result

            except Exception as error:
                duration = time.time() - start_time

                # Record failed operation metrics
                _metrics_collector.increment_counter(
                    "operations_total",
                    operation=op_name,
                    status="error",
                    error_type=type(error).__name__,
                )
                _metrics_collector.record_timer(
                    "operation_duration", duration, operation=op_name, status="error"
                )

                if track_errors:
                    logger.error(
                        f"Failed {op_name}",
                        error=error,
                        operation=op_name,
                        duration_ms=duration * 1000,
                        status="error",
                    )

                raise

        # Return appropriate wrapper based on function type
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
