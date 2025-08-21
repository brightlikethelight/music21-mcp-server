"""
Rate Limiting Module for Music21 MCP Server

Implements rate limiting to protect API endpoints from abuse
and ensure fair resource usage across clients.
"""

import asyncio
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse


class RateLimitStrategy(Enum):
    """Rate limiting strategies"""

    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"  # noqa: S105
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""

    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_size: int = 10
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW

    # Different limits for different endpoints
    endpoint_limits: dict[str, int] | None = None

    def __post_init__(self):
        if self.endpoint_limits is None:
            self.endpoint_limits = {
                # High-cost operations get lower limits
                "/scores/import": 10,  # per minute
                "/scores/upload": 5,  # per minute
                "/analysis/harmony": 20,  # per minute
                "/generation/counterpoint": 10,  # per minute
                # Low-cost operations get higher limits
                "/health": 120,  # per minute
                "/scores": 60,  # per minute
                "/": 120,  # per minute
            }


class RateLimiter:
    """Token bucket rate limiter implementation"""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.buckets: dict[str, TokenBucket] = {}
        self.request_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=10000)
        )
        self._cleanup_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    async def check_rate_limit(
        self, identifier: str, endpoint: str | None = None, cost: int = 1
    ) -> tuple[bool, dict[str, Any]]:
        """
        Check if request is within rate limits

        Returns:
            tuple: (allowed, metadata)
            - allowed: Whether request is allowed
            - metadata: Rate limit information for headers
        """
        async with self._lock:
            # Get or create bucket for this identifier
            if identifier not in self.buckets:
                self.buckets[identifier] = TokenBucket(
                    capacity=self.config.burst_size,
                    refill_rate=self.config.requests_per_minute / 60.0,
                )

            bucket = self.buckets[identifier]

            # Check endpoint-specific limits
            if (
                endpoint
                and self.config.endpoint_limits
                and endpoint in self.config.endpoint_limits
            ):
                endpoint_limit = self.config.endpoint_limits[endpoint]
                endpoint_bucket_key = f"{identifier}:{endpoint}"

                if endpoint_bucket_key not in self.buckets:
                    self.buckets[endpoint_bucket_key] = TokenBucket(
                        capacity=min(5, endpoint_limit),
                        refill_rate=endpoint_limit / 60.0,
                    )

                endpoint_bucket = self.buckets[endpoint_bucket_key]
                endpoint_allowed = endpoint_bucket.consume(cost)

                if not endpoint_allowed:
                    return False, self._get_metadata(endpoint_bucket, endpoint)

            # Check global limits
            allowed = bucket.consume(cost)

            # Track request history for sliding window
            if self.config.strategy == RateLimitStrategy.SLIDING_WINDOW:
                now = time.time()
                history = self.request_history[identifier]
                history.append(now)

                # Check sliding window limits
                minute_ago = now - 60
                hour_ago = now - 3600
                day_ago = now - 86400

                minute_count = sum(1 for t in history if t > minute_ago)
                hour_count = sum(1 for t in history if t > hour_ago)
                day_count = sum(1 for t in history if t > day_ago)

                if (
                    minute_count > self.config.requests_per_minute
                    or hour_count > self.config.requests_per_hour
                    or day_count > self.config.requests_per_day
                ):
                    allowed = False

            return allowed, self._get_metadata(bucket, endpoint)

    def _get_metadata(
        self, bucket: "TokenBucket", endpoint: str | None = None
    ) -> dict[str, Any]:
        """Get rate limit metadata for response headers"""
        return {
            "limit": int(bucket.capacity),
            "remaining": int(bucket.tokens),
            "reset": int(
                time.time() + (bucket.capacity - bucket.tokens) / bucket.refill_rate
            ),
            "retry_after": None
            if bucket.tokens > 0
            else int((1 - bucket.tokens) / bucket.refill_rate),
            "endpoint": endpoint,
        }

    async def cleanup_expired(self):
        """Clean up expired request history"""
        async with self._lock:
            now = time.time()
            day_ago = now - 86400

            for identifier in list(self.request_history.keys()):
                history = self.request_history[identifier]
                # Remove old entries
                while history and history[0] < day_ago:
                    history.popleft()

                # Remove empty histories
                if not history:
                    del self.request_history[identifier]

            # Clean up unused buckets
            for key in list(self.buckets.keys()):
                bucket = self.buckets[key]
                if bucket.last_update < now - 3600:  # Inactive for 1 hour
                    del self.buckets[key]

    def start_cleanup_task(self):
        """Start background cleanup task"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self):
        """Background task to clean up expired data"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self.cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue
                import logging

                logging.getLogger(__name__).error(f"Error in rate limiter cleanup: {e}")

    def stop_cleanup_task(self):
        """Stop background cleanup task"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None


class TokenBucket:
    """Token bucket algorithm implementation"""

    def __init__(self, capacity: float, refill_rate: float):
        """
        Initialize token bucket

        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_update = time.time()

    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from bucket

        Returns:
            bool: Whether tokens were available
        """
        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def _refill(self):
        """Refill bucket based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_update

        # Add tokens based on refill rate
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)

        self.last_update = now


class RateLimitMiddleware:
    """FastAPI middleware for rate limiting"""

    def __init__(self, config: RateLimitConfig | None = None):
        self.config = config or RateLimitConfig()
        self.limiter = RateLimiter(self.config)
        self.limiter.start_cleanup_task()

    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting"""
        # Get client identifier (IP address or API key)
        client_ip = request.client.host if request.client else "unknown"
        api_key = request.headers.get("X-API-Key")
        identifier = api_key if api_key else client_ip

        # Get endpoint path
        endpoint = request.url.path

        # Check rate limit
        allowed, metadata = await self.limiter.check_rate_limit(
            identifier=identifier, endpoint=endpoint
        )

        if not allowed:
            # Return 429 Too Many Requests
            retry_after = metadata.get("retry_after", 60)
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Please retry after {retry_after} seconds.",
                    "retry_after": retry_after,
                },
                headers={
                    "X-RateLimit-Limit": str(metadata["limit"]),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(metadata["reset"]),
                    "Retry-After": str(retry_after),
                },
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(metadata["limit"])
        response.headers["X-RateLimit-Remaining"] = str(metadata["remaining"])
        response.headers["X-RateLimit-Reset"] = str(metadata["reset"])

        return response


def create_rate_limiter(
    requests_per_minute: int = 60,
    requests_per_hour: int = 1000,
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW,
) -> RateLimitMiddleware:
    """
    Create a rate limiter middleware

    Args:
        requests_per_minute: Maximum requests per minute
        requests_per_hour: Maximum requests per hour
        strategy: Rate limiting strategy to use

    Returns:
        RateLimitMiddleware: Configured middleware
    """
    config = RateLimitConfig(
        requests_per_minute=requests_per_minute,
        requests_per_hour=requests_per_hour,
        strategy=strategy,
    )

    return RateLimitMiddleware(config)


# Decorator for rate-limited endpoints
def rate_limit(
    requests_per_minute: int = 60, cost: int = 1, key_func: Callable | None = None
):
    """
    Decorator to apply rate limiting to specific endpoints

    Args:
        requests_per_minute: Rate limit for this endpoint
        cost: Cost of this operation (for weighted rate limiting)
        key_func: Function to extract identifier from request
    """

    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            # Get identifier
            if key_func:
                identifier = key_func(request)
            else:
                identifier = request.client.host if request.client else "unknown"

            # Simple in-memory rate limit check
            # In production, use Redis or similar
            if not hasattr(wrapper, "_rate_limit_data"):
                wrapper._rate_limit_data = {}  # type: ignore

            now = time.time()
            data = wrapper._rate_limit_data.get(  # type: ignore
                identifier, {"count": 0, "reset": now + 60}
            )

            if now > data["reset"]:
                data = {"count": 0, "reset": now + 60}

            if data["count"] >= requests_per_minute:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded",
                    headers={"Retry-After": str(int(data["reset"] - now))},
                )

            data["count"] += cost
            wrapper._rate_limit_data[identifier] = data  # type: ignore

            # Clean up old entries
            if len(wrapper._rate_limit_data) > 1000:  # type: ignore
                # Remove expired entries
                wrapper._rate_limit_data = {  # type: ignore
                    k: v
                    for k, v in wrapper._rate_limit_data.items()  # type: ignore
                    if v["reset"] > now
                }

            return await func(request, *args, **kwargs)

        return wrapper

    return decorator
