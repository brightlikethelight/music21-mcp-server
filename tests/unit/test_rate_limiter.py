"""
Comprehensive tests for rate_limiter module to boost coverage to 76%+

Tests all components:
- RateLimitStrategy enum
- RateLimitConfig dataclass
- TokenBucket class
- RateLimiter class
- RateLimitMiddleware class
- Utility functions and decorators
"""

import asyncio
import contextlib
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse

from music21_mcp.rate_limiter import (
    RateLimitConfig,
    RateLimiter,
    RateLimitMiddleware,
    RateLimitStrategy,
    TokenBucket,
    create_rate_limiter,
    rate_limit,
)


class TestRateLimitStrategy:
    """Test RateLimitStrategy enum"""

    def test_strategy_values(self):
        """Test enum values"""
        assert RateLimitStrategy.FIXED_WINDOW.value == "fixed_window"
        assert RateLimitStrategy.SLIDING_WINDOW.value == "sliding_window"
        assert RateLimitStrategy.TOKEN_BUCKET.value == "token_bucket"
        assert RateLimitStrategy.LEAKY_BUCKET.value == "leaky_bucket"

    def test_strategy_comparison(self):
        """Test enum comparison"""
        assert RateLimitStrategy.FIXED_WINDOW != RateLimitStrategy.SLIDING_WINDOW
        assert str(RateLimitStrategy.TOKEN_BUCKET) == "RateLimitStrategy.TOKEN_BUCKET"


class TestRateLimitConfig:
    """Test RateLimitConfig dataclass"""

    def test_config_default_initialization(self):
        """Test default config initialization"""
        config = RateLimitConfig()

        assert config.requests_per_minute == 60
        assert config.requests_per_hour == 1000
        assert config.requests_per_day == 10000
        assert config.burst_size == 10
        assert config.strategy == RateLimitStrategy.SLIDING_WINDOW
        assert config.endpoint_limits is not None
        assert isinstance(config.endpoint_limits, dict)

    def test_config_custom_initialization(self):
        """Test custom config initialization"""
        custom_limits = {"/custom": 5}
        config = RateLimitConfig(
            requests_per_minute=30,
            requests_per_hour=500,
            requests_per_day=5000,
            burst_size=5,
            strategy=RateLimitStrategy.TOKEN_BUCKET,
            endpoint_limits=custom_limits,
        )

        assert config.requests_per_minute == 30
        assert config.requests_per_hour == 500
        assert config.requests_per_day == 5000
        assert config.burst_size == 5
        assert config.strategy == RateLimitStrategy.TOKEN_BUCKET
        assert config.endpoint_limits == custom_limits

    def test_config_post_init_default_limits(self):
        """Test __post_init__ sets default endpoint limits"""
        config = RateLimitConfig(endpoint_limits=None)

        assert config.endpoint_limits is not None
        assert "/scores/import" in config.endpoint_limits
        assert "/health" in config.endpoint_limits
        assert config.endpoint_limits["/scores/import"] == 10
        assert config.endpoint_limits["/health"] == 120

    def test_config_post_init_preserves_custom_limits(self):
        """Test __post_init__ preserves existing limits"""
        custom_limits = {"/test": 15}
        config = RateLimitConfig(endpoint_limits=custom_limits)

        assert config.endpoint_limits == custom_limits


class TestTokenBucket:
    """Test TokenBucket class"""

    def test_bucket_initialization(self):
        """Test TokenBucket initialization"""
        bucket = TokenBucket(capacity=10.0, refill_rate=2.0)

        assert bucket.capacity == 10.0
        assert bucket.refill_rate == 2.0
        assert bucket.tokens == 10.0  # Starts full
        assert bucket.last_update <= time.time()

    def test_consume_with_available_tokens(self):
        """Test consuming tokens when available"""
        bucket = TokenBucket(capacity=10.0, refill_rate=1.0)

        # Should consume successfully
        assert bucket.consume(5) is True
        assert bucket.tokens == 5.0

    def test_consume_with_insufficient_tokens(self):
        """Test consuming tokens when insufficient"""
        bucket = TokenBucket(capacity=10.0, refill_rate=1.0)

        # Consume most tokens
        bucket.consume(9)

        # Mock time to prevent automatic refill
        with patch("time.time", return_value=bucket.last_update):
            # Should fail to consume more than available
            assert bucket.consume(5) is False
            assert bucket.tokens == 1.0  # Unchanged

    def test_consume_exact_tokens(self):
        """Test consuming exact number of tokens"""
        bucket = TokenBucket(capacity=10.0, refill_rate=1.0)

        # Consume exact amount
        assert bucket.consume(10) is True
        assert bucket.tokens == 0.0

    def test_refill_over_time(self):
        """Test token refill over time"""
        bucket = TokenBucket(capacity=10.0, refill_rate=2.0)  # 2 tokens per second

        # Consume all tokens
        bucket.consume(10)
        assert bucket.tokens == 0.0

        # Mock time passage (1 second)
        original_time = bucket.last_update
        with patch("time.time", return_value=original_time + 1.0) as mock_time:
            # Should refill 2 tokens
            bucket._refill()
            assert bucket.tokens == 2.0

    def test_refill_caps_at_capacity(self):
        """Test refill doesn't exceed capacity"""
        bucket = TokenBucket(capacity=10.0, refill_rate=5.0)

        # Start with some tokens
        bucket.tokens = 8.0

        # Mock long time passage (10 seconds - would add 50 tokens)
        original_time = bucket.last_update
        with patch("time.time", return_value=original_time + 10.0):
            bucket._refill()
            # Should be capped at capacity
            assert bucket.tokens == 10.0

    def test_consume_triggers_refill(self):
        """Test that consume() triggers refill"""
        bucket = TokenBucket(capacity=10.0, refill_rate=2.0)

        # Consume some tokens
        bucket.consume(5)
        original_time = bucket.last_update

        # Mock time passage and consume again
        with patch("time.time") as mock_time:
            mock_time.side_effect = [original_time + 1.0, original_time + 1.0]

            # Should refill 2 tokens before consuming
            assert bucket.consume(3) is True
            # 5 remaining + 2 refilled - 3 consumed = 4
            assert bucket.tokens == 4.0


class TestRateLimiter:
    """Test RateLimiter class"""

    @pytest.fixture
    def rate_limiter(self):
        """Create a RateLimiter instance"""
        config = RateLimitConfig(
            requests_per_minute=60,
            burst_size=10,
        )
        return RateLimiter(config)

    def test_limiter_initialization(self, rate_limiter):
        """Test RateLimiter initialization"""
        assert isinstance(rate_limiter.config, RateLimitConfig)
        assert rate_limiter.buckets == {}
        assert len(rate_limiter.request_history) == 0
        assert rate_limiter._cleanup_task is None
        assert rate_limiter._lock is not None

    @pytest.mark.asyncio
    async def test_check_rate_limit_first_request(self, rate_limiter):
        """Test first request creates bucket and succeeds"""
        allowed, metadata = await rate_limiter.check_rate_limit("user1")

        assert allowed is True
        assert "user1" in rate_limiter.buckets
        assert metadata["limit"] == 10  # burst_size
        assert metadata["remaining"] == 9  # 10 - 1 consumed

    @pytest.mark.asyncio
    async def test_check_rate_limit_multiple_requests(self, rate_limiter):
        """Test multiple requests from same identifier"""
        identifier = "user1"

        # Make several requests
        for i in range(5):
            allowed, metadata = await rate_limiter.check_rate_limit(identifier)
            assert allowed is True
            assert metadata["remaining"] == 10 - i - 1

    @pytest.mark.asyncio
    async def test_check_rate_limit_exceed_burst(self, rate_limiter):
        """Test exceeding burst limit"""
        identifier = "user1"

        # Consume all burst tokens
        for _ in range(10):
            allowed, _ = await rate_limiter.check_rate_limit(identifier)
            assert allowed is True

        # Next request should be rejected
        allowed, metadata = await rate_limiter.check_rate_limit(identifier)
        assert allowed is False
        assert metadata["remaining"] == 0

    @pytest.mark.asyncio
    async def test_check_rate_limit_with_cost(self, rate_limiter):
        """Test rate limiting with custom cost"""
        identifier = "user1"

        # Consume tokens with custom cost
        allowed, metadata = await rate_limiter.check_rate_limit(identifier, cost=5)
        assert allowed is True
        assert metadata["remaining"] == 5  # 10 - 5

        # Next high-cost request should be rejected
        allowed, _ = await rate_limiter.check_rate_limit(identifier, cost=8)
        assert allowed is False

    @pytest.mark.asyncio
    async def test_check_rate_limit_endpoint_specific(self, rate_limiter):
        """Test endpoint-specific rate limiting"""
        identifier = "user1"
        endpoint = "/scores/import"  # Limited to 10 per minute in default config

        # First request should work
        allowed, metadata = await rate_limiter.check_rate_limit(
            identifier, endpoint=endpoint
        )
        assert allowed is True
        assert metadata["endpoint"] == endpoint

    @pytest.mark.asyncio
    async def test_check_rate_limit_endpoint_exceeded(self, rate_limiter):
        """Test endpoint rate limit exceeded"""
        identifier = "user1"
        endpoint = "/scores/import"

        # Get endpoint bucket and exhaust it
        bucket_key = f"{identifier}:{endpoint}"

        # Make requests until endpoint limit is hit
        attempts = 0
        while attempts < 20:  # Safety limit
            allowed, metadata = await rate_limiter.check_rate_limit(
                identifier, endpoint=endpoint
            )
            attempts += 1
            if not allowed:
                break

        # Should eventually be rate limited
        assert not allowed
        assert metadata["endpoint"] == endpoint

    @pytest.mark.asyncio
    async def test_check_rate_limit_sliding_window(self, rate_limiter):
        """Test sliding window rate limiting"""
        rate_limiter.config.strategy = RateLimitStrategy.SLIDING_WINDOW
        rate_limiter.config.requests_per_minute = 5  # Low limit for testing
        identifier = "user1"

        # Make requests within limit
        for _ in range(3):
            allowed, _ = await rate_limiter.check_rate_limit(identifier)
            assert allowed is True

        # Check history was recorded
        assert identifier in rate_limiter.request_history
        assert len(rate_limiter.request_history[identifier]) == 3

    @pytest.mark.asyncio
    async def test_check_rate_limit_sliding_window_exceeded(self, rate_limiter):
        """Test sliding window limit exceeded"""
        rate_limiter.config.strategy = RateLimitStrategy.SLIDING_WINDOW
        rate_limiter.config.requests_per_minute = 3
        identifier = "user1"

        # Fill the bucket first (this should work)
        bucket = rate_limiter.buckets.get(identifier)
        if bucket is None:
            await rate_limiter.check_rate_limit(identifier)
            bucket = rate_limiter.buckets[identifier]

        # Manually add requests to history to exceed minute limit
        now = time.time()
        rate_limiter.request_history[identifier].extend([now] * 4)  # Exceed limit of 3

        # Next request should be rejected due to sliding window
        allowed, _ = await rate_limiter.check_rate_limit(identifier)
        assert allowed is False

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, rate_limiter):
        """Test cleanup of expired data"""
        # Use a fixed time for consistent testing
        fixed_time = 1000000.0

        # Mock time consistently for all operations
        with patch("music21_mcp.rate_limiter.time.time", return_value=fixed_time):
            # Add some test data
            identifier = "user1"
            await rate_limiter.check_rate_limit(identifier)

            # Add old time that should be cleaned up (put it at the front since cleanup removes from front)
            old_time = (
                fixed_time - 100000
            )  # More than 24 hours ago (86400 seconds in a day)
            rate_limiter.request_history[identifier].appendleft(old_time)

            # Add old bucket
            old_bucket = TokenBucket(10, 1.0)
            old_bucket.last_update = fixed_time - 7200  # 2 hours ago
            rate_limiter.buckets["old_user"] = old_bucket

            await rate_limiter.cleanup_expired()

        # Old entries should be cleaned up
        assert old_time not in rate_limiter.request_history[identifier]
        assert "old_user" not in rate_limiter.buckets

    @pytest.mark.asyncio
    async def test_cleanup_empty_histories(self, rate_limiter):
        """Test cleanup removes empty history entries"""
        # Add empty history
        rate_limiter.request_history["empty_user"] = (
            rate_limiter.request_history.default_factory()
        )

        await rate_limiter.cleanup_expired()

        # Empty history should be removed
        assert "empty_user" not in rate_limiter.request_history

    @pytest.mark.asyncio
    async def test_start_cleanup_task(self, rate_limiter):
        """Test starting cleanup task"""
        assert rate_limiter._cleanup_task is None

        rate_limiter.start_cleanup_task()

        assert rate_limiter._cleanup_task is not None
        assert not rate_limiter._cleanup_task.done()

        # Cleanup
        rate_limiter.stop_cleanup_task()

    @pytest.mark.asyncio
    async def test_start_cleanup_task_idempotent(self, rate_limiter):
        """Test starting cleanup task multiple times"""
        rate_limiter.start_cleanup_task()
        first_task = rate_limiter._cleanup_task

        rate_limiter.start_cleanup_task()
        second_task = rate_limiter._cleanup_task

        # Should be the same task
        assert first_task is second_task

        # Cleanup
        rate_limiter.stop_cleanup_task()

    @pytest.mark.asyncio
    async def test_stop_cleanup_task(self, rate_limiter):
        """Test stopping cleanup task"""
        rate_limiter.start_cleanup_task()
        assert rate_limiter._cleanup_task is not None

        rate_limiter.stop_cleanup_task()

        assert rate_limiter._cleanup_task is None

    def test_stop_cleanup_task_when_none(self, rate_limiter):
        """Test stopping cleanup task when none exists"""
        assert rate_limiter._cleanup_task is None

        # Should not raise exception
        rate_limiter.stop_cleanup_task()

        assert rate_limiter._cleanup_task is None

    @pytest.mark.asyncio
    async def test_get_metadata(self, rate_limiter):
        """Test _get_metadata method"""
        bucket = TokenBucket(10.0, 2.0)
        bucket.tokens = 7.0

        metadata = rate_limiter._get_metadata(bucket, "/test")

        assert metadata["limit"] == 10
        assert metadata["remaining"] == 7
        assert metadata["endpoint"] == "/test"
        assert "reset" in metadata
        assert isinstance(metadata["reset"], int)

    @pytest.mark.asyncio
    async def test_get_metadata_with_retry_after(self, rate_limiter):
        """Test _get_metadata with retry_after calculation"""
        bucket = TokenBucket(10.0, 2.0)
        bucket.tokens = -1.0  # Negative tokens to ensure retry_after > 0

        metadata = rate_limiter._get_metadata(bucket)

        assert metadata["remaining"] == -1  # Floor to int
        assert metadata["retry_after"] is not None
        assert isinstance(metadata["retry_after"], int)
        assert metadata["retry_after"] > 0

    @pytest.mark.asyncio
    async def test_cleanup_loop_exception_handling(self, rate_limiter):
        """Test cleanup loop handles exceptions"""
        # Start cleanup task
        rate_limiter.start_cleanup_task()

        # Mock cleanup_expired to raise exception
        with (
            patch.object(
                rate_limiter, "cleanup_expired", side_effect=Exception("cleanup error")
            ),
            patch("asyncio.sleep", side_effect=[None, asyncio.CancelledError()]),
            contextlib.suppress(asyncio.CancelledError),
        ):
            # Mock sleep to return quickly and avoid infinite loop
            await rate_limiter._cleanup_loop()

        # Should handle exception gracefully (test that it doesn't crash)


class TestRateLimitMiddleware:
    """Test RateLimitMiddleware class"""

    @pytest.fixture
    def middleware(self):
        """Create middleware instance"""
        config = RateLimitConfig(requests_per_minute=60, burst_size=5)
        return RateLimitMiddleware(config)

    @pytest.fixture
    def mock_request(self):
        """Create mock request"""
        request = Mock(spec=Request)
        request.client = Mock()
        request.client.host = "127.0.0.1"
        request.url = Mock()
        request.url.path = "/test"
        request.headers = {}
        return request

    @pytest.fixture
    def mock_call_next(self):
        """Create mock call_next function"""

        async def call_next(request):
            response = Mock(spec=Response)
            response.headers = {}
            return response

        return call_next

    @pytest.mark.asyncio
    async def test_middleware_initialization(self, middleware):
        """Test middleware initialization"""
        assert isinstance(middleware.config, RateLimitConfig)
        assert isinstance(middleware.limiter, RateLimiter)
        # Cleanup task should be None initially (started on first use)
        assert middleware.limiter._cleanup_task is None

    @pytest.mark.asyncio
    async def test_middleware_initialization_default_config(self):
        """Test middleware with default config"""
        middleware = RateLimitMiddleware()

        assert isinstance(middleware.config, RateLimitConfig)
        assert middleware.config.requests_per_minute == 60  # Default value

    @pytest.mark.asyncio
    async def test_middleware_allowed_request(
        self, middleware, mock_request, mock_call_next
    ):
        """Test middleware allows request within limits"""
        response = await middleware(mock_request, mock_call_next)

        # Should pass through to next handler
        assert hasattr(response, "headers")
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

    @pytest.mark.asyncio
    async def test_middleware_rate_limited(
        self, middleware, mock_request, mock_call_next
    ):
        """Test middleware blocks request when rate limited"""
        # Exhaust rate limit
        for _ in range(5):  # burst_size = 5
            await middleware(mock_request, mock_call_next)

        # Next request should be blocked
        response = await middleware(mock_request, mock_call_next)

        assert isinstance(response, JSONResponse)
        assert response.status_code == 429

    @pytest.mark.asyncio
    async def test_middleware_uses_api_key(
        self, middleware, mock_request, mock_call_next
    ):
        """Test middleware uses API key as identifier"""
        mock_request.headers = {"X-API-Key": "test-key-123"}

        response = await middleware(mock_request, mock_call_next)

        # Should use API key as identifier (check by making sure bucket exists)
        assert "test-key-123" in middleware.limiter.buckets

    @pytest.mark.asyncio
    async def test_middleware_uses_ip_fallback(
        self, middleware, mock_request, mock_call_next
    ):
        """Test middleware falls back to IP when no API key"""
        mock_request.headers = {}
        mock_request.client.host = "192.168.1.1"

        response = await middleware(mock_request, mock_call_next)

        # Should use IP as identifier
        assert "192.168.1.1" in middleware.limiter.buckets

    @pytest.mark.asyncio
    async def test_middleware_no_client(self, middleware, mock_call_next):
        """Test middleware handles missing client info"""
        mock_request = Mock(spec=Request)
        mock_request.client = None
        mock_request.url = Mock()
        mock_request.url.path = "/test"
        mock_request.headers = {}

        response = await middleware(mock_request, mock_call_next)

        # Should use "unknown" as identifier
        assert "unknown" in middleware.limiter.buckets

    @pytest.mark.asyncio
    async def test_middleware_rate_limit_headers(
        self, middleware, mock_request, mock_call_next
    ):
        """Test middleware sets correct rate limit headers"""
        # Make request that gets rate limited
        for _ in range(5):  # Use up burst
            await middleware(mock_request, mock_call_next)

        response = await middleware(mock_request, mock_call_next)

        assert response.status_code == 429
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert response.headers["X-RateLimit-Remaining"] == "0"
        assert "Retry-After" in response.headers

    @pytest.mark.asyncio
    async def test_middleware_endpoint_specific_limit(
        self, middleware, mock_request, mock_call_next
    ):
        """Test middleware applies endpoint-specific limits"""
        mock_request.url.path = "/scores/import"  # Has specific limit in config

        response = await middleware(mock_request, mock_call_next)

        # Should create endpoint-specific bucket
        bucket_key = f"{mock_request.client.host}:/scores/import"
        assert bucket_key in middleware.limiter.buckets


class TestUtilityFunctions:
    """Test utility functions"""

    @pytest.mark.asyncio
    async def test_create_rate_limiter_default(self):
        """Test create_rate_limiter with defaults"""
        limiter = create_rate_limiter()

        assert isinstance(limiter, RateLimitMiddleware)
        assert limiter.config.requests_per_minute == 60
        assert limiter.config.requests_per_hour == 1000
        assert limiter.config.strategy == RateLimitStrategy.SLIDING_WINDOW

    @pytest.mark.asyncio
    async def test_create_rate_limiter_custom(self):
        """Test create_rate_limiter with custom values"""
        limiter = create_rate_limiter(
            requests_per_minute=30,
            requests_per_hour=500,
            strategy=RateLimitStrategy.TOKEN_BUCKET,
        )

        assert limiter.config.requests_per_minute == 30
        assert limiter.config.requests_per_hour == 500
        assert limiter.config.strategy == RateLimitStrategy.TOKEN_BUCKET


class TestRateLimitDecorator:
    """Test rate_limit decorator"""

    def test_decorator_basic(self):
        """Test basic decorator usage"""

        @rate_limit(requests_per_minute=10)
        async def test_endpoint(request):
            return "success"

        # Should have rate limit data attribute
        assert hasattr(test_endpoint, "__wrapped__")

    @pytest.mark.asyncio
    async def test_decorator_allows_request(self):
        """Test decorator allows request within limits"""

        @rate_limit(requests_per_minute=10)
        async def test_endpoint(request):
            return "success"

        mock_request = Mock()
        mock_request.client = Mock()
        mock_request.client.host = "127.0.0.1"

        result = await test_endpoint(mock_request)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_decorator_blocks_excess_requests(self):
        """Test decorator blocks excess requests"""

        @rate_limit(requests_per_minute=2)  # Very low limit
        async def test_endpoint(request):
            return "success"

        mock_request = Mock()
        mock_request.client = Mock()
        mock_request.client.host = "127.0.0.1"

        # First two requests should work
        await test_endpoint(mock_request)
        await test_endpoint(mock_request)

        # Third should be blocked
        with pytest.raises(HTTPException) as exc_info:
            await test_endpoint(mock_request)

        assert exc_info.value.status_code == 429
        assert "Rate limit exceeded" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_decorator_with_cost(self):
        """Test decorator with custom cost"""

        @rate_limit(requests_per_minute=10, cost=5)
        async def expensive_endpoint(request):
            return "success"

        mock_request = Mock()
        mock_request.client = Mock()
        mock_request.client.host = "127.0.0.1"

        # Two requests should exhaust the limit (2 * 5 = 10)
        await expensive_endpoint(mock_request)
        await expensive_endpoint(mock_request)

        # Third should be blocked
        with pytest.raises(HTTPException):
            await expensive_endpoint(mock_request)

    @pytest.mark.asyncio
    async def test_decorator_with_key_func(self):
        """Test decorator with custom key function"""

        def custom_key_func(request):
            return request.headers.get("User-ID", "anonymous")

        @rate_limit(requests_per_minute=2, key_func=custom_key_func)
        async def test_endpoint(request):
            return "success"

        # Create requests with different User-IDs
        mock_request1 = Mock()
        mock_request1.headers = {"User-ID": "user1"}
        mock_request1.client = Mock()
        mock_request1.client.host = "127.0.0.1"

        mock_request2 = Mock()
        mock_request2.headers = {"User-ID": "user2"}
        mock_request2.client = Mock()
        mock_request2.client.host = "127.0.0.1"

        # Each user should have separate limits
        await test_endpoint(mock_request1)
        await test_endpoint(mock_request1)
        await test_endpoint(mock_request2)  # Should still work (different user)

        # user1 should be blocked
        with pytest.raises(HTTPException):
            await test_endpoint(mock_request1)

    @pytest.mark.asyncio
    async def test_decorator_no_client(self):
        """Test decorator handles missing client"""

        @rate_limit(requests_per_minute=10)
        async def test_endpoint(request):
            return "success"

        mock_request = Mock()
        mock_request.client = None

        result = await test_endpoint(mock_request)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_decorator_time_reset(self):
        """Test decorator resets after time window"""

        @rate_limit(requests_per_minute=1)
        async def test_endpoint(request):
            return "success"

        mock_request = Mock()
        mock_request.client = Mock()
        mock_request.client.host = "127.0.0.1"

        # First request should work
        await test_endpoint(mock_request)

        # Mock time passage to reset the window
        current_time = time.time()
        with patch("time.time", return_value=current_time + 61):
            # Should work again after reset
            result = await test_endpoint(mock_request)
            assert result == "success"

    @pytest.mark.asyncio
    async def test_decorator_cleanup_large_data(self):
        """Test decorator cleans up when data gets large"""

        @rate_limit(requests_per_minute=10)
        async def test_endpoint(request):
            return "success"

        # Fill up rate limit data with many entries
        test_endpoint._rate_limit_data = {}
        for i in range(1001):  # More than 1000 limit
            test_endpoint._rate_limit_data[f"user{i}"] = {
                "count": 1,
                "reset": time.time() - 61,  # Expired
            }

        mock_request = Mock()
        mock_request.client = Mock()
        mock_request.client.host = "127.0.0.1"

        # Should trigger cleanup
        await test_endpoint(mock_request)

        # Should have cleaned up expired entries
        assert len(test_endpoint._rate_limit_data) < 1001


class TestIntegration:
    """Integration tests"""

    @pytest.mark.asyncio
    async def test_full_rate_limiting_flow(self):
        """Test complete rate limiting flow"""
        # Create limiter with low limits for testing
        config = RateLimitConfig(
            requests_per_minute=3,
            burst_size=2,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
        )
        limiter = RateLimiter(config)

        identifier = "integration_test_user"

        # First requests should succeed
        for i in range(2):
            allowed, metadata = await limiter.check_rate_limit(identifier)
            assert allowed is True
            assert metadata["remaining"] == 2 - i - 1

        # Should be blocked at burst limit
        allowed, metadata = await limiter.check_rate_limit(identifier)
        assert allowed is False
        assert metadata["remaining"] == 0

    @pytest.mark.asyncio
    async def test_middleware_integration(self):
        """Test middleware integration"""
        middleware = RateLimitMiddleware(
            RateLimitConfig(burst_size=1)  # Very restrictive
        )

        mock_request = Mock(spec=Request)
        mock_request.client = Mock()
        mock_request.client.host = "test-integration"
        mock_request.url = Mock()
        mock_request.url.path = "/integration-test"
        mock_request.headers = {}

        async def mock_next(request):
            response = Mock(spec=Response)
            response.headers = {}
            return response

        # First request should work
        response = await middleware(mock_request, mock_next)
        assert hasattr(response, "headers")

        # Second request should be blocked
        response = await middleware(mock_request, mock_next)
        assert isinstance(response, JSONResponse)
        assert response.status_code == 429

    def teardown_method(self):
        """Cleanup after each test"""
        # Clean up any lingering tasks
        import asyncio

        try:
            tasks = [t for t in asyncio.all_tasks() if not t.done()]
            for task in tasks:
                if "cleanup" in str(task):
                    task.cancel()
        except RuntimeError:
            # No running event loop, nothing to clean up
            pass
