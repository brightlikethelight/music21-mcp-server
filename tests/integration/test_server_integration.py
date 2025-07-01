"""
Integration tests for the Music21 MCP Server
Tests the server with all components working together
"""
import pytest
import asyncio
import time
import psutil
from unittest.mock import patch, Mock
from datetime import datetime

from music21_mcp.server import (
    ServerConfig, SimpleCircuitBreaker, SimpleRateLimiter,
    ScoreManager, initialize_tools, create_resilient_tool,
    health_check, cleanup_memory
)


class TestServerConfig:
    """Test server configuration"""
    
    def test_default_config(self):
        config = ServerConfig()
        assert config.max_scores == 100
        assert config.max_score_size_mb == 50
        assert config.memory_limit_mb == 2048
        assert config.gc_threshold_mb == 1500
        assert config.cache_ttl_seconds == 3600
        assert config.rate_limit_per_minute == 100
        assert config.circuit_breaker_threshold == 5
        assert config.circuit_breaker_timeout == 60


class TestSimpleCircuitBreaker:
    """Test circuit breaker implementation"""
    
    def test_circuit_breaker_opens_after_threshold(self):
        cb = SimpleCircuitBreaker("test", threshold=3, timeout=60)
        
        # Initial state should be closed
        assert cb.state == "closed"
        assert cb.can_execute()
        
        # Record failures
        for _ in range(3):
            cb.record_failure()
        
        # Should be open now
        assert cb.state == "open"
        assert not cb.can_execute()
    
    def test_circuit_breaker_half_open_after_timeout(self):
        cb = SimpleCircuitBreaker("test", threshold=2, timeout=0.1)
        
        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"
        
        # Wait for timeout
        time.sleep(0.2)
        
        # Should allow one attempt
        assert cb.can_execute()
        assert cb.state == "half_open"
    
    def test_circuit_breaker_closes_on_success(self):
        cb = SimpleCircuitBreaker("test", threshold=2, timeout=0.1)
        
        # Open and then half-open
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.2)
        cb.can_execute()
        
        # Success should close it
        cb.record_success()
        assert cb.state == "closed"
        assert cb.failures == 0


class TestSimpleRateLimiter:
    """Test rate limiter implementation"""
    
    def test_rate_limiter_allows_burst(self):
        rl = SimpleRateLimiter(rate=10, burst=5)
        
        # Should allow burst
        for _ in range(5):
            assert rl.acquire()
        
        # Should deny after burst
        assert not rl.acquire()
    
    def test_rate_limiter_refills_over_time(self):
        rl = SimpleRateLimiter(rate=10, burst=2)
        
        # Use all tokens
        assert rl.acquire()
        assert rl.acquire()
        assert not rl.acquire()
        
        # Wait for refill
        time.sleep(0.15)  # Should refill ~1.5 tokens
        
        # Should allow one more
        assert rl.acquire()


class TestScoreManager:
    """Test score manager functionality"""
    
    @pytest.mark.asyncio
    async def test_add_and_get_score(self):
        manager = ScoreManager(max_scores=10)
        
        # Mock score
        score = Mock()
        score.metadata.title = "Test Score"
        
        # Add score
        await manager.add_score("test_id", score)
        
        # Get score
        retrieved = await manager.get_score("test_id")
        assert retrieved == score
        assert "test_id" in manager.scores
    
    @pytest.mark.asyncio
    async def test_score_manager_capacity_limit(self):
        manager = ScoreManager(max_scores=2)
        
        # Add scores up to limit
        for i in range(3):
            score = Mock()
            score.metadata.title = f"Score {i}"
            await manager.add_score(f"score_{i}", score)
        
        # Should only have 2 scores (oldest removed)
        assert len(manager.scores) == 2
        assert "score_0" not in manager.scores
        assert "score_1" in manager.scores
        assert "score_2" in manager.scores
    
    @pytest.mark.asyncio
    async def test_score_manager_cleanup(self):
        manager = ScoreManager(max_scores=10)
        
        # Add old score
        score = Mock()
        score.metadata.title = "Old Score"
        await manager.add_score("old_score", score)
        
        # Manipulate access time to make it old
        manager.access_times["old_score"] = time.time() - 7200  # 2 hours ago
        
        # Run cleanup
        await manager._cleanup_old_scores()
        
        # Old score should be removed
        assert "old_score" not in manager.scores
    
    @pytest.mark.asyncio
    async def test_list_scores(self):
        manager = ScoreManager(max_scores=10)
        
        # Add multiple scores
        for i in range(3):
            score = Mock()
            score.metadata.title = f"Score {i}"
            await manager.add_score(f"score_{i}", score)
        
        # List scores
        scores = await manager.list_scores()
        assert len(scores) == 3
        assert all("id" in s and "title" in s for s in scores)


class TestHealthCheck:
    """Test health check functionality"""
    
    @pytest.mark.asyncio
    async def test_health_check_returns_metrics(self):
        # Mock score manager
        with patch('music21_mcp.server.score_manager') as mock_manager:
            mock_manager.scores = {"score1": Mock(), "score2": Mock()}
            
            result = await health_check()
            
            assert result["status"] == "healthy"
            assert "timestamp" in result
            assert "uptime_seconds" in result
            assert "memory" in result
            assert result["scores"]["count"] == 2
            assert "circuit_breakers" in result


class TestMemoryCleanup:
    """Test memory cleanup functionality"""
    
    @pytest.mark.asyncio
    async def test_cleanup_memory(self):
        # Mock score manager cleanup
        with patch('music21_mcp.server.score_manager._cleanup_old_scores') as mock_cleanup:
            mock_cleanup.return_value = asyncio.Future()
            mock_cleanup.return_value.set_result(None)
            
            result = await cleanup_memory()
            
            assert result["status"] == "success"
            assert "memory_before_mb" in result
            assert "memory_after_mb" in result
            assert "freed_mb" in result
            assert "scores_remaining" in result
            mock_cleanup.assert_called_once()


class TestToolInitialization:
    """Test tool initialization with resilience"""
    
    @patch('music21_mcp.server.mcp')
    def test_initialize_tools_registers_all(self, mock_mcp):
        # Mock the tool decorator
        mock_mcp.tool.return_value = lambda f: f
        
        # Initialize tools
        initialize_tools()
        
        # Should register 13 tools
        assert mock_mcp.tool.call_count == 13
        
        # Check tool names were registered
        registered_names = [call[1]['name'] for call in mock_mcp.tool.call_args_list]
        expected_tools = [
            'import_score', 'list_scores', 'analyze_key',
            'analyze_chords', 'get_score_info', 'export_score',
            'delete_score', 'analyze_harmony', 'analyze_voice_leading',
            'recognize_patterns', 'harmonize', 'generate_counterpoint',
            'imitate_style'
        ]
        for tool_name in expected_tools:
            assert tool_name in registered_names


class TestResilientToolWrapper:
    """Test resilient tool wrapper functionality"""
    
    @pytest.mark.asyncio
    async def test_resilient_tool_rate_limiting(self):
        # Create mocks
        tool = Mock()
        tool.name = "test_tool"
        tool.description = "Test tool"
        tool.execute = AsyncMock(return_value={"result": "success"})
        
        cb = SimpleCircuitBreaker("test", threshold=5, timeout=60)
        rl = SimpleRateLimiter(rate=1, burst=1)
        
        # Create resilient wrapper
        resilient = create_resilient_tool(tool, cb, rl)
        
        # First call should succeed
        result1 = await resilient()
        assert result1["result"] == "success"
        
        # Second call should be rate limited
        result2 = await resilient()
        assert "error" in result2
        assert "Rate limit" in result2["error"]
    
    @pytest.mark.asyncio
    async def test_resilient_tool_circuit_breaker(self):
        # Create mocks
        tool = Mock()
        tool.name = "test_tool"
        tool.description = "Test tool"
        tool.execute = AsyncMock(side_effect=Exception("Test error"))
        
        cb = SimpleCircuitBreaker("test", threshold=2, timeout=60)
        rl = SimpleRateLimiter(rate=100, burst=100)
        
        # Create resilient wrapper
        resilient = create_resilient_tool(tool, cb, rl)
        
        # Cause failures to open circuit
        for _ in range(2):
            result = await resilient()
            assert "error" in result
        
        # Circuit should be open now
        result = await resilient()
        assert "error" in result
        assert "temporarily unavailable" in result["error"]
    
    @pytest.mark.asyncio
    async def test_resilient_tool_success_path(self):
        # Create mocks
        tool = Mock()
        tool.name = "test_tool"
        tool.description = "Test tool"
        tool.execute = AsyncMock(return_value={"data": "test_data"})
        
        cb = SimpleCircuitBreaker("test", threshold=5, timeout=60)
        rl = SimpleRateLimiter(rate=100, burst=100)
        
        # Create resilient wrapper
        resilient = create_resilient_tool(tool, cb, rl)
        
        # Should succeed
        result = await resilient(param1="value1", param2="value2")
        assert result["data"] == "test_data"
        tool.execute.assert_called_once_with(param1="value1", param2="value2")


class TestServerIntegration:
    """Full server integration tests"""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_server_startup_sequence(self):
        """Test that server initializes correctly"""
        with patch('music21_mcp.server.mcp.run'):
            # Import should not raise errors
            from music21_mcp import server
            
            # Check components are initialized
            assert hasattr(server, 'config')
            assert hasattr(server, 'score_manager')
            assert hasattr(server, 'rate_limiter')
            assert hasattr(server, 'circuit_breakers')
            
            # Check circuit breakers
            assert 'import' in server.circuit_breakers
            assert 'analysis' in server.circuit_breakers
            assert 'export' in server.circuit_breakers
            assert 'generation' in server.circuit_breakers