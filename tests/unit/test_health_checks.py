"""
Comprehensive tests for health_checks module to boost coverage to 76%+

Tests all components:
- HealthStatus enum
- HealthCheckResult class
- HealthChecker class and all its methods
- Singleton pattern and convenience functions
"""

import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from music21 import chord, key

from music21_mcp.health_checks import (
    HealthChecker,
    HealthCheckResult,
    HealthStatus,
    get_health_checker,
    health_check,
    liveness_check,
    readiness_check,
)


class TestHealthStatus:
    """Test HealthStatus enum"""

    def test_health_status_values(self):
        """Test enum values"""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"

    def test_health_status_comparison(self):
        """Test enum comparison"""
        assert HealthStatus.HEALTHY != HealthStatus.DEGRADED
        assert HealthStatus.DEGRADED != HealthStatus.UNHEALTHY
        assert str(HealthStatus.HEALTHY) == "HealthStatus.HEALTHY"


class TestHealthCheckResult:
    """Test HealthCheckResult class"""

    def test_result_initialization_minimal(self):
        """Test minimal initialization"""
        result = HealthCheckResult("test", HealthStatus.HEALTHY)

        assert result.name == "test"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == ""
        assert result.details == {}
        assert result.duration_ms is None
        assert isinstance(result.timestamp, str)

    def test_result_initialization_full(self):
        """Test full initialization"""
        details = {"key": "value"}
        result = HealthCheckResult(
            name="full_test",
            status=HealthStatus.DEGRADED,
            message="Test message",
            details=details,
            duration_ms=123.45,
        )

        assert result.name == "full_test"
        assert result.status == HealthStatus.DEGRADED
        assert result.message == "Test message"
        assert result.details == details
        assert result.duration_ms == 123.45

    def test_result_to_dict(self):
        """Test conversion to dictionary"""
        result = HealthCheckResult(
            name="dict_test",
            status=HealthStatus.UNHEALTHY,
            message="Error occurred",
            details={"error_code": 500},
            duration_ms=500.0,
        )

        result_dict = result.to_dict()

        assert result_dict["name"] == "dict_test"
        assert result_dict["status"] == "unhealthy"
        assert result_dict["message"] == "Error occurred"
        assert result_dict["details"] == {"error_code": 500}
        assert result_dict["duration_ms"] == 500.0
        assert "timestamp" in result_dict

    def test_timestamp_format(self):
        """Test timestamp is in ISO format"""
        result = HealthCheckResult("timestamp_test", HealthStatus.HEALTHY)

        # Should be parseable as ISO datetime
        parsed_time = datetime.fromisoformat(result.timestamp.replace("Z", "+00:00"))
        assert isinstance(parsed_time, datetime)
        assert parsed_time.tzinfo is not None


class TestHealthChecker:
    """Test HealthChecker class"""

    @pytest.fixture
    def health_checker(self):
        """Create a HealthChecker instance"""
        return HealthChecker(
            memory_threshold_percent=80.0,
            cpu_threshold_percent=90.0,
            response_time_threshold_ms=5000.0,
        )

    def test_health_checker_initialization(self, health_checker):
        """Test HealthChecker initialization"""
        assert health_checker.memory_threshold == 80.0
        assert health_checker.cpu_threshold == 90.0
        assert health_checker.response_time_threshold == 5000.0
        assert health_checker.check_history == []
        assert health_checker.last_check_time is None
        assert health_checker.request_count == 0
        assert health_checker.error_count == 0
        assert health_checker.total_response_time_ms == 0.0

    @pytest.mark.asyncio
    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_percent")
    @patch("psutil.disk_usage")
    @patch("psutil.cpu_count")
    async def test_check_system_resources_healthy(
        self,
        mock_cpu_count,
        mock_disk_usage,
        mock_cpu_percent,
        mock_virtual_memory,
        health_checker,
    ):
        """Test system resources check - healthy state"""
        # Mock healthy system
        mock_virtual_memory.return_value = MagicMock(
            percent=50.0,
            available=2**30,  # 50% memory, 1GB available
        )
        mock_cpu_percent.return_value = 30.0  # 30% CPU
        mock_cpu_count.return_value = 8
        mock_disk_usage.return_value = MagicMock(
            percent=40.0,
            free=10 * (2**30),  # 40% disk, 10GB free
        )

        result = await health_checker.check_system_resources()

        assert result.name == "system_resources"
        assert result.status == HealthStatus.HEALTHY
        assert "within normal limits" in result.message
        assert result.details["memory_percent"] == 50.0
        assert result.details["cpu_percent"] == 30.0
        assert result.details["cpu_count"] == 8
        assert result.duration_ms is not None
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_percent")
    @patch("psutil.disk_usage")
    @patch("psutil.cpu_count")
    async def test_check_system_resources_degraded(
        self,
        mock_cpu_count,
        mock_disk_usage,
        mock_cpu_percent,
        mock_virtual_memory,
        health_checker,
    ):
        """Test system resources check - degraded state"""
        # Mock degraded system (high but not critical)
        mock_virtual_memory.return_value = MagicMock(
            percent=65.0, available=512 * 1024 * 1024
        )  # 65% memory
        mock_cpu_percent.return_value = 75.0  # 75% CPU (80% of 90% threshold)
        mock_cpu_count.return_value = 4
        mock_disk_usage.return_value = MagicMock(percent=60.0, free=5 * (2**30))

        result = await health_checker.check_system_resources()

        assert result.name == "system_resources"
        assert result.status == HealthStatus.DEGRADED
        assert "Moderate resource usage" in result.message

    @pytest.mark.asyncio
    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_percent")
    @patch("psutil.disk_usage")
    @patch("psutil.cpu_count")
    async def test_check_system_resources_unhealthy(
        self,
        mock_cpu_count,
        mock_disk_usage,
        mock_cpu_percent,
        mock_virtual_memory,
        health_checker,
    ):
        """Test system resources check - unhealthy state"""
        # Mock unhealthy system (over thresholds)
        mock_virtual_memory.return_value = MagicMock(
            percent=85.0, available=100 * 1024 * 1024
        )  # 85% memory
        mock_cpu_percent.return_value = 95.0  # 95% CPU (over 90% threshold)
        mock_cpu_count.return_value = 2
        mock_disk_usage.return_value = MagicMock(percent=95.0, free=1 * (2**30))

        result = await health_checker.check_system_resources()

        assert result.name == "system_resources"
        assert result.status == HealthStatus.UNHEALTHY
        assert "High resource usage" in result.message

    @pytest.mark.asyncio
    @patch("psutil.virtual_memory")
    async def test_check_system_resources_exception(
        self, mock_virtual_memory, health_checker
    ):
        """Test system resources check with exception"""
        mock_virtual_memory.side_effect = Exception("psutil error")

        result = await health_checker.check_system_resources()

        assert result.name == "system_resources"
        assert result.status == HealthStatus.UNHEALTHY
        assert "Failed to check system resources" in result.message

    @pytest.mark.asyncio
    @patch("music21.stream.Score")
    @patch("music21.chord.Chord")
    @patch("music21.converter.parse")
    async def test_check_music21_functionality_success(
        self, mock_parse, mock_chord, mock_score, health_checker
    ):
        """Test music21 functionality check - success"""
        # Mock successful music21 operations
        mock_score_instance = MagicMock()
        mock_part = MagicMock()
        mock_key = MagicMock()
        mock_key.__str__ = MagicMock(return_value="C major")

        mock_score.return_value = mock_score_instance
        mock_score_instance.append = MagicMock()
        mock_score_instance.analyze.return_value = mock_key
        mock_score_instance.write.return_value = b"midi_data"

        mock_chord.return_value = MagicMock()
        mock_parse.return_value = MagicMock()

        with patch("music21.stream.Part", return_value=mock_part):
            result = await health_checker.check_music21_functionality()

        assert result.name == "music21_functionality"
        assert result.status == HealthStatus.HEALTHY
        assert "functionality operational" in result.message
        assert "operations_tested" in result.details
        assert len(result.details["operations_tested"]) == 4

    @pytest.mark.asyncio
    async def test_check_music21_functionality_slow(self, health_checker):
        """Test music21 functionality check - slow response"""
        # Set a very low threshold to trigger slow response
        health_checker.response_time_threshold = 0.001  # 1ms threshold

        with patch("time.time", side_effect=[0, 0.1]):  # 100ms duration
            with patch("music21.stream.Score"):
                with patch("music21.chord.Chord"):
                    with patch("music21.converter.parse"):
                        result = await health_checker.check_music21_functionality()

        assert result.name == "music21_functionality"
        assert result.status == HealthStatus.DEGRADED
        assert "operations slow" in result.message

    @pytest.mark.asyncio
    async def test_check_music21_functionality_exception(self, health_checker):
        """Test music21 functionality check with exception"""
        with patch("music21.stream.Score", side_effect=Exception("music21 error")):
            result = await health_checker.check_music21_functionality()

        assert result.name == "music21_functionality"
        assert result.status == HealthStatus.UNHEALTHY
        assert "Music21 operations failed" in result.message

    @pytest.mark.asyncio
    async def test_check_cache_systems_success(self, health_checker):
        """Test cache systems check - success"""
        mock_optimizer = MagicMock()
        mock_optimizer.get_cached_roman_numeral.return_value = "I"
        mock_optimizer.get_performance_metrics.return_value = {
            "current_metrics": {"cache_stats": {"hit_rate": 0.8}}
        }
        mock_optimizer.roman_cache = {"test": "value"}
        mock_optimizer.key_cache = {}
        mock_optimizer.chord_analysis_cache = {}
        mock_optimizer.shutdown = MagicMock()

        with patch(
            "music21_mcp.health_checks.PerformanceOptimizer",
            return_value=mock_optimizer,
        ):
            with patch("music21.chord.Chord"):
                with patch("music21.key.Key"):
                    result = await health_checker.check_cache_systems()

        assert result.name == "cache_systems"
        assert result.status == HealthStatus.HEALTHY
        assert "80.0% hit rate" in result.message
        assert "cache_sizes" in result.details
        mock_optimizer.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_cache_systems_no_hits(self, health_checker):
        """Test cache systems check - no cache hits"""
        mock_optimizer = MagicMock()
        mock_optimizer.get_cached_roman_numeral.return_value = "I"
        mock_optimizer.get_performance_metrics.return_value = {
            "current_metrics": {"cache_stats": {"hit_rate": 0}}
        }
        mock_optimizer.roman_cache = {}
        mock_optimizer.key_cache = {}
        mock_optimizer.chord_analysis_cache = {}
        mock_optimizer.shutdown = MagicMock()

        with patch(
            "music21_mcp.health_checks.PerformanceOptimizer",
            return_value=mock_optimizer,
        ):
            with patch("music21.chord.Chord"):
                with patch("music21.key.Key"):
                    result = await health_checker.check_cache_systems()

        assert result.name == "cache_systems"
        assert result.status == HealthStatus.DEGRADED
        assert "no hits recorded" in result.message

    @pytest.mark.asyncio
    async def test_check_cache_systems_exception(self, health_checker):
        """Test cache systems check with exception"""
        with patch(
            "music21_mcp.health_checks.PerformanceOptimizer",
            side_effect=ImportError("module not found"),
        ):
            result = await health_checker.check_cache_systems()

        assert result.name == "cache_systems"
        assert result.status == HealthStatus.DEGRADED
        assert "Cache system check partial failure" in result.message

    @pytest.mark.asyncio
    async def test_check_dependencies_success(self, health_checker):
        """Test dependencies check - all available"""
        with patch("builtins.__import__") as mock_import:
            # Mock successful imports
            mock_module = MagicMock()
            mock_module.__version__ = "1.0.0"
            mock_import.return_value = mock_module

            result = await health_checker.check_dependencies()

        assert result.name == "dependencies"
        assert result.status == HealthStatus.HEALTHY
        assert "All dependencies available" in result.message
        assert "dependencies" in result.details

    @pytest.mark.asyncio
    async def test_check_dependencies_missing(self, health_checker):
        """Test dependencies check - some missing"""

        def mock_import(name):
            if name == "music21":
                raise ImportError("Module not found")
            mock_module = MagicMock()
            mock_module.__version__ = "1.0.0"
            return mock_module

        with patch("builtins.__import__", side_effect=mock_import):
            result = await health_checker.check_dependencies()

        assert result.name == "dependencies"
        assert result.status == HealthStatus.UNHEALTHY
        assert "Missing dependencies: music21" in result.message

    @pytest.mark.asyncio
    async def test_check_dependencies_no_version(self, health_checker):
        """Test dependencies check - modules without version"""
        with patch("builtins.__import__") as mock_import:
            mock_module = MagicMock()
            del mock_module.__version__  # No version attribute
            mock_import.return_value = mock_module

            result = await health_checker.check_dependencies()

        assert result.name == "dependencies"
        assert result.status == HealthStatus.HEALTHY
        # Should show "installed" for modules without version
        for dep_value in result.details["dependencies"].values():
            assert dep_value == "installed"

    @pytest.mark.asyncio
    async def test_check_dependencies_exception(self, health_checker):
        """Test dependencies check with exception"""
        with patch("builtins.__import__", side_effect=Exception("import system error")):
            result = await health_checker.check_dependencies()

        assert result.name == "dependencies"
        assert result.status == HealthStatus.UNHEALTHY
        assert "Failed to check dependencies" in result.message

    @pytest.mark.asyncio
    async def test_check_performance_metrics_no_requests(self, health_checker):
        """Test performance metrics with no recorded requests"""
        result = await health_checker.check_performance_metrics()

        assert result.name == "performance_metrics"
        assert result.status == HealthStatus.HEALTHY
        assert "within normal range" in result.message
        assert result.details["request_count"] == 0
        assert result.details["error_count"] == 0
        assert result.details["error_rate"] == 0
        assert result.details["avg_response_time_ms"] == 0

    @pytest.mark.asyncio
    async def test_check_performance_metrics_high_error_rate(self, health_checker):
        """Test performance metrics with high error rate"""
        # Record some requests with high error rate
        health_checker.request_count = 10
        health_checker.error_count = 2  # 20% error rate
        health_checker.total_response_time_ms = 1000

        result = await health_checker.check_performance_metrics()

        assert result.name == "performance_metrics"
        assert result.status == HealthStatus.UNHEALTHY
        assert "High error rate: 20.0%" in result.message

    @pytest.mark.asyncio
    async def test_check_performance_metrics_moderate_error_rate(self, health_checker):
        """Test performance metrics with moderate error rate"""
        # Record some requests with moderate error rate
        health_checker.request_count = 100
        health_checker.error_count = 7  # 7% error rate
        health_checker.total_response_time_ms = 50000  # 500ms average

        result = await health_checker.check_performance_metrics()

        assert result.name == "performance_metrics"
        assert result.status == HealthStatus.DEGRADED
        assert "Moderate error rate: 7.0%" in result.message

    @pytest.mark.asyncio
    async def test_check_performance_metrics_slow_response(self, health_checker):
        """Test performance metrics with slow response times"""
        # Record requests with slow response times
        health_checker.request_count = 10
        health_checker.error_count = 0
        health_checker.total_response_time_ms = (
            70000  # 7000ms average (over 5000ms threshold)
        )

        result = await health_checker.check_performance_metrics()

        assert result.name == "performance_metrics"
        assert result.status == HealthStatus.DEGRADED
        assert "Slow response times: 7000.0ms" in result.message

    @pytest.mark.asyncio
    async def test_check_performance_metrics_exception(self, health_checker):
        """Test performance metrics check with exception"""
        # Force an exception by mocking time.time
        with patch("time.time", side_effect=Exception("time error")):
            result = await health_checker.check_performance_metrics()

        assert result.name == "performance_metrics"
        assert result.status == HealthStatus.DEGRADED
        assert "Failed to check performance metrics" in result.message

    def test_record_request_success(self, health_checker):
        """Test recording successful request"""
        health_checker.record_request(100.0, success=True)

        assert health_checker.request_count == 1
        assert health_checker.total_response_time_ms == 100.0
        assert health_checker.error_count == 0

    def test_record_request_failure(self, health_checker):
        """Test recording failed request"""
        health_checker.record_request(500.0, success=False)

        assert health_checker.request_count == 1
        assert health_checker.total_response_time_ms == 500.0
        assert health_checker.error_count == 1

    def test_record_multiple_requests(self, health_checker):
        """Test recording multiple requests"""
        health_checker.record_request(100.0, success=True)
        health_checker.record_request(200.0, success=False)
        health_checker.record_request(150.0, success=True)

        assert health_checker.request_count == 3
        assert health_checker.total_response_time_ms == 450.0
        assert health_checker.error_count == 1

    @pytest.mark.asyncio
    async def test_check_all_success(self, health_checker):
        """Test check_all method with all checks passing"""
        with patch.object(health_checker, "check_system_resources") as mock_system:
            with patch.object(
                health_checker, "check_music21_functionality"
            ) as mock_music21:
                with patch.object(health_checker, "check_cache_systems") as mock_cache:
                    with patch.object(
                        health_checker, "check_dependencies"
                    ) as mock_deps:
                        with patch.object(
                            health_checker, "check_performance_metrics"
                        ) as mock_perf:
                            # Mock all checks as healthy
                            mock_system.return_value = HealthCheckResult(
                                "system", HealthStatus.HEALTHY
                            )
                            mock_music21.return_value = HealthCheckResult(
                                "music21", HealthStatus.HEALTHY
                            )
                            mock_cache.return_value = HealthCheckResult(
                                "cache", HealthStatus.HEALTHY
                            )
                            mock_deps.return_value = HealthCheckResult(
                                "deps", HealthStatus.HEALTHY
                            )
                            mock_perf.return_value = HealthCheckResult(
                                "perf", HealthStatus.HEALTHY
                            )

                            result = await health_checker.check_all()

        assert result["status"] == "healthy"
        assert len(result["checks"]) == 5
        assert "timestamp" in result
        assert "duration_ms" in result
        assert "system" in result
        assert len(health_checker.check_history) == 5

    @pytest.mark.asyncio
    async def test_check_all_with_degraded(self, health_checker):
        """Test check_all method with degraded checks"""
        with patch.object(health_checker, "check_system_resources") as mock_system:
            with patch.object(
                health_checker, "check_music21_functionality"
            ) as mock_music21:
                with patch.object(health_checker, "check_cache_systems") as mock_cache:
                    with patch.object(
                        health_checker, "check_dependencies"
                    ) as mock_deps:
                        with patch.object(
                            health_checker, "check_performance_metrics"
                        ) as mock_perf:
                            # Some checks degraded
                            mock_system.return_value = HealthCheckResult(
                                "system", HealthStatus.DEGRADED
                            )
                            mock_music21.return_value = HealthCheckResult(
                                "music21", HealthStatus.HEALTHY
                            )
                            mock_cache.return_value = HealthCheckResult(
                                "cache", HealthStatus.HEALTHY
                            )
                            mock_deps.return_value = HealthCheckResult(
                                "deps", HealthStatus.HEALTHY
                            )
                            mock_perf.return_value = HealthCheckResult(
                                "perf", HealthStatus.HEALTHY
                            )

                            result = await health_checker.check_all()

        assert result["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_check_all_with_unhealthy(self, health_checker):
        """Test check_all method with unhealthy checks"""
        with patch.object(health_checker, "check_system_resources") as mock_system:
            with patch.object(
                health_checker, "check_music21_functionality"
            ) as mock_music21:
                with patch.object(health_checker, "check_cache_systems") as mock_cache:
                    with patch.object(
                        health_checker, "check_dependencies"
                    ) as mock_deps:
                        with patch.object(
                            health_checker, "check_performance_metrics"
                        ) as mock_perf:
                            # Some checks unhealthy
                            mock_system.return_value = HealthCheckResult(
                                "system", HealthStatus.HEALTHY
                            )
                            mock_music21.return_value = HealthCheckResult(
                                "music21", HealthStatus.UNHEALTHY
                            )
                            mock_cache.return_value = HealthCheckResult(
                                "cache", HealthStatus.DEGRADED
                            )
                            mock_deps.return_value = HealthCheckResult(
                                "deps", HealthStatus.HEALTHY
                            )
                            mock_perf.return_value = HealthCheckResult(
                                "perf", HealthStatus.HEALTHY
                            )

                            result = await health_checker.check_all()

        assert result["status"] == "unhealthy"  # Unhealthy takes precedence

    @pytest.mark.asyncio
    async def test_check_all_with_exception(self, health_checker):
        """Test check_all method with check exception"""
        with patch.object(
            health_checker,
            "check_system_resources",
            side_effect=Exception("check failed"),
        ):
            with patch.object(
                health_checker, "check_music21_functionality"
            ) as mock_music21:
                with patch.object(health_checker, "check_cache_systems") as mock_cache:
                    with patch.object(
                        health_checker, "check_dependencies"
                    ) as mock_deps:
                        with patch.object(
                            health_checker, "check_performance_metrics"
                        ) as mock_perf:
                            mock_music21.return_value = HealthCheckResult(
                                "music21", HealthStatus.HEALTHY
                            )
                            mock_cache.return_value = HealthCheckResult(
                                "cache", HealthStatus.HEALTHY
                            )
                            mock_deps.return_value = HealthCheckResult(
                                "deps", HealthStatus.HEALTHY
                            )
                            mock_perf.return_value = HealthCheckResult(
                                "perf", HealthStatus.HEALTHY
                            )

                            result = await health_checker.check_all()

        assert result["status"] == "unhealthy"  # Exception causes unhealthy
        # Should have 5 results (4 good + 1 exception)
        assert len(result["checks"]) == 5
        # One check should be marked as unknown with error message
        error_checks = [c for c in result["checks"] if c["name"] == "unknown"]
        assert len(error_checks) == 1
        assert "Health check failed" in error_checks[0]["message"]

    @pytest.mark.asyncio
    async def test_check_all_history_trimming(self, health_checker):
        """Test that check history is trimmed to 100 entries"""
        # Fill up history beyond limit
        for i in range(105):
            health_checker.check_history.append(
                HealthCheckResult(f"test_{i}", HealthStatus.HEALTHY)
            )

        with patch.object(health_checker, "check_system_resources") as mock_system:
            with patch.object(
                health_checker, "check_music21_functionality"
            ) as mock_music21:
                with patch.object(health_checker, "check_cache_systems") as mock_cache:
                    with patch.object(
                        health_checker, "check_dependencies"
                    ) as mock_deps:
                        with patch.object(
                            health_checker, "check_performance_metrics"
                        ) as mock_perf:
                            # Mock all checks
                            mock_system.return_value = HealthCheckResult(
                                "system", HealthStatus.HEALTHY
                            )
                            mock_music21.return_value = HealthCheckResult(
                                "music21", HealthStatus.HEALTHY
                            )
                            mock_cache.return_value = HealthCheckResult(
                                "cache", HealthStatus.HEALTHY
                            )
                            mock_deps.return_value = HealthCheckResult(
                                "deps", HealthStatus.HEALTHY
                            )
                            mock_perf.return_value = HealthCheckResult(
                                "perf", HealthStatus.HEALTHY
                            )

                            await health_checker.check_all()

        # Should be trimmed to 100
        assert len(health_checker.check_history) == 100
        # Should keep the most recent ones
        assert health_checker.check_history[-1].name == "perf"

    @pytest.mark.asyncio
    async def test_get_readiness_success(self, health_checker):
        """Test readiness check - ready"""
        with patch.object(
            health_checker, "check_music21_functionality"
        ) as mock_music21:
            with patch.object(health_checker, "check_dependencies") as mock_deps:
                mock_music21.return_value = HealthCheckResult(
                    "music21", HealthStatus.HEALTHY
                )
                mock_deps.return_value = HealthCheckResult(
                    "deps", HealthStatus.DEGRADED
                )  # Degraded is OK

                result = await health_checker.get_readiness()

        assert result["ready"] is True
        assert len(result["checks"]) == 2

    @pytest.mark.asyncio
    async def test_get_readiness_not_ready(self, health_checker):
        """Test readiness check - not ready"""
        with patch.object(
            health_checker, "check_music21_functionality"
        ) as mock_music21:
            with patch.object(health_checker, "check_dependencies") as mock_deps:
                mock_music21.return_value = HealthCheckResult(
                    "music21", HealthStatus.UNHEALTHY
                )
                mock_deps.return_value = HealthCheckResult("deps", HealthStatus.HEALTHY)

                result = await health_checker.get_readiness()

        assert result["ready"] is False

    @pytest.mark.asyncio
    async def test_get_readiness_with_exception(self, health_checker):
        """Test readiness check with exception"""
        with patch.object(
            health_checker,
            "check_music21_functionality",
            side_effect=Exception("check failed"),
        ):
            with patch.object(health_checker, "check_dependencies") as mock_deps:
                mock_deps.return_value = HealthCheckResult("deps", HealthStatus.HEALTHY)

                result = await health_checker.get_readiness()

        assert result["ready"] is False  # Exception makes it not ready
        assert len(result["checks"]) == 2
        # One result should be the error
        error_results = [c for c in result["checks"] if "error" in c]
        assert len(error_results) == 1

    @pytest.mark.asyncio
    async def test_get_liveness_success(self, health_checker):
        """Test liveness check - alive"""
        with patch("music21.chord.Chord"):
            result = await health_checker.get_liveness()

        assert result["alive"] is True
        assert "timestamp" in result
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_get_liveness_failure(self, health_checker):
        """Test liveness check - not alive"""
        with patch(
            "music21.chord.Chord", side_effect=Exception("music21 not available")
        ):
            result = await health_checker.get_liveness()

        assert result["alive"] is False
        assert "error" in result
        assert "timestamp" in result
        assert "music21 not available" in result["error"]


class TestSingletonAndConvenience:
    """Test singleton pattern and convenience functions"""

    def test_get_health_checker_singleton(self):
        """Test singleton pattern"""
        checker1 = get_health_checker()
        checker2 = get_health_checker()

        assert checker1 is checker2  # Should be same instance
        assert isinstance(checker1, HealthChecker)

    @pytest.mark.asyncio
    async def test_health_check_convenience(self):
        """Test convenience health_check function"""
        with patch.object(HealthChecker, "check_all") as mock_check_all:
            mock_check_all.return_value = {"status": "healthy"}

            result = await health_check()

            assert result == {"status": "healthy"}
            mock_check_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_readiness_check_convenience(self):
        """Test convenience readiness_check function"""
        with patch.object(HealthChecker, "get_readiness") as mock_readiness:
            mock_readiness.return_value = {"ready": True}

            result = await readiness_check()

            assert result == {"ready": True}
            mock_readiness.assert_called_once()

    @pytest.mark.asyncio
    async def test_liveness_check_convenience(self):
        """Test convenience liveness_check function"""
        with patch.object(HealthChecker, "get_liveness") as mock_liveness:
            mock_liveness.return_value = {"alive": True}

            result = await liveness_check()

            assert result == {"alive": True}
            mock_liveness.assert_called_once()

    def teardown_method(self):
        """Clean up singleton between tests"""
        import music21_mcp.health_checks

        music21_mcp.health_checks._health_checker = None


class TestIntegration:
    """Integration tests with real dependencies where possible"""

    @pytest.mark.asyncio
    async def test_real_music21_functionality_check(self):
        """Test with real music21 (if available)"""
        health_checker = HealthChecker()

        # This should work with real music21
        result = await health_checker.check_music21_functionality()

        assert result.name == "music21_functionality"
        assert result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        # Should complete without exception

    @pytest.mark.asyncio
    async def test_real_dependencies_check(self):
        """Test with real dependencies"""
        health_checker = HealthChecker()

        result = await health_checker.check_dependencies()

        assert result.name == "dependencies"
        # Should find at least some dependencies
        assert "dependencies" in result.details
        assert len(result.details["dependencies"]) > 0

    def teardown_method(self):
        """Clean up singleton between tests"""
        import music21_mcp.health_checks

        music21_mcp.health_checks._health_checker = None
# ruff: noqa: SIM117
