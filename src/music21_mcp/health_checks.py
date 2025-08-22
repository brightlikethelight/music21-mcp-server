"""
Comprehensive Health Check System for Music21 MCP Server

Provides health monitoring endpoints and automated checks for:
- System resources (memory, CPU)
- Music21 functionality
- Cache systems
- Dependencies
- Performance metrics
"""

import asyncio
import logging
import os
import platform
import sys
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import psutil
from music21 import chord, converter, key, stream

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status levels"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthCheckResult:
    """Result of a health check"""

    def __init__(
        self,
        name: str,
        status: HealthStatus,
        message: str = "",
        details: dict[str, Any] | None = None,
        duration_ms: float | None = None,
    ):
        self.name = name
        self.status = status
        self.message = message
        self.details = details or {}
        self.duration_ms = duration_ms
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
        }


class HealthChecker:
    """Comprehensive health checking system"""

    def __init__(
        self,
        memory_threshold_percent: float = 80.0,
        cpu_threshold_percent: float = 90.0,
        response_time_threshold_ms: float = 5000.0,
    ):
        self.memory_threshold = memory_threshold_percent
        self.cpu_threshold = cpu_threshold_percent
        self.response_time_threshold = response_time_threshold_ms

        # Track health history
        self.check_history: list[HealthCheckResult] = []
        self.last_check_time: datetime | None = None

        # Performance metrics
        self.request_count = 0
        self.error_count = 0
        self.total_response_time_ms = 0.0

    async def check_all(self) -> dict[str, Any]:
        """Run all health checks"""
        start_time = time.time()

        # Run all checks in parallel
        checks = await asyncio.gather(
            self.check_system_resources(),
            self.check_music21_functionality(),
            self.check_cache_systems(),
            self.check_dependencies(),
            self.check_performance_metrics(),
            return_exceptions=True,
        )

        # Process results
        results: list[HealthCheckResult] = []
        overall_status = HealthStatus.HEALTHY

        for check in checks:
            if isinstance(check, Exception):
                # Health check itself failed
                result = HealthCheckResult(
                    name="unknown",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {str(check)}",
                )
                results.append(result)
                overall_status = HealthStatus.UNHEALTHY
            else:
                # check is HealthCheckResult in this branch
                assert isinstance(check, HealthCheckResult)
                results.append(check)
                # Downgrade overall status if needed
                if check.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif (
                    check.status == HealthStatus.DEGRADED
                    and overall_status != HealthStatus.UNHEALTHY
                ):
                    overall_status = HealthStatus.DEGRADED

        # Store in history
        self.check_history.extend(results)
        self.last_check_time = datetime.now(timezone.utc)

        # Trim history to last 100 checks
        if len(self.check_history) > 100:
            self.check_history = self.check_history[-100:]

        total_duration = (time.time() - start_time) * 1000

        return {
            "status": overall_status.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_ms": total_duration,
            "checks": [r.to_dict() for r in results],
            "system": {
                "platform": platform.platform(),
                "python_version": sys.version,
                "process_uptime_seconds": time.time() - psutil.Process().create_time(),
            },
        }

    async def check_system_resources(self) -> HealthCheckResult:
        """Check system resource usage"""
        start_time = time.time()

        try:
            # Memory check
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # CPU check
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # Disk check (for temp directory)
            temp_dir = os.path.dirname(os.path.realpath(__file__))
            disk = psutil.disk_usage(temp_dir)
            disk_percent = disk.percent

            # Determine status
            if (
                memory_percent > self.memory_threshold
                or cpu_percent > self.cpu_threshold
            ):
                status = HealthStatus.UNHEALTHY
                message = f"High resource usage - Memory: {memory_percent:.1f}%, CPU: {cpu_percent:.1f}%"
            elif (
                memory_percent > self.memory_threshold * 0.8
                or cpu_percent > self.cpu_threshold * 0.8
            ):
                status = HealthStatus.DEGRADED
                message = f"Moderate resource usage - Memory: {memory_percent:.1f}%, CPU: {cpu_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = "System resources within normal limits"

            duration = (time.time() - start_time) * 1000

            return HealthCheckResult(
                name="system_resources",
                status=status,
                message=message,
                details={
                    "memory_percent": memory_percent,
                    "memory_available_mb": memory.available / (1024 * 1024),
                    "cpu_percent": cpu_percent,
                    "cpu_count": psutil.cpu_count(),
                    "disk_percent": disk_percent,
                    "disk_free_gb": disk.free / (1024 * 1024 * 1024),
                },
                duration_ms=duration,
            )

        except Exception as e:
            logger.error(f"System resource check failed: {e}")
            return HealthCheckResult(
                name="system_resources",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check system resources: {str(e)}",
            )

    async def check_music21_functionality(self) -> HealthCheckResult:
        """Verify core music21 operations work"""
        start_time = time.time()

        try:
            # Test 1: Create a simple score
            test_score = stream.Score()
            test_part = stream.Part()
            test_part.append(chord.Chord(["C4", "E4", "G4"]))
            test_score.append(test_part)

            # Test 2: Key analysis
            test_key = test_score.analyze("key")

            # Test 3: Conversion
            midi_data = test_score.write("midi")

            # Test 4: Parse musicxml
            xml_test = converter.parse("tinyNotation: 4/4 c4 d4 e4 f4")

            duration = (time.time() - start_time) * 1000

            if duration > self.response_time_threshold:
                status = HealthStatus.DEGRADED
                message = f"Music21 operations slow: {duration:.1f}ms"
            else:
                status = HealthStatus.HEALTHY
                message = "Music21 functionality operational"

            return HealthCheckResult(
                name="music21_functionality",
                status=status,
                message=message,
                details={
                    "operations_tested": [
                        "score_creation",
                        "key_analysis",
                        "midi_conversion",
                        "tinynotation_parsing",
                    ],
                    "test_key": str(test_key),
                },
                duration_ms=duration,
            )

        except Exception as e:
            logger.error(f"Music21 functionality check failed: {e}")
            return HealthCheckResult(
                name="music21_functionality",
                status=HealthStatus.UNHEALTHY,
                message=f"Music21 operations failed: {str(e)}",
            )

    async def check_cache_systems(self) -> HealthCheckResult:
        """Check cache system health"""
        start_time = time.time()

        try:
            # Import cache-related modules
            from .performance_optimizations import PerformanceOptimizer

            # Create test optimizer
            test_optimizer = PerformanceOptimizer(cache_ttl=60, max_cache_size=10)

            # Test cache operations
            test_chord = chord.Chord(["C4", "E4", "G4"])
            test_key = key.Key("C")

            # Test caching
            result1 = test_optimizer.get_cached_roman_numeral(test_chord, test_key)
            result2 = test_optimizer.get_cached_roman_numeral(
                test_chord, test_key
            )  # Should hit cache

            # Get cache stats
            metrics = test_optimizer.get_performance_metrics()

            duration = (time.time() - start_time) * 1000

            # Check cache hit rate
            cache_hit_rate = (
                metrics.get("current_metrics", {})
                .get("cache_stats", {})
                .get("hit_rate", 0)
            )

            if cache_hit_rate > 0:
                status = HealthStatus.HEALTHY
                message = (
                    f"Cache systems operational with {cache_hit_rate:.1%} hit rate"
                )
            else:
                status = HealthStatus.DEGRADED
                message = "Cache systems operational but no hits recorded"

            # Cleanup
            test_optimizer.shutdown()

            return HealthCheckResult(
                name="cache_systems",
                status=status,
                message=message,
                details={
                    "cache_sizes": {
                        "roman": len(test_optimizer.roman_cache),
                        "key": len(test_optimizer.key_cache),
                        "chord_analysis": len(test_optimizer.chord_analysis_cache),
                    },
                    "hit_rate": cache_hit_rate,
                },
                duration_ms=duration,
            )

        except Exception as e:
            logger.error(f"Cache system check failed: {e}")
            return HealthCheckResult(
                name="cache_systems",
                status=HealthStatus.DEGRADED,
                message=f"Cache system check partial failure: {str(e)}",
            )

    async def check_dependencies(self) -> HealthCheckResult:
        """Check required dependencies are available"""
        start_time = time.time()

        try:
            dependencies: dict[str, str | None] = {
                "music21": None,
                "mcp": None,
                "asyncio": None,
                "cachetools": None,
                "psutil": None,
            }

            # Check each dependency
            for dep_name in dependencies:
                try:
                    module = __import__(dep_name)
                    if hasattr(module, "__version__"):
                        dependencies[dep_name] = module.__version__
                    else:
                        dependencies[dep_name] = "installed"
                except ImportError:
                    dependencies[dep_name] = "missing"

            # Check for missing dependencies
            missing = [k for k, v in dependencies.items() if v == "missing"]

            duration = (time.time() - start_time) * 1000

            if missing:
                status = HealthStatus.UNHEALTHY
                message = f"Missing dependencies: {', '.join(missing)}"
            else:
                status = HealthStatus.HEALTHY
                message = "All dependencies available"

            return HealthCheckResult(
                name="dependencies",
                status=status,
                message=message,
                details={"dependencies": dependencies},
                duration_ms=duration,
            )

        except Exception as e:
            logger.error(f"Dependency check failed: {e}")
            return HealthCheckResult(
                name="dependencies",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check dependencies: {str(e)}",
            )

    async def check_performance_metrics(self) -> HealthCheckResult:
        """Check performance metrics"""
        start_time = time.time()

        try:
            # Calculate metrics
            avg_response_time = (
                self.total_response_time_ms / self.request_count
                if self.request_count > 0
                else 0
            )

            error_rate = (
                self.error_count / self.request_count if self.request_count > 0 else 0
            )

            duration = (time.time() - start_time) * 1000

            # Determine status based on metrics
            if error_rate > 0.1:  # >10% error rate
                status = HealthStatus.UNHEALTHY
                message = f"High error rate: {error_rate:.1%}"
            elif avg_response_time > self.response_time_threshold:
                status = HealthStatus.DEGRADED
                message = f"Slow response times: {avg_response_time:.1f}ms"
            elif error_rate > 0.05:  # >5% error rate
                status = HealthStatus.DEGRADED
                message = f"Moderate error rate: {error_rate:.1%}"
            else:
                status = HealthStatus.HEALTHY
                message = "Performance metrics within normal range"

            return HealthCheckResult(
                name="performance_metrics",
                status=status,
                message=message,
                details={
                    "request_count": self.request_count,
                    "error_count": self.error_count,
                    "error_rate": error_rate,
                    "avg_response_time_ms": avg_response_time,
                },
                duration_ms=duration,
            )

        except Exception as e:
            logger.error(f"Performance metrics check failed: {e}")
            return HealthCheckResult(
                name="performance_metrics",
                status=HealthStatus.DEGRADED,
                message=f"Failed to check performance metrics: {str(e)}",
            )

    def record_request(self, response_time_ms: float, success: bool = True):
        """Record a request for metrics tracking"""
        self.request_count += 1
        self.total_response_time_ms += response_time_ms

        if not success:
            self.error_count += 1

    async def get_readiness(self) -> dict[str, Any]:
        """Check if service is ready to handle requests"""
        checks = await asyncio.gather(
            self.check_music21_functionality(),
            self.check_dependencies(),
            return_exceptions=True,
        )

        # If any check failed with exception, not ready
        has_exceptions = any(isinstance(check, Exception) for check in checks)

        ready = not has_exceptions and all(
            isinstance(check, HealthCheckResult)
            and check.status != HealthStatus.UNHEALTHY
            for check in checks
            if not isinstance(check, Exception)
        )

        return {
            "ready": ready,
            "checks": [
                c.to_dict() if isinstance(c, HealthCheckResult) else {"error": str(c)}
                for c in checks
            ],
        }

    async def get_liveness(self) -> dict[str, Any]:
        """Check if service is alive (basic health)"""
        try:
            # Simple check - can we create a basic music21 object?
            test_chord = chord.Chord(["C4"])

            return {
                "alive": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            return {
                "alive": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }


# Singleton health checker instance
_health_checker: HealthChecker | None = None


def get_health_checker() -> HealthChecker:
    """Get or create the singleton health checker"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


# Convenience functions for health endpoints
async def health_check() -> dict[str, Any]:
    """Comprehensive health check endpoint"""
    checker = get_health_checker()
    return await checker.check_all()


async def readiness_check() -> dict[str, Any]:
    """Readiness check endpoint (for Kubernetes)"""
    checker = get_health_checker()
    return await checker.get_readiness()


async def liveness_check() -> dict[str, Any]:
    """Liveness check endpoint (for Kubernetes)"""
    checker = get_health_checker()
    return await checker.get_liveness()
