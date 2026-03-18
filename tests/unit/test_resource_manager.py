"""Tests for ResourceManager health check thresholds and cleanup paths."""

from unittest.mock import MagicMock, patch

import pytest

from music21_mcp.resource_manager import ResourceManager, ScoreStorage


class TestHealthCheckThresholds:
    """Test check_health() returns correct status for memory thresholds."""

    def _make_manager(self) -> ResourceManager:
        return ResourceManager(max_memory_mb=512, max_scores=100, score_ttl_seconds=300)

    def test_health_check_critical_threshold(self):
        """Memory utilization >90% yields 'critical' status."""
        manager = self._make_manager()
        try:
            # Patch get_system_stats to simulate high memory utilization
            fake_stats = {
                "storage": {
                    "total_scores": 5,
                    "max_scores": 100,
                    "memory_usage_mb": 480.0,
                    "max_memory_mb": 512,
                    "memory_utilization_percent": 93.75,
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "hit_rate_percent": 0,
                    "total_scores_loaded": 5,
                    "cleanup_runs": 0,
                    "memory_warnings": 0,
                },
                "system": {
                    "process_memory_mb": 200.0,
                    "process_memory_percent": 10.0,
                    "cpu_percent": 5.0,
                    "open_files": 10,
                    "threads": 4,
                },
                "limits": {
                    "max_memory_mb": 512,
                    "max_scores": 100,
                },
            }
            with patch.object(manager, "get_system_stats", return_value=fake_stats):
                health = manager.check_health()
            assert health["status"] == "critical"
            assert len(health["errors"]) >= 1
            assert any("critical" in e.lower() for e in health["errors"])
        finally:
            manager.shutdown()

    def test_health_check_degraded_threshold(self):
        """Storage utilization 75-90% AND high system memory yields 'warning' status."""
        manager = self._make_manager()
        try:
            fake_stats = {
                "storage": {
                    "total_scores": 5,
                    "max_scores": 100,
                    "memory_usage_mb": 400.0,
                    "max_memory_mb": 512,
                    "memory_utilization_percent": 78.1,
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "hit_rate_percent": 0,
                    "total_scores_loaded": 5,
                    "cleanup_runs": 0,
                    "memory_warnings": 0,
                },
                "system": {
                    "process_memory_mb": 200.0,
                    "process_memory_percent": 85.0,
                    "cpu_percent": 5.0,
                    "open_files": 10,
                    "threads": 4,
                },
                "limits": {
                    "max_memory_mb": 512,
                    "max_scores": 100,
                },
            }
            with patch.object(manager, "get_system_stats", return_value=fake_stats):
                health = manager.check_health()
            assert health["status"] == "warning"
            assert len(health["warnings"]) >= 1
        finally:
            manager.shutdown()

    def test_force_cleanup_eviction(self):
        """Filling storage to capacity and triggering cleanup removes items."""
        from music21 import stream

        storage = ScoreStorage(max_scores=3, score_ttl_seconds=1, max_memory_mb=1024)
        try:
            # Fill to capacity
            for i in range(3):
                s = stream.Score()
                storage[f"score_{i}"] = s
            assert len(storage) == 3

            # Force cleanup — TTL is 1s so entries will expire after sleep
            import time

            time.sleep(1.2)
            stats = storage.cleanup()
            # After TTL expiry, cleanup should have removed entries
            assert stats["removed_scores"] >= 1 or len(storage) < 3
        finally:
            storage.shutdown()
