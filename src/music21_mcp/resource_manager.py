#!/usr/bin/env python3
"""
Resource Management System for Music21 MCP Server

Provides memory-safe score storage with automatic cleanup, resource limits,
and monitoring to prevent OOM crashes and ensure reliable operation.
"""

import gc
import logging
import threading
import time
from collections.abc import MutableMapping
from typing import Any

import psutil
from cachetools import TTLCache  # type: ignore

logger = logging.getLogger(__name__)


class ResourceExhaustedError(Exception):
    """Raised when resource limits are exceeded"""

    pass


class ScoreStorage(MutableMapping[str, Any]):
    """
    Memory-managed score storage with automatic cleanup and limits.

    Features:
    - TTL-based automatic cleanup
    - Memory usage monitoring and limits
    - Resource tracking and metrics
    - Thread-safe operations
    - Graceful degradation under pressure
    """

    def __init__(
        self,
        max_scores: int = 100,
        score_ttl_seconds: int = 3600,  # 1 hour
        max_memory_mb: int = 512,
        cleanup_interval_seconds: int = 300,  # 5 minutes
    ):
        self.max_scores = max(1, max_scores)  # Ensure at least 1
        self.max_memory_mb = max(1, max_memory_mb)  # Ensure at least 1 MB
        self.cleanup_interval = cleanup_interval_seconds

        # TTL cache for automatic expiration
        self._cache = TTLCache(maxsize=max_scores, ttl=score_ttl_seconds)

        # Metadata tracking
        self._access_times: dict[str, float] = {}
        self._memory_usage: dict[str, int] = {}
        self._lock = threading.RLock()

        # Metrics
        self._total_scores_loaded = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._cleanup_runs = 0
        self._memory_warnings = 0

        # Background thread management
        self._cleanup_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()
        self._start_cleanup_thread()

        logger.info(
            f"ScoreStorage initialized: max_scores={max_scores}, "
            f"ttl={score_ttl_seconds}s, max_memory={max_memory_mb}MB"
        )

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            if key in self._cache:
                self._access_times[key] = time.time()
                self._cache_hits += 1
                return self._cache[key]
            self._cache_misses += 1
            raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        with self._lock:
            # Check memory limits before adding
            estimated_size = self._estimate_object_size(value)

            if self._would_exceed_memory_limit(estimated_size):
                # Try cleanup first
                self._force_cleanup()

                # Check again after cleanup
                if self._would_exceed_memory_limit(estimated_size):
                    self._memory_warnings += 1
                    raise ResourceExhaustedError(
                        f"Adding score '{key}' would exceed memory limit "
                        f"({self.max_memory_mb}MB). Current usage: {self._get_memory_usage_mb():.1f}MB"
                    )

            # Store with metadata
            self._cache[key] = value
            self._access_times[key] = time.time()
            self._memory_usage[key] = estimated_size
            self._total_scores_loaded += 1

            logger.debug(
                f"Stored score '{key}' ({estimated_size / 1024 / 1024:.1f}MB). "
                f"Total scores: {len(self._cache)}"
            )

    def __delitem__(self, key: str) -> None:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._access_times.pop(key, None)
                freed_mb = self._memory_usage.pop(key, 0) / 1024 / 1024
                logger.debug(f"Deleted score '{key}' (freed {freed_mb:.1f}MB)")
            else:
                raise KeyError(key)

    def __iter__(self) -> Any:
        with self._lock:
            return iter(self._cache)

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def get_stats(self) -> dict[str, Any]:
        """Get detailed storage statistics"""
        with self._lock:
            memory_mb = self._get_memory_usage_mb()
            return {
                "total_scores": len(self._cache),
                "max_scores": self.max_scores,
                "memory_usage_mb": memory_mb,
                "max_memory_mb": self.max_memory_mb,
                "memory_utilization_percent": (memory_mb / self.max_memory_mb) * 100,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "hit_rate_percent": (
                    (self._cache_hits / (self._cache_hits + self._cache_misses)) * 100
                    if (self._cache_hits + self._cache_misses) > 0
                    else 0
                ),
                "total_scores_loaded": self._total_scores_loaded,
                "cleanup_runs": self._cleanup_runs,
                "memory_warnings": self._memory_warnings,
            }

    def cleanup(self) -> dict[str, Any]:
        """Force cleanup and return statistics"""
        with self._lock:
            initial_count = len(self._cache)
            initial_memory = self._get_memory_usage_mb()

            # TTL cache automatically removes expired items
            # Force access to trigger cleanup
            list(self._cache.keys())

            # Clean up metadata for removed items
            current_keys = set(self._cache.keys())
            orphaned_access_times = set(self._access_times.keys()) - current_keys
            orphaned_memory_usage = set(self._memory_usage.keys()) - current_keys

            for key in orphaned_access_times:
                del self._access_times[key]

            for key in orphaned_memory_usage:
                del self._memory_usage[key]

            # Force garbage collection
            collected = gc.collect()

            final_count = len(self._cache)
            final_memory = self._get_memory_usage_mb()

            self._cleanup_runs += 1

            stats = {
                "removed_scores": initial_count - final_count,
                "freed_memory_mb": initial_memory - final_memory,
                "gc_collected_objects": collected,
                "remaining_scores": final_count,
            }

            if stats["removed_scores"] > 0:
                logger.info(
                    f"Cleanup completed: removed {stats['removed_scores']} scores, "
                    f"freed {stats['freed_memory_mb']:.1f}MB"
                )

            return stats

    def _estimate_object_size(self, obj: Any) -> int:
        """Estimate memory usage of an object in bytes"""
        try:
            # For music21 objects, use a heuristic based on typical sizes
            if hasattr(obj, "flat") and hasattr(obj, "notes"):
                # This is likely a music21 Score
                note_count = (
                    len(list(obj.flat.notes)) if hasattr(obj.flat, "notes") else 100
                )
                # Estimate ~1KB per note plus base overhead
                return max(note_count * 1024, 50 * 1024)  # Minimum 50KB
            # Fallback to a conservative estimate
            return 100 * 1024  # 100KB default
        except Exception:
            return 100 * 1024  # Safe fallback

    def _get_memory_usage_mb(self) -> float:
        """Get current estimated memory usage in MB"""
        return sum(self._memory_usage.values()) / 1024 / 1024

    def _would_exceed_memory_limit(self, additional_bytes: int) -> bool:
        """Check if adding object would exceed memory limit"""
        current_mb = self._get_memory_usage_mb()
        additional_mb = additional_bytes / 1024 / 1024
        return (current_mb + additional_mb) > self.max_memory_mb

    def _force_cleanup(self) -> None:
        """Force immediate cleanup of expired items"""
        self.cleanup()

    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread with proper shutdown mechanism"""

        def cleanup_worker() -> None:
            logger.debug("Cleanup worker thread started")
            while not self._shutdown_event.is_set():
                try:
                    # Use wait() instead of sleep() to allow immediate shutdown
                    if self._shutdown_event.wait(timeout=self.cleanup_interval):
                        # Shutdown event was set, exit
                        break

                    # Perform cleanup if not shutting down
                    if not self._shutdown_event.is_set():
                        self.cleanup()

                except Exception as e:
                    logger.error(f"Error in cleanup thread: {e}")
                    # Continue running even if cleanup fails

            logger.debug("Cleanup worker thread exiting")

        self._cleanup_thread = threading.Thread(
            target=cleanup_worker, daemon=True, name="ScoreStorage-Cleanup"
        )
        self._cleanup_thread.start()
        logger.debug("Background cleanup thread started")

    def shutdown(self) -> None:
        """Gracefully shutdown the background cleanup thread"""
        if self._shutdown_event and not self._shutdown_event.is_set():
            logger.debug("Initiating ScoreStorage shutdown")
            self._shutdown_event.set()

            # Wait for cleanup thread to finish with timeout
            if self._cleanup_thread and self._cleanup_thread.is_alive():
                self._cleanup_thread.join(timeout=5.0)
                if self._cleanup_thread.is_alive():
                    logger.warning("Cleanup thread did not shutdown within timeout")
                else:
                    logger.debug("Cleanup thread shutdown completed")

            # Final cleanup
            try:
                self.cleanup()
                logger.info("ScoreStorage shutdown completed")
            except Exception as e:
                logger.error(f"Error during final cleanup: {e}")

    def __del__(self) -> None:
        """Ensure cleanup thread is shutdown when object is destroyed"""
        import contextlib

        with contextlib.suppress(Exception):
            self.shutdown()


class ResourceManager:
    """
    Global resource manager for the music21 MCP server.

    Provides centralized resource management, monitoring, and limits
    to ensure reliable operation under load.
    """

    def __init__(
        self,
        max_memory_mb: int = 512,
        max_scores: int = 100,
        score_ttl_seconds: int = 3600,
    ):
        self.max_memory_mb = max(1, max_memory_mb)  # Ensure at least 1 MB
        self.max_scores = max(1, max_scores)  # Ensure at least 1

        # Create managed score storage
        self.scores = ScoreStorage(
            max_scores=max_scores,
            score_ttl_seconds=score_ttl_seconds,
            max_memory_mb=max_memory_mb,
        )

        logger.info(f"ResourceManager initialized with {max_memory_mb}MB limit")

    def shutdown(self) -> None:
        """Gracefully shutdown the resource manager and all its components"""
        logger.info("Shutting down ResourceManager")
        try:
            # Shutdown score storage (which shuts down its cleanup thread)
            self.scores.shutdown()
            logger.info("ResourceManager shutdown completed")
        except Exception as e:
            logger.error(f"Error during ResourceManager shutdown: {e}")

    def __del__(self) -> None:
        """Ensure proper cleanup when ResourceManager is destroyed"""
        import contextlib

        with contextlib.suppress(Exception):
            self.shutdown()

    def get_system_stats(self) -> dict[str, Any]:
        """Get comprehensive system resource statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()

        storage_stats = self.scores.get_stats()

        return {
            "storage": storage_stats,
            "system": {
                "process_memory_mb": memory_info.rss / 1024 / 1024,
                "process_memory_percent": process.memory_percent(),
                "cpu_percent": process.cpu_percent(),
                "open_files": len(process.open_files()),
                "threads": process.num_threads(),
            },
            "limits": {
                "max_memory_mb": self.max_memory_mb,
                "max_scores": self.max_scores,
            },
        }

    def check_health(self) -> dict[str, Any]:
        """Perform health check and return status"""
        stats = self.get_system_stats()
        storage = stats["storage"]
        system = stats["system"]

        # Determine health status
        warnings = []
        errors = []

        # Check memory usage
        if storage["memory_utilization_percent"] > 90:
            errors.append(
                f"Storage memory usage critical: {storage['memory_utilization_percent']:.1f}%"
            )
        elif storage["memory_utilization_percent"] > 75:
            warnings.append(
                f"Storage memory usage high: {storage['memory_utilization_percent']:.1f}%"
            )

        # Check system memory
        if system["process_memory_percent"] > 80:
            warnings.append(
                f"System memory usage high: {system['process_memory_percent']:.1f}%"
            )

        # Check score count
        if storage["total_scores"] > self.max_scores * 0.9:
            warnings.append(
                f"Score count near limit: {storage['total_scores']}/{self.max_scores}"
            )

        # Determine overall status
        if errors:
            status = "critical"
        elif warnings:
            status = "warning"
        else:
            status = "healthy"

        return {
            "status": status,
            "warnings": warnings,
            "errors": errors,
            "stats": stats,
            "timestamp": time.time(),
        }

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return self.scores._get_memory_usage_mb()

    def check_memory(self, additional_mb: int) -> bool:
        """Check if we can allocate additional memory"""
        current_usage = self.get_memory_usage()
        return (current_usage + additional_mb) <= self.max_memory_mb

    def cleanup(self) -> dict[str, Any]:
        """Force cleanup and return statistics"""
        memory_before = self.get_memory_usage()
        cleanup_stats = self.scores.cleanup()
        memory_after = self.get_memory_usage()

        return {
            "memory_before": memory_before,
            "memory_after": memory_after,
            "memory_freed": memory_before - memory_after,
            **cleanup_stats,
        }

    def _monitor_resources(self) -> None:
        """Monitor resource usage and log warnings if needed"""
        stats = self.get_system_stats()
        storage = stats["storage"]

        if storage["memory_utilization_percent"] > 75:
            logger.warning(
                f"High memory usage: {storage['memory_utilization_percent']:.1f}%"
            )

        if storage["total_scores"] > self.max_scores * 0.8:
            logger.warning(
                f"High score count: {storage['total_scores']}/{self.max_scores}"
            )
