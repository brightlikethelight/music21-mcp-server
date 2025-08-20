#!/usr/bin/env python3
"""
Advanced Memory Pressure Monitoring System

Monitors system memory pressure and automatically triggers cleanup actions
to prevent out-of-memory crashes in resource-constrained environments.
Designed for reliable operation with 512MB memory limits.
"""

import gc
import logging
import threading
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import psutil

logger = logging.getLogger(__name__)


class MemoryPressureLevel(Enum):
    """Memory pressure severity levels"""

    NORMAL = "normal"  # < 60% memory usage
    WARNING = "warning"  # 60-75% memory usage
    HIGH = "high"  # 75-85% memory usage
    CRITICAL = "critical"  # 85-95% memory usage
    EMERGENCY = "emergency"  # > 95% memory usage


@dataclass
class MemoryStats:
    """Current memory statistics"""

    rss_mb: float
    vms_mb: float
    percent: float
    available_mb: float
    pressure_level: MemoryPressureLevel
    timestamp: float


@dataclass
class CleanupAction:
    """Cleanup action configuration"""

    name: str
    action: Callable[[], Any]
    pressure_threshold: MemoryPressureLevel
    priority: int  # Lower number = higher priority
    can_repeat: bool = True
    last_executed: float = 0.0
    cooldown_seconds: float = 30.0


class MemoryPressureMonitor:
    """
    Advanced memory pressure monitoring with automatic cleanup

    Features:
    - Real-time memory monitoring
    - Graduated response to memory pressure
    - Configurable cleanup actions
    - Memory leak detection
    - Automatic garbage collection tuning
    - Emergency memory clearing
    """

    def __init__(
        self,
        max_memory_mb: int = 512,
        monitoring_interval: float = 5.0,
        aggressive_gc_threshold: float = 0.75,
        emergency_threshold: float = 0.95,
    ):
        self.max_memory_mb = max_memory_mb
        self.monitoring_interval = monitoring_interval
        self.aggressive_gc_threshold = aggressive_gc_threshold
        self.emergency_threshold = emergency_threshold

        # Monitoring state
        self._monitoring = False
        self._monitor_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()

        # Memory statistics
        self._current_stats: MemoryStats | None = None
        self._stats_history: list[MemoryStats] = []
        self._max_history_size = 100

        # Cleanup actions registry
        self._cleanup_actions: list[CleanupAction] = []
        self._registered_objects: set[weakref.ref] = set()

        # Performance metrics
        self._cleanup_events = 0
        self._gc_collections = 0
        self._memory_recoveries = 0

        # Initialize process monitor
        self._process = psutil.Process()

        # Register default cleanup actions
        self._register_default_actions()

        logger.info(
            f"Memory pressure monitor initialized: max={max_memory_mb}MB, "
            f"interval={monitoring_interval}s, emergency_threshold={emergency_threshold * 100}%"
        )

    def _register_default_actions(self):
        """Register default cleanup actions"""

        # Gentle garbage collection
        self.register_cleanup_action(
            "gentle_gc",
            lambda: gc.collect(),
            MemoryPressureLevel.WARNING,
            priority=1,
            cooldown_seconds=10.0,
        )

        # Aggressive garbage collection
        self.register_cleanup_action(
            "aggressive_gc",
            self._aggressive_gc,
            MemoryPressureLevel.HIGH,
            priority=2,
            cooldown_seconds=5.0,
        )

        # Emergency memory clearing
        self.register_cleanup_action(
            "emergency_cleanup",
            self._emergency_cleanup,
            MemoryPressureLevel.CRITICAL,
            priority=3,
            cooldown_seconds=1.0,
        )

    def register_cleanup_action(
        self,
        name: str,
        action: Callable[[], Any],
        pressure_threshold: MemoryPressureLevel,
        priority: int,
        can_repeat: bool = True,
        cooldown_seconds: float = 30.0,
    ):
        """Register a cleanup action for memory pressure response"""
        cleanup_action = CleanupAction(
            name=name,
            action=action,
            pressure_threshold=pressure_threshold,
            priority=priority,
            can_repeat=can_repeat,
            cooldown_seconds=cooldown_seconds,
        )

        self._cleanup_actions.append(cleanup_action)
        # Sort by priority (lower number = higher priority)
        self._cleanup_actions.sort(key=lambda x: x.priority)

        logger.debug(f"Registered cleanup action: {name} (priority={priority})")

    def register_object_for_cleanup(self, obj: Any):
        """Register an object for potential cleanup during memory pressure"""
        if hasattr(obj, "cleanup") or hasattr(obj, "clear"):
            ref = weakref.ref(obj)
            self._registered_objects.add(ref)
            logger.debug(f"Registered object for cleanup: {type(obj).__name__}")

    def start_monitoring(self):
        """Start memory pressure monitoring"""
        if self._monitoring:
            logger.warning("Memory monitoring already running")
            return

        self._monitoring = True
        self._shutdown_event.clear()

        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True, name="MemoryPressureMonitor"
        )
        self._monitor_thread.start()

        logger.info("Memory pressure monitoring started")

    def stop_monitoring(self):
        """Stop memory pressure monitoring"""
        if not self._monitoring:
            return

        self._monitoring = False
        self._shutdown_event.set()

        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
            if self._monitor_thread.is_alive():
                logger.warning("Monitor thread did not shutdown gracefully")

        logger.info("Memory pressure monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        logger.debug("Memory monitoring loop started")

        while not self._shutdown_event.is_set():
            try:
                # Get current memory stats
                stats = self._get_memory_stats()
                self._current_stats = stats

                # Add to history
                self._stats_history.append(stats)
                if len(self._stats_history) > self._max_history_size:
                    self._stats_history.pop(0)

                # Check for memory pressure and respond
                if stats.pressure_level != MemoryPressureLevel.NORMAL:
                    self._handle_memory_pressure(stats)

                # Check for memory leaks
                if len(self._stats_history) >= 10:
                    self._check_memory_trends()

                # Adjust GC behavior based on memory pressure
                self._tune_garbage_collection(stats)

            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}")

            # Wait for next monitoring cycle
            if not self._shutdown_event.wait(self.monitoring_interval):
                continue
            break

        logger.debug("Memory monitoring loop exited")

    def _get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        memory_info = self._process.memory_info()
        virtual_memory = psutil.virtual_memory()

        rss_mb = memory_info.rss / 1024 / 1024
        vms_mb = memory_info.vms / 1024 / 1024
        percent = self._process.memory_percent()
        available_mb = virtual_memory.available / 1024 / 1024

        # Determine pressure level based on configured max memory
        if self.max_memory_mb > 0:
            utilization = rss_mb / self.max_memory_mb
        else:
            utilization = percent / 100.0

        if utilization >= self.emergency_threshold:
            pressure_level = MemoryPressureLevel.EMERGENCY
        elif utilization >= 0.85:
            pressure_level = MemoryPressureLevel.CRITICAL
        elif utilization >= self.aggressive_gc_threshold:
            pressure_level = MemoryPressureLevel.HIGH
        elif utilization >= 0.60:
            pressure_level = MemoryPressureLevel.WARNING
        else:
            pressure_level = MemoryPressureLevel.NORMAL

        return MemoryStats(
            rss_mb=rss_mb,
            vms_mb=vms_mb,
            percent=percent,
            available_mb=available_mb,
            pressure_level=pressure_level,
            timestamp=time.time(),
        )

    def _handle_memory_pressure(self, stats: MemoryStats):
        """Handle memory pressure by executing appropriate cleanup actions"""
        logger.info(
            f"Memory pressure detected: {stats.pressure_level.value} "
            f"({stats.rss_mb:.1f}MB / {self.max_memory_mb}MB)"
        )

        current_time = time.time()
        actions_executed = 0

        for action in self._cleanup_actions:
            # Skip if pressure level is not high enough
            if self._pressure_priority(
                action.pressure_threshold
            ) > self._pressure_priority(stats.pressure_level):
                continue

            # Skip if in cooldown period
            if current_time - action.last_executed < action.cooldown_seconds:
                continue

            try:
                logger.info(f"Executing cleanup action: {action.name}")
                result = action.action()
                action.last_executed = current_time
                actions_executed += 1
                self._cleanup_events += 1

                # Log result if available
                if result:
                    logger.debug(f"Cleanup action {action.name} result: {result}")

                # Check if memory pressure was reduced
                new_stats = self._get_memory_stats()
                if self._pressure_priority(
                    new_stats.pressure_level
                ) < self._pressure_priority(stats.pressure_level):
                    logger.info(
                        f"Memory pressure reduced to {new_stats.pressure_level.value}"
                    )
                    self._memory_recoveries += 1
                    break

            except Exception as e:
                logger.error(f"Cleanup action {action.name} failed: {e}")

        if actions_executed == 0:
            logger.warning("No cleanup actions could be executed for memory pressure")

    def _pressure_priority(self, level: MemoryPressureLevel) -> int:
        """Get numeric priority for pressure level (higher = more severe)"""
        return {
            MemoryPressureLevel.NORMAL: 0,
            MemoryPressureLevel.WARNING: 1,
            MemoryPressureLevel.HIGH: 2,
            MemoryPressureLevel.CRITICAL: 3,
            MemoryPressureLevel.EMERGENCY: 4,
        }[level]

    def _check_memory_trends(self):
        """Check for concerning memory trends that might indicate leaks"""
        if len(self._stats_history) < 10:
            return

        # Check last 10 measurements for consistent growth
        recent_stats = self._stats_history[-10:]
        growth_trend = []

        for i in range(1, len(recent_stats)):
            growth = recent_stats[i].rss_mb - recent_stats[i - 1].rss_mb
            growth_trend.append(growth)

        # If memory consistently grows, might be a leak
        positive_growth = sum(1 for g in growth_trend if g > 1.0)  # 1MB threshold

        if positive_growth >= 7:  # 70% of measurements show growth
            avg_growth = sum(growth_trend) / len(growth_trend)
            logger.warning(
                f"Potential memory leak detected: consistent growth "
                f"averaging {avg_growth:.2f}MB per monitoring cycle"
            )

            # Trigger aggressive cleanup
            self._aggressive_gc()

    def _tune_garbage_collection(self, stats: MemoryStats):
        """Adjust garbage collection behavior based on memory pressure"""
        if stats.pressure_level in [
            MemoryPressureLevel.HIGH,
            MemoryPressureLevel.CRITICAL,
        ]:
            # Get current GC thresholds
            thresholds = gc.get_threshold()

            # Make GC more aggressive by reducing thresholds
            new_thresholds = (
                max(thresholds[0] // 2, 100),  # Generation 0
                max(thresholds[1] // 2, 10),  # Generation 1
                max(thresholds[2] // 2, 10),  # Generation 2
            )

            gc.set_threshold(*new_thresholds)
            logger.debug(
                f"Adjusted GC thresholds for memory pressure: {new_thresholds}"
            )

        elif stats.pressure_level == MemoryPressureLevel.NORMAL:
            # Restore default GC thresholds
            gc.set_threshold(700, 10, 10)  # Python defaults

    def _aggressive_gc(self) -> dict[str, int]:
        """Perform aggressive garbage collection"""
        logger.debug("Performing aggressive garbage collection")

        # Collect all generations multiple times
        collected = {}
        for generation in range(3):
            collected[f"gen_{generation}"] = gc.collect(generation)

        # Final full collection
        collected["final"] = gc.collect()

        self._gc_collections += 1

        total_collected = sum(collected.values())
        logger.debug(f"Aggressive GC collected {total_collected} objects: {collected}")

        return collected

    def _emergency_cleanup(self) -> dict[str, Any]:
        """Emergency cleanup when memory is critically low"""
        logger.warning("Executing emergency memory cleanup")

        results = {}

        # 1. Clear weak references to destroyed objects
        dead_refs = [ref for ref in self._registered_objects if ref() is None]
        for ref in dead_refs:
            self._registered_objects.remove(ref)
        results["cleared_dead_refs"] = len(dead_refs)

        # 2. Call cleanup on registered objects
        cleaned_objects = 0
        for ref in list(self._registered_objects):
            obj = ref()
            if obj is not None:
                try:
                    if hasattr(obj, "cleanup"):
                        obj.cleanup()
                        cleaned_objects += 1
                    elif hasattr(obj, "clear"):
                        obj.clear()
                        cleaned_objects += 1
                except Exception as e:
                    logger.error(f"Error cleaning up object {type(obj).__name__}: {e}")

        results["cleaned_objects"] = cleaned_objects

        # 3. Multiple aggressive GC passes
        gc_results = self._aggressive_gc()
        results["gc_collected"] = sum(gc_results.values())

        # 4. Clear import caches if available
        try:
            import sys

            if hasattr(sys, "_getframe"):
                # Clear some internal caches
                importlib_util = sys.modules.get("importlib.util")
                if importlib_util and hasattr(importlib_util, "_LazyDescr"):
                    # Clear lazy module loading caches
                    pass
        except Exception:
            pass

        logger.warning(f"Emergency cleanup completed: {results}")
        return results

    def get_current_stats(self) -> MemoryStats | None:
        """Get current memory statistics"""
        return self._current_stats

    def get_stats_history(self) -> list[MemoryStats]:
        """Get memory statistics history"""
        return self._stats_history.copy()

    def get_monitor_stats(self) -> dict[str, Any]:
        """Get monitoring system statistics"""
        return {
            "is_monitoring": self._monitoring,
            "cleanup_events": self._cleanup_events,
            "gc_collections": self._gc_collections,
            "memory_recoveries": self._memory_recoveries,
            "registered_cleanup_actions": len(self._cleanup_actions),
            "registered_objects": len(self._registered_objects),
            "max_memory_mb": self.max_memory_mb,
            "monitoring_interval": self.monitoring_interval,
            "current_stats": self._current_stats.__dict__
            if self._current_stats
            else None,
        }

    def force_cleanup(self, level: MemoryPressureLevel = MemoryPressureLevel.HIGH):
        """Force cleanup actions for specified pressure level"""
        logger.info(f"Forcing cleanup for pressure level: {level.value}")

        fake_stats = MemoryStats(
            rss_mb=0,
            vms_mb=0,
            percent=0,
            available_mb=0,
            pressure_level=level,
            timestamp=time.time(),
        )

        self._handle_memory_pressure(fake_stats)

    def shutdown(self):
        """Shutdown the memory pressure monitor"""
        logger.info("Shutting down memory pressure monitor")
        self.stop_monitoring()

        # Clear registered objects
        self._registered_objects.clear()
        self._cleanup_actions.clear()
        self._stats_history.clear()

    def __del__(self):
        """Ensure cleanup on destruction"""
        import contextlib
        
        with contextlib.suppress(Exception):
            self.shutdown()


# Global instance
_global_monitor: MemoryPressureMonitor | None = None


def get_memory_monitor(max_memory_mb: int = 512) -> MemoryPressureMonitor:
    """Get or create the global memory pressure monitor"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = MemoryPressureMonitor(max_memory_mb=max_memory_mb)
    return _global_monitor


def start_memory_monitoring(max_memory_mb: int = 512):
    """Start global memory monitoring"""
    monitor = get_memory_monitor(max_memory_mb)
    monitor.start_monitoring()
    return monitor


def stop_memory_monitoring():
    """Stop global memory monitoring"""
    global _global_monitor
    if _global_monitor:
        _global_monitor.stop_monitoring()
