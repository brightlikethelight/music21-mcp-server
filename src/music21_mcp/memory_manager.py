"""
Memory Manager - Advanced memory management for music21 operations

Prevents memory leaks and ensures efficient resource usage in long-running operations.
"""

import gc
import logging
import os
import sys
import weakref
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Optional

import psutil
from music21 import stream

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Advanced memory management for music21 operations.
    
    Features:
    - Automatic garbage collection after large operations
    - Memory usage monitoring and alerts
    - Weak references for cached scores
    - Memory pressure detection and response
    - Automatic cleanup of music21 internal caches
    """
    
    def __init__(self, max_memory_mb: int = 512, gc_threshold_mb: int = 100):
        self.max_memory_mb = max_memory_mb
        self.gc_threshold_mb = gc_threshold_mb
        self.process = psutil.Process(os.getpid())
        self._last_gc_memory = self._get_memory_usage_mb()
        self._operation_count = 0
        self._weak_refs: dict[str, weakref.ref] = {}
        
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def _get_memory_percent(self) -> float:
        """Get memory usage as percentage of system memory"""
        return self.process.memory_percent()
    
    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure"""
        memory_mb = self._get_memory_usage_mb()
        memory_percent = self._get_memory_percent()
        
        # Check absolute and percentage thresholds
        if memory_mb > self.max_memory_mb:
            logger.warning(f"Memory usage {memory_mb:.1f}MB exceeds limit {self.max_memory_mb}MB")
            return True
        
        if memory_percent > 80.0:
            logger.warning(f"Memory usage {memory_percent:.1f}% of system memory")
            return True
        
        return False
    
    def force_cleanup(self):
        """Force aggressive memory cleanup"""
        logger.info("Forcing memory cleanup...")
        
        # Clear music21 internal caches
        try:
            from music21 import environment
            env = environment.Environment()
            # Clear the parse cache
            if hasattr(env, '_parseCache'):
                env._parseCache = {}
            # Clear the ref cache
            if hasattr(env, '_refCache'):
                env._refCache = {}
        except Exception as e:
            logger.debug(f"Could not clear music21 caches: {e}")
        
        # Clear weak references to dead objects
        dead_refs = []
        for key, ref in self._weak_refs.items():
            if ref() is None:
                dead_refs.append(key)
        for key in dead_refs:
            del self._weak_refs[key]
        
        # Force garbage collection
        before_mb = self._get_memory_usage_mb()
        gc.collect(2)  # Full collection including cyclic references
        after_mb = self._get_memory_usage_mb()
        
        freed_mb = before_mb - after_mb
        if freed_mb > 0:
            logger.info(f"Freed {freed_mb:.1f}MB of memory")
        
        self._last_gc_memory = after_mb
    
    def maybe_cleanup(self):
        """Cleanup if memory usage has grown significantly"""
        current_mb = self._get_memory_usage_mb()
        growth_mb = current_mb - self._last_gc_memory
        
        if growth_mb > self.gc_threshold_mb:
            logger.debug(f"Memory grew by {growth_mb:.1f}MB, triggering cleanup")
            self.force_cleanup()
    
    def store_weak_reference(self, key: str, obj: Any):
        """Store a weak reference to an object"""
        try:
            self._weak_refs[key] = weakref.ref(obj)
        except TypeError:
            # Some objects can't be weakly referenced
            logger.debug(f"Cannot create weak reference for {type(obj)}")
    
    def get_weak_reference(self, key: str) -> Optional[Any]:
        """Get an object from weak reference if still alive"""
        ref = self._weak_refs.get(key)
        if ref:
            obj = ref()
            if obj is not None:
                return obj
            else:
                # Object was garbage collected
                del self._weak_refs[key]
        return None
    
    @contextmanager
    def managed_operation(self, operation_name: str = "operation"):
        """Context manager for memory-managed operations"""
        start_mb = self._get_memory_usage_mb()
        logger.debug(f"Starting {operation_name}, memory: {start_mb:.1f}MB")
        
        try:
            yield self
        finally:
            # Always cleanup after operation
            self._operation_count += 1
            
            # Check memory pressure
            if self.check_memory_pressure():
                self.force_cleanup()
            elif self._operation_count % 10 == 0:
                # Periodic cleanup every 10 operations
                self.maybe_cleanup()
            
            end_mb = self._get_memory_usage_mb()
            logger.debug(f"Completed {operation_name}, memory: {end_mb:.1f}MB (Δ{end_mb-start_mb:+.1f}MB)")
    
    def clean_score(self, score: stream.Score) -> stream.Score:
        """Clean a score object to reduce memory usage"""
        if score is None:
            return None
        
        try:
            # Remove unnecessary metadata
            if hasattr(score, '_storedDict'):
                score._storedDict = {}
            
            # Clear caches in the score
            if hasattr(score, '_cache'):
                score._cache = {}
            
            # Flatten if too complex (reduces object graph)
            if len(score.parts) > 10:
                logger.debug("Flattening complex score to reduce memory")
                score = score.flatten()
            
            return score
        except Exception as e:
            logger.warning(f"Could not clean score: {e}")
            return score


def memory_managed(cleanup_after: bool = True):
    """
    Decorator for automatic memory management of functions.
    
    Args:
        cleanup_after: Force cleanup after function execution
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            manager = MemoryManager()
            with manager.managed_operation(func.__name__):
                result = await func(*args, **kwargs)
                if cleanup_after:
                    manager.force_cleanup()
                return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            manager = MemoryManager()
            with manager.managed_operation(func.__name__):
                result = func(*args, **kwargs)
                if cleanup_after:
                    manager.force_cleanup()
                return result
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class ScoreMemoryPool:
    """
    Memory pool for efficient score object reuse.
    
    Reduces allocation overhead by reusing score objects.
    """
    
    def __init__(self, pool_size: int = 10):
        self.pool_size = pool_size
        self._pool: list[stream.Score] = []
        self._in_use: set[int] = set()
    
    def acquire(self) -> stream.Score:
        """Get a score from the pool or create new one"""
        # Try to get from pool
        for score in self._pool:
            score_id = id(score)
            if score_id not in self._in_use:
                self._in_use.add(score_id)
                # Clear the score for reuse
                score.clear()
                return score
        
        # Create new if pool not full
        if len(self._pool) < self.pool_size:
            score = stream.Score()
            self._pool.append(score)
            self._in_use.add(id(score))
            return score
        
        # Pool full, create temporary
        return stream.Score()
    
    def release(self, score: stream.Score):
        """Return a score to the pool"""
        score_id = id(score)
        if score_id in self._in_use:
            self._in_use.discard(score_id)
            # Clear for next use
            try:
                score.clear()
            except (AttributeError, Exception) as e:
                logger.debug(f"Could not clear score {score_id}: {e}")
    
    def clear(self):
        """Clear the entire pool"""
        self._pool.clear()
        self._in_use.clear()


# Global memory manager instance
_global_memory_manager = None


def get_memory_manager() -> MemoryManager:
    """Get or create global memory manager"""
    global _global_memory_manager
    if _global_memory_manager is None:
        max_memory = int(os.getenv("MUSIC21_MAX_MEMORY_MB", "512"))
        gc_threshold = int(os.getenv("MUSIC21_GC_THRESHOLD_MB", "100"))
        _global_memory_manager = MemoryManager(max_memory, gc_threshold)
    return _global_memory_manager


def monitor_memory_usage():
    """Log current memory usage statistics"""
    manager = get_memory_manager()
    memory_mb = manager._get_memory_usage_mb()
    memory_percent = manager._get_memory_percent()
    
    # Get detailed stats
    memory_info = manager.process.memory_info()
    
    stats = {
        "rss_mb": memory_mb,
        "vms_mb": memory_info.vms / 1024 / 1024,
        "percent": memory_percent,
        "available_mb": psutil.virtual_memory().available / 1024 / 1024,
        "gc_stats": gc.get_stats(),
    }
    
    logger.info(f"Memory stats: {stats}")
    return stats