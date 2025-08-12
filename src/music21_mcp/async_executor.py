"""
Async Executor Utilities - Proper async/await execution for music21 operations

This module provides utilities to run synchronous music21 operations in background
threads without blocking the async event loop, improving performance and responsiveness.
"""

import asyncio
import functools
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar, Optional

logger = logging.getLogger(__name__)

# Type variable for generic function returns
T = TypeVar('T')

class Music21AsyncExecutor:
    """
    Singleton executor for running music21 operations in background threads
    
    This prevents blocking the async event loop when performing CPU-intensive
    music21 operations like parsing, analysis, and generation.
    """
    
    _instance: Optional['Music21AsyncExecutor'] = None
    _lock = asyncio.Lock()
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize the executor with a thread pool
        
        Args:
            max_workers: Maximum number of worker threads for music21 operations
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="music21-worker"
        )
        self._total_operations = 0
        self._total_time = 0.0
        logger.info(f"Music21 async executor initialized with {max_workers} workers")
    
    @classmethod
    async def get_instance(cls, max_workers: int = 4) -> 'Music21AsyncExecutor':
        """Get or create the singleton executor instance"""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(max_workers)
        return cls._instance
    
    async def run(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Run a synchronous function in the background thread pool
        
        Args:
            func: The synchronous function to run
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            The result of the function execution
            
        Raises:
            Any exception raised by the function
        """
        loop = asyncio.get_event_loop()
        start_time = time.time()
        
        try:
            # Use functools.partial to bind kwargs if needed
            if kwargs:
                bound_func = functools.partial(func, **kwargs)
                result = await loop.run_in_executor(self.executor, bound_func, *args)
            else:
                result = await loop.run_in_executor(self.executor, func, *args)
            
            duration = time.time() - start_time
            self._total_operations += 1
            self._total_time += duration
            
            if duration > 1.0:  # Log slow operations
                logger.info(f"Music21 operation {func.__name__} completed in {duration:.2f}s")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Music21 operation {func.__name__} failed after {duration:.2f}s: {e}")
            raise
    
    def get_stats(self) -> dict[str, Any]:
        """Get executor performance statistics"""
        avg_time = self._total_time / max(1, self._total_operations)
        return {
            "total_operations": self._total_operations,
            "total_time_seconds": self._total_time,
            "average_time_seconds": avg_time,
            "max_workers": self.max_workers,
            "active_threads": self.executor._threads and len(self.executor._threads) or 0
        }
    
    def shutdown(self, wait: bool = True):
        """Shutdown the executor"""
        if self.executor:
            self.executor.shutdown(wait=wait)
            logger.info("Music21 async executor shutdown")


# Global convenience functions

async def run_in_thread(func: Callable[..., T], *args, **kwargs) -> T:
    """
    Convenience function to run a synchronous function in a background thread
    
    This is the main function tools should use for music21 operations.
    
    Args:
        func: The synchronous function to run
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        The result of the function execution
    """
    executor = await Music21AsyncExecutor.get_instance()
    return await executor.run(func, *args, **kwargs)


def async_music21(func: Callable[..., T]) -> Callable[..., Any]:
    """
    Decorator to automatically run music21 functions in background threads
    
    Usage:
        @async_music21
        def my_music21_operation(score):
            return score.analyze('key')
        
        # Now can be awaited
        result = await my_music21_operation(score)
    
    Args:
        func: The synchronous function to wrap
        
    Returns:
        An async function that runs the original in a background thread
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        return await run_in_thread(func, *args, **kwargs)
    
    return wrapper


# Common music21 operations wrapped for async use

async def parse_file_async(file_path: str) -> Any:
    """Parse a music file asynchronously"""
    from music21 import converter
    return await run_in_thread(converter.parse, file_path)


async def parse_corpus_async(corpus_path: str) -> Any:
    """Parse a corpus file asynchronously"""
    from music21 import corpus
    return await run_in_thread(corpus.parse, corpus_path)


async def analyze_key_async(score: Any, algorithm: str = 'key') -> Any:
    """Analyze key signature asynchronously"""
    return await run_in_thread(score.analyze, algorithm)


async def chordify_async(score: Any, **kwargs) -> Any:
    """Chordify a score asynchronously"""
    return await run_in_thread(score.chordify, **kwargs)


async def flatten_score_async(score: Any) -> Any:
    """Flatten a score asynchronously"""
    return await run_in_thread(score.flatten)


# Utility for progress reporting during async operations
class AsyncProgressReporter:
    """
    Helper class for reporting progress during long-running async operations
    
    This allows tools to report progress even when the actual work is happening
    in background threads.
    """
    
    def __init__(self, callback: Optional[Callable[[float, str], None]] = None):
        self.callback = callback
        self.current_progress = 0.0
        self.current_message = ""
    
    def update(self, progress: float, message: str = ""):
        """Update progress and message"""
        self.current_progress = progress
        self.current_message = message
        if self.callback:
            self.callback(progress, message)
    
    async def run_with_progress(self, 
                              func: Callable[..., T], 
                              progress_start: float = 0.0,
                              progress_end: float = 1.0,
                              message: str = "Processing...",
                              *args, **kwargs) -> T:
        """
        Run a function with progress reporting
        
        Args:
            func: The function to run
            progress_start: Starting progress value (0.0-1.0)
            progress_end: Ending progress value (0.0-1.0)
            message: Message to display during execution
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        self.update(progress_start, message)
        result = await run_in_thread(func, *args, **kwargs)
        self.update(progress_end, f"{message} complete")
        return result


# Graceful shutdown handler
async def shutdown_executor():
    """Shutdown the global executor gracefully"""
    if Music21AsyncExecutor._instance:
        Music21AsyncExecutor._instance.shutdown(wait=True)
        Music21AsyncExecutor._instance = None


# Statistics accessor
async def get_executor_stats() -> dict[str, Any]:
    """Get statistics from the global executor"""
    if Music21AsyncExecutor._instance:
        return Music21AsyncExecutor._instance.get_stats()
    return {
        "total_operations": 0,
        "total_time_seconds": 0.0,
        "average_time_seconds": 0.0,
        "max_workers": 0,
        "active_threads": 0
    }