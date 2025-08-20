"""
Parallel Processing System for Music Analysis

Provides concurrent processing for CPU-intensive music21 operations
to further improve performance beyond caching alone.
"""

import asyncio
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class ParallelProcessor:
    """High-performance parallel processor for music analysis operations"""

    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize parallel processor

        Args:
            max_workers: Maximum number of worker threads (defaults to CPU)
        """
        self.max_workers = max_workers or min(
            4, (asyncio.get_event_loop().is_running() and 4) or 4
        )
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        logger.info(f"Parallel processor initialized with {self.max_workers} workers")

    async def process_batch(
        self,
        items: list[T],
        processor_func: Callable[[T], R],
        batch_size: int = 10,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[R]:
        """
        Process a batch of items in parallel

        Args:
            items: List of items to process
            processor_func: Function to apply to each item
            batch_size: Number of items to process per batch
            progress_callback: Optional callback for progress updates

        Returns:
            List of processed results in original order
        """
        if not items:
            return []

        total_items = len(items)
        results: list[R] = []
        for _ in range(total_items):
            results.append(None)  # type: ignore

        # Process items in batches to avoid overwhelming the system
        for batch_start in range(0, total_items, batch_size):
            batch_end = min(batch_start + batch_size, total_items)
            batch_items = items[batch_start:batch_end]

            # Create futures for this batch
            loop = asyncio.get_event_loop()
            futures = [
                loop.run_in_executor(self._executor, processor_func, item)
                for item in batch_items
            ]

            # Wait for batch completion
            batch_results = await asyncio.gather(*futures, return_exceptions=True)

            # Store results in correct positions
            for i, result in enumerate(batch_results):
                results[batch_start + i] = result  # type: ignore

            # Report progress if callback provided
            if progress_callback:
                progress_callback(batch_end, total_items)

            # Small delay between batches to prevent system overload
            if batch_end < total_items:
                await asyncio.sleep(0.01)

        return results

    async def process_chord_batch(
        self, chord_items: list[Any], analysis_func: Callable[..., dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Specialized method for processing chord batches

        Args:
            chord_items: List of chord objects to analyze
            analysis_func: Cached analysis function to apply

        Returns:
            List of chord analysis dictionaries
        """

        def safe_analysis(item: Any) -> dict[str, Any]:
            """Wrapper to handle exceptions in parallel processing"""
            try:
                return analysis_func(item)
            except Exception as e:
                logger.warning(f"Parallel chord analysis failed: {e}")
                pitches = (
                    [str(p) for p in item.pitches] if hasattr(item, "pitches") else []
                )
                return {"error": str(e), "pitches": pitches, "failed": True}

        # Use smaller batch size for chord analysis
        return await self.process_batch(
            chord_items,
            safe_analysis,
            batch_size=8,  # Optimized for chord analysis
        )

    def __del__(self) -> None:
        """Clean up executor on deletion"""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)


# Global processor instance
_global_processor: Optional[ParallelProcessor] = None


def get_parallel_processor() -> ParallelProcessor:
    """Get the global parallel processor instance"""
    global _global_processor
    if _global_processor is None:
        _global_processor = ParallelProcessor()
    return _global_processor
