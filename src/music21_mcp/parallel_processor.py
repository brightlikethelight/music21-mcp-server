"""
Parallel Processing System for Music Analysis

Provides concurrent processing for CPU-intensive music21 operations
to further improve performance beyond caching alone.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class ParallelProcessor:
    """High-performance parallel processor for music analysis operations"""
    
    def __init__(self, max_workers: int = None):
        """
        Initialize parallel processor
        
        Args:
            max_workers: Maximum number of worker threads (defaults to CPU count)
        """
        self.max_workers = max_workers or min(4, (asyncio.get_event_loop().is_running() and 4) or 4)
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        logger.info(f"Parallel processor initialized with {self.max_workers} workers")
    
    async def process_batch(
        self, 
        items: List[T], 
        processor_func: Callable[[T], R], 
        batch_size: int = 10,
        progress_callback: Callable[[int, int], None] = None
    ) -> List[R]:
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
        results = [None] * total_items  # Pre-allocate results list
        
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
                results[batch_start + i] = result
            
            # Report progress if callback provided
            if progress_callback:
                progress_callback(batch_end, total_items)
            
            # Small delay between batches to prevent system overload
            if batch_end < total_items:
                await asyncio.sleep(0.01)
        
        return results
    
    async def process_chord_batch(self, chord_items: List[Any], analysis_func: Callable) -> List[dict]:
        """
        Specialized method for processing chord batches with optimized settings
        
        Args:
            chord_items: List of chord objects to analyze
            analysis_func: Cached analysis function to apply
            
        Returns:
            List of chord analysis dictionaries
        """
        def safe_analysis(item):
            """Wrapper to handle exceptions in parallel processing"""
            try:
                return analysis_func(item)
            except Exception as e:
                logger.warning(f"Parallel chord analysis failed: {e}")
                return {
                    "error": str(e),
                    "pitches": [str(p) for p in item.pitches] if hasattr(item, 'pitches') else [],
                    "failed": True
                }
        
        # Use smaller batch size for chord analysis to balance memory and performance
        return await self.process_batch(
            chord_items, 
            safe_analysis, 
            batch_size=8  # Optimized for chord analysis
        )
    
    def __del__(self):
        """Clean up executor on deletion"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)


# Global processor instance
_global_processor = None

def get_parallel_processor() -> ParallelProcessor:
    """Get the global parallel processor instance"""
    global _global_processor
    if _global_processor is None:
        _global_processor = ParallelProcessor()
    return _global_processor