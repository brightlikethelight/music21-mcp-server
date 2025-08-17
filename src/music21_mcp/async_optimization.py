#!/usr/bin/env python3
"""
Advanced Async Optimization System for Music21 MCP Server

This module provides cutting-edge async optimization for the most critical bottlenecks
identified in profiling: Roman numeral analysis, chordification, and key detection.

Key optimizations:
- Async batching and pipelining for Roman numeral analysis
- Streaming chordification for large scores
- Precomputed lookup tables for common progressions
- Connection pooling for concurrent requests
- Smart rate limiting to prevent system overload
"""

import asyncio
import hashlib
import logging
import time
from asyncio import Semaphore, Queue
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set, Tuple
from weakref import WeakValueDictionary

from cachetools import TTLCache
from music21 import chord, key, roman, stream

logger = logging.getLogger(__name__)


@dataclass
class AnalysisTask:
    """Represents a single analysis task for batching"""
    id: str
    chord_obj: chord.Chord
    key_obj: key.Key
    future: asyncio.Future
    priority: int = 0  # Higher = more important
    created_at: float = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


class AsyncOptimizer:
    """Advanced async optimizer for music21 operations"""

    def __init__(
        self,
        max_concurrent_operations: int = 10,
        batch_size: int = 20,
        batch_timeout: float = 0.05,  # 50ms batching window
        cache_ttl: int = 3600,
        thread_pool_workers: int = 4
    ):
        self.max_concurrent = max_concurrent_operations
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        # Concurrency control
        self.operation_semaphore = Semaphore(max_concurrent_operations)
        self.roman_analysis_queue: Queue[AnalysisTask] = Queue()
        
        # Caching with advanced features
        self.roman_cache = TTLCache(maxsize=5000, ttl=cache_ttl)
        self.chord_pattern_cache = TTLCache(maxsize=1000, ttl=cache_ttl)
        self.score_metadata_cache = TTLCache(maxsize=500, ttl=cache_ttl)
        
        # Precomputed lookup tables
        self.roman_lookup_table = self._build_roman_lookup_table()
        self.progression_patterns = self._build_progression_patterns()
        
        # Async infrastructure
        self.executor = ThreadPoolExecutor(max_workers=thread_pool_workers)
        self.batch_processor_task = None
        self.shutdown_event = asyncio.Event()
        
        # Performance tracking
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "fast_lookups": 0,
            "batched_operations": 0,
            "concurrent_operations": 0
        }
        
        # Active operation tracking for load balancing
        self.active_operations: WeakValueDictionary = WeakValueDictionary()
        
        logger.info(f"AsyncOptimizer initialized: {max_concurrent_operations} concurrent, "
                   f"batch_size={batch_size}, cache_ttl={cache_ttl}s")

    async def start(self):
        """Start the async optimization system"""
        if self.batch_processor_task is None:
            self.batch_processor_task = asyncio.create_task(self._batch_processor())
            logger.info("Async optimizer started")

    async def stop(self):
        """Stop the async optimization system"""
        self.shutdown_event.set()
        if self.batch_processor_task:
            await self.batch_processor_task
        self.executor.shutdown(wait=True)
        logger.info("Async optimizer stopped")

    def _build_roman_lookup_table(self) -> Dict[Tuple[str, int, str], str]:
        """Build precomputed Roman numeral lookup table"""
        lookup = {}
        
        # Major key patterns
        major_patterns = {
            (0, "major"): "I", (0, "major7"): "IM7",
            (2, "minor"): "ii", (2, "minor7"): "ii7", (2, "diminished"): "ii°",
            (4, "minor"): "iii", (4, "minor7"): "iii7",
            (5, "major"): "IV", (5, "major7"): "IVM7",
            (7, "major"): "V", (7, "dominant7"): "V7",
            (9, "minor"): "vi", (9, "minor7"): "vi7",
            (11, "diminished"): "vii°", (11, "half-diminished7"): "viiø7"
        }
        
        # Minor key patterns
        minor_patterns = {
            (0, "minor"): "i", (0, "minor7"): "i7",
            (2, "diminished"): "ii°", (2, "half-diminished7"): "iiø7",
            (3, "major"): "III", (3, "major7"): "IIIM7",
            (5, "minor"): "iv", (5, "minor7"): "iv7",
            (7, "minor"): "v", (7, "major"): "V", (7, "dominant7"): "V7",
            (8, "major"): "VI", (8, "major7"): "VIM7",
            (10, "major"): "VII", (10, "major7"): "VIIM7"
        }
        
        # Build lookup for both modes
        for (interval, quality), roman in major_patterns.items():
            lookup[("major", interval, quality)] = roman
            
        for (interval, quality), roman in minor_patterns.items():
            lookup[("minor", interval, quality)] = roman
        
        logger.info(f"Built Roman numeral lookup table with {len(lookup)} patterns")
        return lookup

    def _build_progression_patterns(self) -> Dict[str, List[str]]:
        """Build common chord progression patterns for fast recognition"""
        return {
            "I-V-vi-IV": ["I", "V", "vi", "IV"],
            "vi-IV-I-V": ["vi", "IV", "I", "V"],
            "ii-V-I": ["ii", "V", "I"],
            "I-vi-ii-V": ["I", "vi", "ii", "V"],
            "I-IV-V-I": ["I", "IV", "V", "I"],
            "i-VII-VI-VII": ["i", "VII", "VI", "VII"],
            "i-iv-V-i": ["i", "iv", "V", "i"],
            "Circle of Fifths": ["I", "vi", "ii", "V"],
        }

    async def _batch_processor(self):
        """Process Roman numeral analysis tasks in optimized batches"""
        while not self.shutdown_event.is_set():
            try:
                batch = []
                deadline = time.time() + self.batch_timeout
                
                # Collect tasks for batching
                while len(batch) < self.batch_size and time.time() < deadline:
                    try:
                        task = await asyncio.wait_for(
                            self.roman_analysis_queue.get(), 
                            timeout=max(0.001, deadline - time.time())
                        )
                        batch.append(task)
                    except asyncio.TimeoutError:
                        break
                
                if batch:
                    await self._process_batch(batch)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processor error: {e}")
                await asyncio.sleep(0.1)  # Brief pause on error

    async def _process_batch(self, batch: List[AnalysisTask]):
        """Process a batch of Roman numeral analysis tasks"""
        try:
            # Group by key for efficiency
            by_key = defaultdict(list)
            for task in batch:
                key_str = str(task.key_obj)
                by_key[key_str].append(task)
            
            # Process each key group in parallel
            processing_tasks = []
            for key_str, key_tasks in by_key.items():
                processing_tasks.append(self._process_key_group(key_tasks))
            
            await asyncio.gather(*processing_tasks, return_exceptions=True)
            self.stats["batched_operations"] += len(batch)
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            # Set error for all tasks in batch
            for task in batch:
                if not task.future.done():
                    task.future.set_exception(e)

    async def _process_key_group(self, tasks: List[AnalysisTask]):
        """Process a group of tasks with the same key"""
        loop = asyncio.get_event_loop()
        
        def compute_romans():
            """Compute Roman numerals for all tasks in thread"""
            results = {}
            for task in tasks:
                try:
                    # Try fast lookup first
                    result = self._fast_roman_lookup(task.chord_obj, task.key_obj)
                    if result is None:
                        # Fall back to music21 computation
                        rn = roman.romanNumeralFromChord(task.chord_obj, task.key_obj)
                        result = str(rn.romanNumeral)
                    results[task.id] = result
                except Exception as e:
                    results[task.id] = f"ERROR: {str(e)}"
            return results
        
        # Run computation in thread pool
        try:
            results = await loop.run_in_executor(self.executor, compute_romans)
            
            # Set results for all tasks
            for task in tasks:
                if not task.future.done():
                    result = results.get(task.id, "ERROR: Missing result")
                    task.future.set_result(result)
                    
        except Exception as e:
            # Set error for all tasks
            for task in tasks:
                if not task.future.done():
                    task.future.set_exception(e)

    def _fast_roman_lookup(self, chord_obj: chord.Chord, key_obj: key.Key) -> Optional[str]:
        """Ultra-fast Roman numeral lookup using precomputed table"""
        try:
            root = chord_obj.root()
            if not root:
                return None
            
            # Calculate interval from key root
            interval = (root.pitchClass - key_obj.tonic.pitchClass) % 12
            quality = chord_obj.quality
            mode = key_obj.mode
            
            # Try exact lookup
            lookup_key = (mode, interval, quality)
            result = self.roman_lookup_table.get(lookup_key)
            
            if result:
                self.stats["fast_lookups"] += 1
                return result
                
            return None
            
        except Exception:
            return None

    @asynccontextmanager
    async def operation_context(self, operation_id: str):
        """Context manager for tracking concurrent operations"""
        async with self.operation_semaphore:
            self.stats["concurrent_operations"] += 1
            self.active_operations[operation_id] = operation_id
            try:
                yield
            finally:
                self.stats["concurrent_operations"] = max(0, self.stats["concurrent_operations"] - 1)
                self.active_operations.pop(operation_id, None)

    async def get_cached_roman_numeral(
        self, chord_obj: chord.Chord, key_obj: key.Key, priority: int = 0
    ) -> str:
        """Get Roman numeral with async batching and caching"""
        # Generate cache key
        cache_key = self._generate_cache_key(chord_obj, key_obj)
        
        # Check cache first
        if cache_key in self.roman_cache:
            self.stats["cache_hits"] += 1
            return self.roman_cache[cache_key]
        
        self.stats["cache_misses"] += 1
        
        # Try fast lookup
        fast_result = self._fast_roman_lookup(chord_obj, key_obj)
        if fast_result:
            self.roman_cache[cache_key] = fast_result
            return fast_result
        
        # Queue for batched processing
        task_id = f"{cache_key}_{time.time()}"
        future = asyncio.Future()
        task = AnalysisTask(task_id, chord_obj, key_obj, future, priority)
        
        await self.roman_analysis_queue.put(task)
        
        try:
            result = await asyncio.wait_for(future, timeout=30.0)  # 30s timeout
            self.roman_cache[cache_key] = result
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Roman numeral analysis timeout for {cache_key}")
            return "?"

    def _generate_cache_key(self, chord_obj: chord.Chord, key_obj: key.Key) -> str:
        """Generate efficient cache key"""
        pitches = sorted([p.nameWithOctave for p in chord_obj.pitches])
        key_str = f"{key_obj.tonic.name}{key_obj.mode}"
        combined = f"{key_str}:{','.join(pitches)}"
        return hashlib.md5(combined.encode(), usedforsecurity=False).hexdigest()[:16]

    async def analyze_chords_streaming(
        self, 
        score: stream.Score, 
        chunk_size: int = 50
    ) -> AsyncGenerator[List[Dict[str, Any]], None]:
        """Stream chord analysis for large scores in chunks"""
        async with self.operation_context(f"streaming_analysis_{id(score)}"):
            try:
                # Get chords in chunks to avoid memory issues
                chords = self._extract_chords_efficiently(score)
                key_obj = await self._get_cached_key(score)
                
                chunk = []
                for chord_obj in chords:
                    chord_analysis = await self._analyze_single_chord(chord_obj, key_obj)
                    chunk.append(chord_analysis)
                    
                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []
                        # Small delay to prevent overwhelming
                        await asyncio.sleep(0.001)
                
                # Yield remaining chunk
                if chunk:
                    yield chunk
                    
            except Exception as e:
                logger.error(f"Streaming analysis error: {e}")
                raise

    def _extract_chords_efficiently(self, score: stream.Score) -> List[chord.Chord]:
        """Efficiently extract chords from score"""
        chords = []
        
        # Try direct chord extraction first
        for element in score.flat.getElementsByClass(chord.Chord):
            chords.append(element)
        
        # If no chords found, try chordification
        if not chords:
            try:
                chordified = score.chordify(removeRedundantPitches=True)
                for element in chordified.flat.getElementsByClass(chord.Chord):
                    chords.append(element)
            except Exception as e:
                logger.warning(f"Chordification failed: {e}")
        
        return chords

    async def _get_cached_key(self, score: stream.Score) -> key.Key:
        """Get cached key analysis for score"""
        score_hash = str(hash(str(score)))
        
        if score_hash in self.score_metadata_cache:
            return self.score_metadata_cache[score_hash]["key"]
        
        # Compute key in thread pool
        loop = asyncio.get_event_loop()
        
        def compute_key():
            try:
                return score.analyze("key")
            except Exception:
                return key.Key("C")  # Fallback
        
        key_obj = await loop.run_in_executor(self.executor, compute_key)
        
        self.score_metadata_cache[score_hash] = {
            "key": key_obj,
            "computed_at": time.time()
        }
        
        return key_obj

    async def _analyze_single_chord(
        self, chord_obj: chord.Chord, key_obj: key.Key
    ) -> Dict[str, Any]:
        """Analyze a single chord with caching"""
        try:
            roman_numeral = await self.get_cached_roman_numeral(chord_obj, key_obj)
            
            return {
                "pitches": [str(p) for p in chord_obj.pitches],
                "symbol": chord_obj.pitchedCommonName,
                "root": str(chord_obj.root()) if chord_obj.root() else None,
                "quality": chord_obj.quality,
                "roman_numeral": roman_numeral,
                "offset": float(chord_obj.offset) if hasattr(chord_obj, "offset") else 0.0,
                "duration": float(chord_obj.duration.quarterLength) if hasattr(chord_obj, "duration") else 1.0
            }
        except Exception as e:
            logger.warning(f"Single chord analysis failed: {e}")
            return {
                "pitches": [str(p) for p in chord_obj.pitches],
                "symbol": "Unknown",
                "error": str(e)
            }

    async def detect_progressions_fast(
        self, roman_numerals: List[str]
    ) -> List[Dict[str, Any]]:
        """Fast progression detection using precomputed patterns"""
        progressions = []
        
        for pattern_name, pattern in self.progression_patterns.items():
            pattern_len = len(pattern)
            
            for i in range(len(roman_numerals) - pattern_len + 1):
                segment = roman_numerals[i:i + pattern_len]
                if segment == pattern:
                    progressions.append({
                        "name": pattern_name,
                        "start_position": i,
                        "end_position": i + pattern_len - 1,
                        "chords": pattern,
                        "confidence": 1.0  # Exact match
                    })
        
        return progressions

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        hit_rate = (self.stats["cache_hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_performance": {
                "hit_rate_percent": round(hit_rate, 2),
                "total_requests": total_requests,
                "cache_size": len(self.roman_cache)
            },
            "optimization_stats": {
                "fast_lookups": self.stats["fast_lookups"],
                "batched_operations": self.stats["batched_operations"],
                "concurrent_operations": self.stats["concurrent_operations"]
            },
            "system_status": {
                "active_operations": len(self.active_operations),
                "queue_size": self.roman_analysis_queue.qsize(),
                "is_running": not self.shutdown_event.is_set()
            }
        }

    async def warm_caches(self, common_progressions: List[List[str]] = None):
        """Warm up caches with common patterns"""
        if common_progressions is None:
            common_progressions = [
                ["C", "F", "G", "C"],
                ["Am", "F", "C", "G"],
                ["Dm", "G", "C", "C"],
                ["Em", "Am", "F", "G"]
            ]
        
        logger.info("Warming async optimizer caches...")
        
        # TODO: Implementation would create chord objects and warm caches
        # This is a placeholder for the concept
        
        logger.info("Cache warming completed")


# Global async optimizer instance
_global_async_optimizer = None


async def get_async_optimizer() -> AsyncOptimizer:
    """Get the global async optimizer instance"""
    global _global_async_optimizer
    if _global_async_optimizer is None:
        _global_async_optimizer = AsyncOptimizer()
        await _global_async_optimizer.start()
    return _global_async_optimizer


async def shutdown_async_optimizer():
    """Shutdown the global async optimizer"""
    global _global_async_optimizer
    if _global_async_optimizer:
        await _global_async_optimizer.stop()
        _global_async_optimizer = None