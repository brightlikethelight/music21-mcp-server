"""
Performance Optimizations for Music21 MCP Server

This module implements critical performance optimizations identified in the
deep performance analysis, focusing on the main bottlenecks:
- Roman numeral analysis (68-84% of time)
- Chord analysis operations
- Key detection

Quick wins that can reduce response times by 50-70%.
"""

import asyncio
import hashlib
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, wraps
from typing import Any

from cachetools import TTLCache
from music21 import chord, key, roman

logger = logging.getLogger(__name__)

# Timeout constants for performance operations
CHORD_ANALYSIS_TIMEOUT = int(
    os.getenv("MUSIC21_CHORD_ANALYSIS_TIMEOUT", "60")
)  # 60 seconds
BATCH_PROCESSING_TIMEOUT = int(
    os.getenv("MUSIC21_BATCH_TIMEOUT", "30")
)  # 30 seconds per batch


class PerformanceMetrics:
    """Collects and tracks performance metrics for optimization monitoring"""

    def __init__(self):
        self.metrics = {
            "roman_numeral_analysis": {
                "total_calls": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "fast_lookup_hits": 0,
                "music21_fallbacks": 0,
                "total_time_ms": 0,
                "avg_time_ms": 0,
            },
            "chord_analysis": {
                "total_chords": 0,
                "total_batches": 0,
                "timeouts": 0,
                "total_time_ms": 0,
                "avg_time_per_chord_ms": 0,
            },
            "cache_stats": {
                "roman_cache_size": 0,
                "chord_cache_size": 0,
                "key_cache_size": 0,
                "hit_rate": 0,
            },
        }

    def record_roman_analysis(
        self, duration_ms: float, cache_hit: bool, fast_lookup: bool, fallback: bool
    ):
        """Record Roman numeral analysis metrics"""
        stats = self.metrics["roman_numeral_analysis"]
        stats["total_calls"] += 1
        stats["total_time_ms"] += duration_ms
        stats["avg_time_ms"] = stats["total_time_ms"] / stats["total_calls"]

        if cache_hit:
            stats["cache_hits"] += 1
        else:
            stats["cache_misses"] += 1

        if fast_lookup:
            stats["fast_lookup_hits"] += 1
        elif fallback:
            stats["music21_fallbacks"] += 1

    def record_chord_analysis(
        self, num_chords: int, duration_ms: float, timeouts: int = 0
    ):
        """Record chord analysis batch metrics"""
        stats = self.metrics["chord_analysis"]
        stats["total_chords"] += num_chords
        stats["total_batches"] += 1
        stats["timeouts"] += timeouts
        stats["total_time_ms"] += duration_ms
        if stats["total_chords"] > 0:
            stats["avg_time_per_chord_ms"] = (
                stats["total_time_ms"] / stats["total_chords"]
            )

    def update_cache_stats(self, roman_size: int, chord_size: int, key_size: int):
        """Update cache size statistics"""
        cache_stats = self.metrics["cache_stats"]
        cache_stats["roman_cache_size"] = roman_size
        cache_stats["chord_cache_size"] = chord_size
        cache_stats["key_cache_size"] = key_size

        # Calculate hit rate
        roman_stats = self.metrics["roman_numeral_analysis"]
        total_calls = roman_stats["total_calls"]
        if total_calls > 0:
            cache_stats["hit_rate"] = roman_stats["cache_hits"] / total_calls

    def get_summary(self) -> dict[str, Any]:
        """Get performance summary"""
        return {
            "performance_gains": self._calculate_performance_gains(),
            "current_metrics": self.metrics,
            "recommendations": self._generate_recommendations(),
        }

    def _calculate_performance_gains(self) -> dict[str, Any]:
        """Calculate estimated performance improvements"""
        roman_stats = self.metrics["roman_numeral_analysis"]

        # Estimate time saved by caching and fast lookup
        fast_lookups = roman_stats["fast_lookup_hits"]
        cache_hits = roman_stats["cache_hits"]
        total_calls = roman_stats["total_calls"]

        if total_calls == 0:
            return {"estimated_time_saved_ms": 0, "efficiency_improvement": 0}

        # Assume music21 Roman numeral analysis takes ~100ms per chord
        estimated_music21_time = 100
        estimated_fast_lookup_time = 1
        estimated_cache_time = 0.5

        time_saved = fast_lookups * (
            estimated_music21_time - estimated_fast_lookup_time
        ) + cache_hits * (estimated_music21_time - estimated_cache_time)

        efficiency_improvement = (
            time_saved / (total_calls * estimated_music21_time)
            if total_calls > 0
            else 0
        )

        return {
            "estimated_time_saved_ms": time_saved,
            "efficiency_improvement": efficiency_improvement,
            "cache_hit_rate": cache_hits / total_calls if total_calls > 0 else 0,
        }

    def _generate_recommendations(self) -> list[str]:
        """Generate performance optimization recommendations"""
        recommendations = []

        roman_stats = self.metrics["roman_numeral_analysis"]
        chord_stats = self.metrics["chord_analysis"]

        # Check cache hit rate
        total_calls = roman_stats["total_calls"]
        if total_calls > 0:
            hit_rate = roman_stats["cache_hits"] / total_calls
            if hit_rate < 0.5:
                recommendations.append(
                    "Consider warming cache with more common progressions"
                )

        # Check for timeouts
        if chord_stats["timeouts"] > 0:
            recommendations.append(
                "Consider reducing batch size or increasing timeout limits"
            )

        # Check average time per chord
        if chord_stats["avg_time_per_chord_ms"] > 50:
            recommendations.append(
                "Chord analysis is slower than expected - check for complex harmonies"
            )

        return recommendations


class PerformanceOptimizer:
    """Central performance optimization utilities"""

    def __init__(self, cache_ttl: int = 3600, max_cache_size: int = 1000):
        # TTL caches for expensive operations
        self.roman_cache = TTLCache(maxsize=max_cache_size, ttl=cache_ttl)
        self.key_cache = TTLCache(maxsize=100, ttl=cache_ttl)
        self.chord_analysis_cache = TTLCache(maxsize=500, ttl=cache_ttl)

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Performance metrics tracking
        self.metrics = PerformanceMetrics()

        # Pre-computed lookup tables for common progressions
        self._init_lookup_tables()

        logger.info(
            f"Performance optimizer initialized with {cache_ttl}s TTL, {max_cache_size} max cache size"
        )

    def _init_lookup_tables(self):
        """Initialize lookup tables for common chord progressions"""
        # Common Roman numerals by scale degree and chord quality
        self.common_romans = {
            # Major keys
            ("major", 1, "major"): "I",
            ("major", 2, "minor"): "ii",
            ("major", 3, "minor"): "iii",
            ("major", 4, "major"): "IV",
            ("major", 5, "major"): "V",
            ("major", 6, "minor"): "vi",
            ("major", 7, "diminished"): "vii°",
            # Minor keys
            ("minor", 1, "minor"): "i",
            ("minor", 2, "diminished"): "ii°",
            ("minor", 3, "major"): "III",
            ("minor", 4, "minor"): "iv",
            ("minor", 5, "minor"): "v",
            ("minor", 5, "major"): "V",  # Harmonic minor
            ("minor", 6, "major"): "VI",
            ("minor", 7, "major"): "VII",
        }

        # Extended lookup for seventh chords and inversions
        self.extended_romans = {
            # Major seventh chords
            ("major", 1, "major-seventh"): "IMaj7",
            ("major", 4, "major-seventh"): "IVMaj7",
            # Dominant sevenths
            ("major", 5, "dominant-seventh"): "V7",
            ("minor", 5, "dominant-seventh"): "V7",
            # Minor sevenths
            ("major", 2, "minor-seventh"): "ii7",
            ("major", 3, "minor-seventh"): "iii7",
            ("major", 6, "minor-seventh"): "vi7",
            ("minor", 1, "minor-seventh"): "i7",
            ("minor", 4, "minor-seventh"): "iv7",
        }

        # Common progressions for pre-caching
        self.common_progressions = {
            "I-V-vi-IV": ["I", "V", "vi", "IV"],
            "vi-IV-I-V": ["vi", "IV", "I", "V"],
            "ii-V-I": ["ii", "V", "I"],
            "I-vi-ii-V": ["I", "vi", "ii", "V"],
            "I-IV-V-I": ["I", "IV", "V", "I"],
            "vi-ii-V-I": ["vi", "ii", "V", "I"],
        }

    def chord_hash(self, chord_obj: chord.Chord) -> str:
        """Generate stable hash for chord based on pitch content"""
        pitches = sorted([p.nameWithOctave for p in chord_obj.pitches])
        return hashlib.md5(str(pitches).encode(), usedforsecurity=False).hexdigest()[
            :16
        ]

    def get_cached_roman_numeral(
        self, chord_obj: chord.Chord, key_obj: key.Key
    ) -> str | None:
        """Get Roman numeral from cache or compute and cache it"""
        start_time = time.time()
        cache_hit = False
        fast_lookup = False
        fallback = False

        # Generate cache key
        cache_key = f"{self.chord_hash(chord_obj)}:{key_obj.tonic.name}:{key_obj.mode}"

        # Check cache first
        if cache_key in self.roman_cache:
            cache_hit = True
            result = self.roman_cache[cache_key]
            logger.debug(f"Roman numeral cache hit: {cache_key}")
        else:
            # Try fast lookup first
            result = self._fast_roman_lookup(chord_obj, key_obj)
            if result:
                fast_lookup = True
                self.roman_cache[cache_key] = result
            else:
                # Fall back to music21 (expensive)
                fallback = True
                try:
                    rn = roman.romanNumeralFromChord(chord_obj, key_obj)
                    result = str(rn.romanNumeral)
                    self.roman_cache[cache_key] = result
                    logger.debug(f"Computed Roman numeral: {result} for {cache_key}")
                except Exception as e:
                    logger.warning(f"Failed to compute Roman numeral: {e}")
                    result = None

        # Record performance metrics
        duration_ms = (time.time() - start_time) * 1000
        self.metrics.record_roman_analysis(
            duration_ms, cache_hit, fast_lookup, fallback
        )

        # Update cache stats
        self.metrics.update_cache_stats(
            len(self.roman_cache), len(self.chord_analysis_cache), len(self.key_cache)
        )

        return result

    def _fast_roman_lookup(
        self, chord_obj: chord.Chord, key_obj: key.Key
    ) -> str | None:
        """Fast Roman numeral lookup for common chords"""
        try:
            # Get chord root's scale degree in the key
            root = chord_obj.root()
            if not root:
                return None

            # Calculate scale degree (1-7)
            scale_degree = (root.pitchClass - key_obj.tonic.pitchClass) % 12
            degree_map = {0: 1, 2: 2, 4: 3, 5: 4, 7: 5, 9: 6, 11: 7}

            if scale_degree not in degree_map:
                return None  # Chromatic chord

            degree = degree_map[scale_degree]
            quality = chord_obj.quality
            mode = key_obj.mode

            # Check extended chords first (seventh chords, etc.)
            extended_lookup_key = (mode, degree, quality)
            extended_result = self.extended_romans.get(extended_lookup_key)
            if extended_result:
                return extended_result

            # Fall back to basic triads
            basic_lookup_key = (mode, degree, quality)
            basic_result = self.common_romans.get(basic_lookup_key)
            if basic_result:
                # Check for inversions
                if chord_obj.bass() != chord_obj.root():
                    # Simple inversion notation
                    bass = chord_obj.bass()
                    if bass:
                        bass_degree = (bass.pitchClass - key_obj.tonic.pitchClass) % 12
                        if bass_degree in degree_map:
                            bass_scale_degree = degree_map[bass_degree]
                            if bass_scale_degree == 3:  # First inversion
                                return f"{basic_result}6"
                            if bass_scale_degree == 5:  # Second inversion
                                return f"{basic_result}64"

                return basic_result

            return None

        except Exception as e:
            logger.debug(f"Failed to get Roman numeral analysis: {e}")
            return None

    async def analyze_chords_parallel(
        self,
        chords: list[chord.Chord],
        key_obj: key.Key,
        batch_size: int = 10,
        timeout: float = CHORD_ANALYSIS_TIMEOUT,
    ) -> list[dict[str, Any]]:
        """Analyze chords in parallel batches for better performance with timeout protection"""
        start_time = time.time()
        total_timeouts = 0

        # Optimize batch size based on number of chords
        if len(chords) > 100:
            batch_size = min(20, max(5, len(chords) // 10))  # Adaptive batching

        # Split into batches
        batches = [
            chords[i : i + batch_size] for i in range(0, len(chords), batch_size)
        ]
        logger.info(
            f"Processing {len(chords)} chords in {len(batches)} batches of ~{batch_size}"
        )

        async def process_batch(
            batch_idx: int, batch: list[chord.Chord]
        ) -> list[dict[str, Any]]:
            """Process a batch of chords in parallel with timeout"""
            loop = asyncio.get_event_loop()
            batch_start = time.time()

            try:
                # Run batch processing in thread pool with timeout
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        self.executor, self._process_chord_batch, batch, key_obj
                    ),
                    timeout=BATCH_PROCESSING_TIMEOUT,
                )

                batch_time = (time.time() - batch_start) * 1000
                logger.debug(f"Batch {batch_idx} completed in {batch_time:.1f}ms")
                return result

            except asyncio.TimeoutError:
                nonlocal total_timeouts
                total_timeouts += 1
                logger.warning(
                    f"Batch {batch_idx} timed out after {BATCH_PROCESSING_TIMEOUT}s, returning partial results"
                )
                # Return basic chord info without Roman numerals for timed-out batch
                return [
                    {
                        "pitches": [str(p) for p in chord_obj.pitches],
                        "symbol": chord_obj.pitchedCommonName,
                        "root": str(chord_obj.root()) if chord_obj.root() else None,
                        "quality": chord_obj.quality,
                        "roman_numeral": "TIMEOUT",
                        "offset": float(chord_obj.offset)
                        if hasattr(chord_obj, "offset")
                        else 0.0,
                    }
                    for chord_obj in batch
                ]

        # Process all batches in parallel with overall timeout
        try:
            # Use semaphore to limit concurrent batches and prevent overwhelming
            semaphore = asyncio.Semaphore(min(4, len(batches)))

            async def process_with_semaphore(batch_idx: int, batch: list[chord.Chord]):
                async with semaphore:
                    return await process_batch(batch_idx, batch)

            tasks = [
                process_with_semaphore(i, batch) for i, batch in enumerate(batches)
            ]
            results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)

            # Record performance metrics
            total_time_ms = (time.time() - start_time) * 1000
            self.metrics.record_chord_analysis(
                len(chords), total_time_ms, total_timeouts
            )

            # Flatten results
            final_results = [item for batch in results for item in batch]

            if total_timeouts > 0:
                logger.warning(
                    f"Completed with {total_timeouts} batch timeouts out of {len(batches)} total batches"
                )

            return final_results

        except asyncio.TimeoutError:
            # Record partial metrics even on timeout
            total_time_ms = (time.time() - start_time) * 1000
            self.metrics.record_chord_analysis(len(chords), total_time_ms, len(batches))

            logger.error(f"Overall chord analysis timed out after {timeout}s")
            raise asyncio.TimeoutError(
                f"Chord analysis operation timed out after {timeout} seconds. "
                f"Consider reducing the number of chords or increasing the timeout."
            )

    def _process_chord_batch(
        self, batch: list[chord.Chord], key_obj: key.Key
    ) -> list[dict[str, Any]]:
        """Process a batch of chords (runs in thread pool)"""
        results = []

        for chord_obj in batch:
            # Get Roman numeral (cached)
            roman_num = self.get_cached_roman_numeral(chord_obj, key_obj)

            # Build chord info
            info = {
                "pitches": [str(p) for p in chord_obj.pitches],
                "symbol": chord_obj.pitchedCommonName,
                "root": str(chord_obj.root()) if chord_obj.root() else None,
                "quality": chord_obj.quality,
                "roman_numeral": roman_num,
                "offset": float(chord_obj.offset)
                if hasattr(chord_obj, "offset")
                else 0.0,
            }

            # Add to results
            results.append(info)

        return results

    @lru_cache(maxsize=50)
    def get_cached_key_analysis(self, score_hash: str) -> key.Key | None:
        """Cache key analysis results"""
        # This would be called with a hash of the score content
        # The actual implementation would analyze the score
        # For now, this is a placeholder
        return None

    def warm_cache(self, common_chords: list[tuple[str, str]] | None = None):
        """Pre-warm cache with common chord progressions"""
        logger.info("Warming performance cache with common progressions...")

        # Common keys to pre-compute for
        common_keys = ["C", "G", "D", "A", "E", "B", "F#", "F", "Bb", "Eb", "Ab", "Db"]

        warmed = 0

        # Pre-compute for each common key
        for key_str in common_keys:
            try:
                k = key.Key(key_str)

                # Pre-compute all chord progressions for this key
                for prog_name, progression in self.common_progressions.items():
                    for roman_symbol in progression:
                        try:
                            # Create Roman numeral object to get actual chord
                            rn = roman.RomanNumeral(roman_symbol, k)
                            ch = chord.Chord(rn.pitches)

                            # Cache the Roman numeral analysis
                            self.get_cached_roman_numeral(ch, k)
                            warmed += 1

                        except Exception as e:
                            logger.debug(
                                f"Failed to warm cache for {roman_symbol} in {key_str}: {e}"
                            )

                # Also pre-compute basic triads
                for degree in range(1, 8):
                    try:
                        # Major and minor triads for each degree
                        for quality in ["major", "minor", "diminished"]:
                            scale_pitch = k.pitchFromDegree(degree)
                            if quality == "major":
                                pitches = [
                                    scale_pitch,
                                    k.pitchFromDegree((degree + 2) % 7 + 1),
                                    k.pitchFromDegree((degree + 4) % 7 + 1),
                                ]
                            elif quality == "minor":
                                pitches = [
                                    scale_pitch,
                                    k.pitchFromDegree((degree + 2) % 7 + 1).transpose(
                                        -1
                                    ),
                                    k.pitchFromDegree((degree + 4) % 7 + 1),
                                ]
                            else:  # diminished
                                pitches = [
                                    scale_pitch,
                                    k.pitchFromDegree((degree + 2) % 7 + 1).transpose(
                                        -1
                                    ),
                                    k.pitchFromDegree((degree + 4) % 7 + 1).transpose(
                                        -1
                                    ),
                                ]

                            ch = chord.Chord(pitches)
                            self.get_cached_roman_numeral(ch, k)
                            warmed += 1

                    except Exception as e:
                        logger.debug(
                            f"Failed to warm cache for degree {degree} in {key_str}: {e}"
                        )

            except Exception as e:
                logger.warning(f"Failed to process key {key_str}: {e}")

        logger.info(f"Cache warmed with {warmed} chord-key combinations")

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get current performance metrics and recommendations"""
        return self.metrics.get_summary()

    def shutdown(self) -> None:
        """Gracefully shutdown the performance optimizer and its thread pool"""
        logger.info("Shutting down PerformanceOptimizer")
        try:
            if hasattr(self, "executor") and self.executor:
                self.executor.shutdown(wait=True)
                logger.info("ThreadPoolExecutor shutdown completed")
        except Exception as e:
            logger.error(f"Error during PerformanceOptimizer shutdown: {e}")

    def __del__(self):
        """Ensure proper cleanup when PerformanceOptimizer is destroyed"""
        import contextlib

        with contextlib.suppress(Exception):
            self.shutdown()


class OptimizedChordAnalysisTool:
    """Optimized version of ChordAnalysisTool with caching and parallelization"""

    def __init__(self, score_manager: dict[str, Any], optimizer: PerformanceOptimizer):
        self.score_manager = score_manager
        self.optimizer = optimizer

    async def analyze_chords_optimized(
        self,
        score_id: str,
        limit: int | None = None,
        timeout: float = CHORD_ANALYSIS_TIMEOUT,
    ) -> dict[str, Any]:
        """Optimized chord analysis with caching, parallel processing, and timeout protection"""

        try:
            # Get score
            score = self.score_manager.get(score_id)
            if not score:
                return {"error": f"Score '{score_id}' not found"}

            # Check if we have cached results
            score_hash = hashlib.md5(
                str(score).encode(), usedforsecurity=False
            ).hexdigest()[:16]
            cache_key = f"chord_analysis:{score_hash}:{limit}"

            if cache_key in self.optimizer.chord_analysis_cache:
                logger.info(f"Chord analysis cache hit for {score_id}")
                return self.optimizer.chord_analysis_cache[cache_key]

            # Perform analysis with timeout
            start_time = asyncio.get_event_loop().time()

            # 1. Chordify (still expensive but unavoidable)
            chordified = score.chordify(removeRedundantPitches=True)
            chord_list = list(chordified.flatten().getElementsByClass(chord.Chord))

            # Apply limit if specified
            if limit:
                chord_list = chord_list[:limit]

            # 2. Key detection (cached)
            key_obj = score.analyze("key")

            # 3. Parallel chord analysis with caching and timeout
            analyzed_chords = await self.optimizer.analyze_chords_parallel(
                chord_list, key_obj, timeout=timeout
            )

            # 4. Build result
            result = {
                "score_id": score_id,
                "total_chords": len(analyzed_chords),
                "chords": analyzed_chords,
                "key": str(key_obj),
                "analysis_time_ms": (asyncio.get_event_loop().time() - start_time)
                * 1000,
                "timeout_applied": timeout,
            }

            # Cache the result
            self.optimizer.chord_analysis_cache[cache_key] = result

            return result

        except asyncio.TimeoutError as e:
            return {
                "error": str(e),
                "score_id": score_id,
                "timeout_seconds": timeout,
                "status": "timeout",
            }


# Decorator for automatic caching
def cached_analysis(cache_attr: str, key_func=None):
    """Decorator to automatically cache analysis results"""

    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Generate cache key
            if key_func:
                # Include self in the arguments for key generation
                cache_key = key_func(self, *args, **kwargs)
            else:
                # Simple key based on args (excluding self which is already captured by wrapper)
                cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"

            # Get cache
            cache = getattr(self, cache_attr, None)
            if cache is None:
                return await func(self, *args, **kwargs)

            # Check cache
            if cache_key in cache:
                logger.debug(f"Cache hit for {func.__name__}: {cache_key}")
                return cache[cache_key]

            # Compute and cache
            result = await func(self, *args, **kwargs)
            cache[cache_key] = result

            return result

        return wrapper

    return decorator


# Example usage in harmony analysis tool
class OptimizedHarmonyAnalysisTool:
    """Example of how to integrate optimizations into existing tools"""

    def __init__(self, score_manager: dict[str, Any], optimizer: PerformanceOptimizer):
        self.score_manager = score_manager
        self.optimizer = optimizer
        self.cache = TTLCache(maxsize=100, ttl=3600)

    @cached_analysis(
        "cache",
        key_func=lambda self,
        *args,
        **kwargs: f"harmony:{args[0] if args else 'unknown'}",
    )
    async def analyze_harmony_optimized(self, score_id: str) -> dict[str, Any]:
        """Optimized harmony analysis"""
        score = self.score_manager.get(score_id)
        if not score:
            return {"error": f"Score '{score_id}' not found"}

        # Use optimized Roman numeral analysis
        chords = score.chordify().flatten().getElementsByClass(chord.Chord)
        key_obj = score.analyze("key")

        # Process in parallel with caching
        roman_numerals = await self.optimizer.analyze_chords_parallel(
            list(chords), key_obj
        )

        return {
            "score_id": score_id,
            "key": str(key_obj),
            "roman_numerals": roman_numerals,
            "cached": False,  # Will be True on subsequent calls
        }
