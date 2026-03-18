"""Tests for performance_cache module"""

import time

import pytest
from music21 import chord, key, stream

from music21_mcp.performance_cache import (
    PerformanceCache,
    cached_analysis,
    clear_performance_cache,
    get_performance_cache,
)


class TestPerformanceCache:
    """Tests for PerformanceCache"""

    def setup_method(self):
        self.cache = PerformanceCache(max_size=100, ttl_seconds=300)

    def test_roman_numeral_cache_roundtrip(self):
        c = chord.Chord(["C4", "E4", "G4"])
        k = key.Key("C")
        self.cache.cache_roman_numeral(c, k, "I")
        assert self.cache.get_cached_roman_numeral(c, k) == "I"

    def test_roman_numeral_cache_miss(self):
        c = chord.Chord(["D4", "F#4", "A4"])
        k = key.Key("G")
        assert self.cache.get_cached_roman_numeral(c, k) is None

    def test_key_analysis_cache_roundtrip(self):
        score = stream.Score()
        k = key.Key("A")
        self.cache.cache_key_analysis(score, k)
        assert self.cache.get_cached_key_analysis(score) == k

    def test_chord_analysis_cache_roundtrip(self):
        c = chord.Chord(["E4", "G4", "B4"])
        analysis = {"root": "E", "quality": "minor"}
        self.cache.cache_chord_analysis(c, analysis)
        assert self.cache.get_cached_chord_analysis(c) == analysis

    def test_chord_analysis_cache_miss(self):
        c = chord.Chord(["F4", "A4", "C5"])
        assert self.cache.get_cached_chord_analysis(c) is None

    def test_get_roman_numeral_computes_and_caches(self):
        c = chord.Chord(["C4", "E4", "G4"])
        k = key.Key("C")
        result = self.cache.get_roman_numeral(c, k)
        assert result is not None
        assert result[0] == "I"
        assert result[1] == 1
        # Second call should be a cache hit
        old_misses = self.cache._misses
        result2 = self.cache.get_roman_numeral(c, k)
        assert result2 == result
        assert self.cache._misses == old_misses  # no new miss

    def test_cache_stats(self):
        c = chord.Chord(["C4", "E4", "G4"])
        k = key.Key("C")
        self.cache.get_roman_numeral(c, k)  # miss
        self.cache.get_roman_numeral(c, k)  # hit
        stats = self.cache.get_cache_stats()
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1
        assert "hit_rate_percent" in stats
        assert stats["roman_numeral_cache_size"] >= 1

    def test_clear_all_caches(self):
        c = chord.Chord(["C4", "E4", "G4"])
        k = key.Key("C")
        self.cache.cache_roman_numeral(c, k, "I")
        self.cache.clear_all_caches()
        assert self.cache._hits == 0
        assert self.cache._misses == 0
        assert self.cache.get_cached_roman_numeral(c, k) is None

    def test_get_chord_analysis_full(self):
        c = chord.Chord(["C4", "E4", "G4"])
        k = key.Key("C")
        result = self.cache.get_chord_analysis(c, k, include_inversions=True)
        assert "pitches" in result
        assert "root" in result
        assert result["root"] is not None
        # Second call should be cached
        old_misses = self.cache._misses
        result2 = self.cache.get_chord_analysis(c, k, include_inversions=True)
        assert result2 == result
        assert self.cache._misses == old_misses


class TestCachedAnalysisDecorator:
    """Tests for cached_analysis decorator"""

    def test_caches_function_results(self):
        call_count = 0

        @cached_analysis("test_cache")
        def expensive(arg):
            nonlocal call_count
            call_count += 1
            return f"result_{arg}"

        r1 = expensive("a")
        r2 = expensive("a")
        assert r1 == r2 == "result_a"
        assert call_count == 1  # only called once

    def test_different_args_not_cached(self):
        @cached_analysis("test_cache")
        def fn(x):
            return x * 2

        assert fn(1) == 2
        assert fn(2) == 4


class TestCacheErrorPaths:
    """Tests for error handling in cache operations."""

    def test_get_roman_numeral_analysis_failure(self):
        """get_roman_numeral caches None on analysis failure."""
        from unittest.mock import MagicMock

        cache = PerformanceCache(max_size=100, ttl_seconds=300)
        # Create a chord-like object that will cause roman numeral analysis to fail
        fake_chord = MagicMock()
        fake_chord.pitches = [MagicMock(__str__=lambda s: "C4")]
        fake_chord.inversion.return_value = 0
        fake_key = MagicMock()
        fake_key.__str__ = lambda s: "C major"

        result = cache.get_roman_numeral(fake_chord, fake_key)
        # Should return None (failure cached)
        assert result is None
        assert cache._misses == 1

    def test_get_chord_analysis_fallback(self):
        """get_chord_analysis returns fallback dict on failure."""
        from unittest.mock import MagicMock, PropertyMock

        cache = PerformanceCache(max_size=100, ttl_seconds=300)
        # Create a chord that will fail during detailed analysis but has basic attrs
        fake_chord = MagicMock()
        fake_chord.pitches = [MagicMock(__str__=lambda s: "C4")]
        fake_chord.inversion.return_value = 0
        fake_chord.offset = 0.0
        fake_chord.duration.quarterLength = 1.0
        # Make pitchedCommonName raise to trigger the except block
        type(fake_chord).pitchedCommonName = PropertyMock(side_effect=Exception("fail"))

        result = cache.get_chord_analysis(fake_chord, None, include_inversions=False)
        assert result["symbol"] == "Unknown"
        assert "pitches" in result


class TestCacheEvictionAndTTL:
    """Tests for eviction, TTL expiry, and clear paths."""

    def test_cache_eviction_on_max_size(self):
        """Filling cache beyond max_size evicts oldest entries."""
        cache = PerformanceCache(max_size=2, ttl_seconds=300)
        c1 = chord.Chord(["C4", "E4", "G4"])
        c2 = chord.Chord(["D4", "F#4", "A4"])
        c3 = chord.Chord(["E4", "G#4", "B4"])
        k = key.Key("C")
        cache.cache_roman_numeral(c1, k, "I")
        cache.cache_roman_numeral(c2, k, "ii")
        # Cache is full (max_size=2); adding a third should evict the oldest
        cache.cache_roman_numeral(c3, k, "iii")
        assert len(cache._roman_numeral_cache) <= 2
        # The newest entry should still be present
        assert cache.get_cached_roman_numeral(c3, k) == "iii"

    def test_cache_ttl_expiry(self):
        """Entries expire after TTL elapses."""
        cache = PerformanceCache(max_size=100, ttl_seconds=1)
        c = chord.Chord(["C4", "E4", "G4"])
        k = key.Key("C")
        cache.cache_roman_numeral(c, k, "I")
        assert cache.get_cached_roman_numeral(c, k) == "I"
        # Wait for TTL to expire
        time.sleep(1.1)
        assert cache.get_cached_roman_numeral(c, k) is None

    def test_clear_cache(self):
        """clear_performance_cache empties all global cache buckets."""
        import music21_mcp.performance_cache as mod

        mod._global_cache = None
        pc = get_performance_cache()
        c = chord.Chord(["C4", "E4", "G4"])
        k = key.Key("C")
        pc.cache_roman_numeral(c, k, "I")
        pc.cache_chord_analysis(c, {"root": "C"})
        pc.cache_key_analysis(stream.Score(), k)
        clear_performance_cache()
        assert len(pc._roman_numeral_cache) == 0
        assert len(pc._chord_analysis_cache) == 0
        assert len(pc._key_analysis_cache) == 0
        assert pc._hits == 0
        assert pc._misses == 0
        mod._global_cache = None


class TestCachedAnalysisDecoratorExtended:
    """Extended tests for the cached_analysis decorator on functions and methods."""

    def test_cached_analysis_decorator_function(self):
        """Decorator caches results for a standalone function."""
        call_count = 0

        @cached_analysis("_fn_cache")
        def compute(x, y):
            nonlocal call_count
            call_count += 1
            return x + y

        assert compute(1, 2) == 3
        assert compute(1, 2) == 3  # cached
        assert call_count == 1
        # Different args should miss cache
        assert compute(3, 4) == 7
        assert call_count == 2

    def test_cached_analysis_decorator_method(self):
        """Decorator caches results for a bound method, storing cache on instance."""

        class Analyzer:
            call_count = 0

            @cached_analysis("_analysis_cache")
            def analyze(self, value):
                self.call_count += 1
                return value * 10

        a = Analyzer()
        assert a.analyze(5) == 50
        assert a.analyze(5) == 50  # cached
        assert a.call_count == 1
        assert a.analyze(6) == 60
        assert a.call_count == 2
        # Cache is stored on the instance
        assert hasattr(a, "_analysis_cache")


class TestGetPerformanceCache:
    """Tests for the singleton accessor"""

    def test_returns_same_instance(self):
        import music21_mcp.performance_cache as mod

        mod._global_cache = None
        c1 = get_performance_cache()
        c2 = get_performance_cache()
        assert c1 is c2
        mod._global_cache = None
