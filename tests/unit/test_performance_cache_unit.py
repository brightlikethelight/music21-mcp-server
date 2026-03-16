"""Tests for performance_cache module"""

import pytest
from music21 import chord, key, stream

from music21_mcp.performance_cache import (
    PerformanceCache,
    cached_analysis,
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


class TestGetPerformanceCache:
    """Tests for the singleton accessor"""

    def test_returns_same_instance(self):
        import music21_mcp.performance_cache as mod

        mod._global_cache = None
        c1 = get_performance_cache()
        c2 = get_performance_cache()
        assert c1 is c2
        mod._global_cache = None
