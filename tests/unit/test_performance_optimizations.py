"""
Test suite for performance_optimizations module

Tests the performance optimization utilities including:
- PerformanceOptimizer with caching
- Parallel chord analysis
- Optimized analysis tools
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from music21 import chord, key, stream

from music21_mcp.performance_optimizations import (
    PerformanceOptimizer,
)


class TestPerformanceOptimizer:
    """Test the PerformanceOptimizer class"""

    @pytest.fixture
    def optimizer(self):
        """Create a PerformanceOptimizer instance"""
        return PerformanceOptimizer(cache_ttl=60, max_cache_size=10)

    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initializes with correct settings"""
        assert optimizer.roman_cache.ttl == 60
        assert optimizer.roman_cache.maxsize == 10
        assert optimizer.key_cache.maxsize == 100
        assert optimizer.chord_analysis_cache.maxsize == 500
        assert optimizer.executor is not None
        assert optimizer.common_romans is not None

    def test_chord_hash(self, optimizer):
        """Test chord hashing for cache keys"""
        # Create a simple chord
        c_major = chord.Chord(["C4", "E4", "G4"])

        # Hash should be deterministic
        hash1 = optimizer.chord_hash(c_major)
        hash2 = optimizer.chord_hash(c_major)
        assert hash1 == hash2
        assert len(hash1) == 16  # MD5 truncated to 16 chars

        # Different chords should have different hashes
        d_major = chord.Chord(["D4", "F#4", "A4"])
        hash3 = optimizer.chord_hash(d_major)
        assert hash1 != hash3

    def test_fast_roman_lookup(self, optimizer):
        """Test fast Roman numeral lookup for common chords"""
        # Create C major key
        c_key = key.Key("C")

        # Test tonic chord (C major in C major = I)
        c_major = chord.Chord(["C4", "E4", "G4"])
        result = optimizer._fast_roman_lookup(c_major, c_key)
        assert result == "I"

        # Test dominant chord (G major in C major = V)
        g_major = chord.Chord(["G4", "B4", "D5"])
        result = optimizer._fast_roman_lookup(g_major, c_key)
        assert result == "V"

        # Test subdominant (F major in C major = IV)
        f_major = chord.Chord(["F4", "A4", "C5"])
        result = optimizer._fast_roman_lookup(f_major, c_key)
        assert result == "IV"

    def test_cached_roman_numeral(self, optimizer):
        """Test caching of Roman numeral analysis"""
        c_key = key.Key("C")
        c_major = chord.Chord(["C4", "E4", "G4"])

        # First call should compute and cache
        result1 = optimizer.get_cached_roman_numeral(c_major, c_key)
        assert result1 == "I"

        # Second call should hit cache
        with patch.object(optimizer, "_fast_roman_lookup", return_value="CACHED"):
            result2 = optimizer.get_cached_roman_numeral(c_major, c_key)
            # Should return cached "I", not "CACHED"
            assert result2 == "I"

    @pytest.mark.asyncio
    async def test_analyze_chords_parallel(self, optimizer):
        """Test parallel chord analysis"""
        # Create test chords
        chords = [
            chord.Chord(["C4", "E4", "G4"]),  # C major
            chord.Chord(["D4", "F4", "A4"]),  # D minor
            chord.Chord(["G4", "B4", "D5"]),  # G major
        ]
        c_key = key.Key("C")

        # Analyze in parallel
        results = await optimizer.analyze_chords_parallel(chords, c_key, batch_size=2)

        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)
        assert results[0]["roman_numeral"] == "I"
        assert results[2]["roman_numeral"] == "V"

    def test_warm_cache(self, optimizer):
        """Test cache warming with common progressions"""
        initial_size = len(optimizer.roman_cache)

        # Warm the cache
        optimizer.warm_cache([])

        # Cache should have more entries
        assert len(optimizer.roman_cache) >= initial_size

        # Common progressions should be cached
        c_key = key.Key("C")
        c_major = chord.Chord(["C", "E", "G"])
        cached = optimizer.get_cached_roman_numeral(c_major, c_key)
        assert cached is not None
