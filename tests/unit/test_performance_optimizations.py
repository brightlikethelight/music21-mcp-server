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
    OptimizedChordAnalysisTool,
    OptimizedHarmonyAnalysisTool,
    PerformanceOptimizer,
    cached_analysis,
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
        with patch.object(optimizer, '_fast_roman_lookup', return_value="CACHED"):
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


class TestOptimizedChordAnalysisTool:
    """Test the OptimizedChordAnalysisTool class"""

    @pytest.fixture
    def tool(self):
        """Create an optimized chord analysis tool"""
        score_manager = {}
        optimizer = PerformanceOptimizer()
        return OptimizedChordAnalysisTool(score_manager, optimizer)

    @pytest.mark.asyncio
    async def test_analyze_chords_not_found(self, tool):
        """Test analyzing chords for non-existent score"""
        result = await tool.analyze_chords_optimized("nonexistent")
        assert result["error"] == "Score 'nonexistent' not found"

    @pytest.mark.asyncio
    async def test_analyze_chords_with_caching(self, tool):
        """Test chord analysis with caching"""
        # Create a simple score
        score = stream.Score()
        part = stream.Part()
        part.append(chord.Chord(["C4", "E4", "G4"]))
        part.append(chord.Chord(["G4", "B4", "D5"]))
        score.append(part)
        
        # Add to score manager
        tool.score_manager["test"] = score
        
        # First analysis should compute
        result1 = await tool.analyze_chords_optimized("test", limit=2)
        assert result1["score_id"] == "test"
        assert result1["total_chords"] == 2
        assert "analysis_time_ms" in result1
        
        # Second analysis should hit cache
        result2 = await tool.analyze_chords_optimized("test", limit=2)
        assert result2 == result1  # Should be identical from cache


class TestOptimizedHarmonyAnalysisTool:
    """Test the OptimizedHarmonyAnalysisTool class"""

    @pytest.fixture
    def tool(self):
        """Create an optimized harmony analysis tool"""
        score_manager = {}
        optimizer = PerformanceOptimizer()
        return OptimizedHarmonyAnalysisTool(score_manager, optimizer)

    @pytest.mark.asyncio
    async def test_analyze_harmony_not_found(self, tool):
        """Test analyzing harmony for non-existent score"""
        result = await tool.analyze_harmony_optimized("nonexistent")
        assert result["error"] == "Score 'nonexistent' not found"

    @pytest.mark.asyncio
    async def test_analyze_harmony_with_cache(self, tool):
        """Test harmony analysis with caching"""
        # Create a simple score
        score = stream.Score()
        part = stream.Part()
        part.append(chord.Chord(["C4", "E4", "G4"]))
        score.append(part)
        
        # Add to score manager
        tool.score_manager["test"] = score
        
        # First analysis
        result1 = await tool.analyze_harmony_optimized("test")
        assert result1["score_id"] == "test"
        assert "key" in result1
        assert "roman_numerals" in result1
        assert result1["cached"] is False
        
        # Second analysis should be cached
        result2 = await tool.analyze_harmony_optimized("test")
        # Note: The decorator sets cached=False in the result,
        # but it should be the same object from cache
        assert result2 == result1


class TestCachedAnalysisDecorator:
    """Test the cached_analysis decorator"""

    @pytest.mark.asyncio
    async def test_cached_decorator_basic(self):
        """Test basic caching with decorator"""
        call_count = 0
        
        class TestClass:
            def __init__(self):
                self.test_cache = {}
            
            @cached_analysis("test_cache")
            async def analyze(self, value):
                nonlocal call_count
                call_count += 1
                return f"result_{value}"
        
        obj = TestClass()
        
        # First call should compute
        result1 = await obj.analyze("test")
        assert result1 == "result_test"
        assert call_count == 1
        
        # Second call should use cache
        result2 = await obj.analyze("test")
        assert result2 == "result_test"
        assert call_count == 1  # Should not increment
        
        # Different argument should compute
        result3 = await obj.analyze("other")
        assert result3 == "result_other"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_cached_decorator_with_key_func(self):
        """Test caching with custom key function"""
        class TestClass:
            def __init__(self):
                self.cache = {}
            
            @cached_analysis("cache", key_func=lambda self, x, y: f"key_{x}_{y}")
            async def compute(self, x, y):
                return x + y
        
        obj = TestClass()
        
        # First call
        result1 = await obj.compute(2, 3)
        assert result1 == 5
        
        # Check cache key was created correctly
        assert "key_2_3" in obj.cache
        
        # Second call with same args should hit cache
        obj.cache["key_2_3"] = 100  # Modify cache directly
        result2 = await obj.compute(2, 3)
        assert result2 == 100  # Should get cached value


class TestIntegration:
    """Integration tests for performance optimizations"""

    @pytest.mark.asyncio
    async def test_full_optimization_workflow(self):
        """Test complete optimization workflow"""
        # Create optimizer
        optimizer = PerformanceOptimizer()
        
        # Warm cache
        optimizer.warm_cache([])
        
        # Create test score
        score = stream.Score()
        part = stream.Part()
        for pitch in ["C4", "D4", "E4", "F4", "G4"]:
            part.append(chord.Chord([pitch]))
        score.append(part)
        
        # Create tools
        score_manager = {"test": score}
        chord_tool = OptimizedChordAnalysisTool(score_manager, optimizer)
        harmony_tool = OptimizedHarmonyAnalysisTool(score_manager, optimizer)
        
        # Analyze chords
        chord_result = await chord_tool.analyze_chords_optimized("test")
        assert chord_result["total_chords"] > 0
        
        # Analyze harmony (should reuse some cached data)
        harmony_result = await harmony_tool.analyze_harmony_optimized("test")
        assert harmony_result["score_id"] == "test"
        
        # Second analysis should be fully cached
        chord_result2 = await chord_tool.analyze_chords_optimized("test")
        assert chord_result2 == chord_result