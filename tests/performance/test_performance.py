#!/usr/bin/env python3
"""
Comprehensive Performance and Load Tests for Music21 MCP Server

Test scenarios:
1. Concurrent client handling (10, 50, 100 clients)
2. Large score processing (symphonies with 100+ instruments)
3. Memory usage under sustained load
4. Tool execution timing benchmarks
5. Rate limiting effectiveness
6. Resource cleanup verification
7. 24-hour stability test simulation
"""

import asyncio
import gc
import json
import logging
import os
import platform
import random
import statistics
import sys
import tempfile
import time
import tracemalloc
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import psutil
import pytest
import pytest_benchmark

# Try to import optional performance analysis tools
try:
    from memory_profiler import profile
    HAS_MEMORY_PROFILER = True
except ImportError:
    HAS_MEMORY_PROFILER = False
    # Create a no-op decorator
    def profile(func):
        return func

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from music21 import chord, converter, corpus, key, meter, note, stream, tempo
from music21_mcp.server import scores, rate_limiter
from music21_mcp.tools import (
    ImportScoreTool,
    ListScoresTool,
    KeyAnalysisTool,
    ChordAnalysisTool,
    ScoreInfoTool,
    ExportScoreTool,
    DeleteScoreTool,
    HarmonyAnalysisTool,
    VoiceLeadingAnalysisTool,
    PatternRecognitionTool,
    HarmonizationTool,
    CounterpointGeneratorTool,
    StyleImitationTool,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance test metrics"""
    
    test_name: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    # Timing metrics
    response_times: List[float] = field(default_factory=list)
    tool_timings: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    
    # Memory metrics
    memory_samples: List[float] = field(default_factory=list)
    peak_memory_mb: float = 0.0
    memory_leaked_mb: float = 0.0
    
    # Throughput metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Resource metrics
    cpu_usage_samples: List[float] = field(default_factory=list)
    open_files: List[int] = field(default_factory=list)
    active_threads: List[int] = field(default_factory=list)
    
    # Error tracking
    errors_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    def add_response_time(self, duration: float):
        """Add a response time measurement"""
        self.response_times.append(duration)
        self.total_requests += 1
    
    def add_tool_timing(self, tool_name: str, duration: float):
        """Add timing for a specific tool"""
        self.tool_timings[tool_name].append(duration)
    
    def record_error(self, error_type: str):
        """Record an error occurrence"""
        self.errors_by_type[error_type] += 1
        self.failed_requests += 1
    
    def record_success(self):
        """Record a successful request"""
        self.successful_requests += 1
    
    def sample_resources(self):
        """Sample current resource usage"""
        process = psutil.Process()
        
        # Memory
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_samples.append(memory_mb)
        self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
        
        # CPU
        cpu_percent = process.cpu_percent(interval=0.1)
        self.cpu_usage_samples.append(cpu_percent)
        
        # File descriptors
        try:
            self.open_files.append(len(process.open_files()))
        except:
            pass
        
        # Threads
        self.active_threads.append(process.num_threads())
    
    def calculate_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics"""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        summary = {
            "test_name": self.test_name,
            "duration_seconds": duration,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (self.successful_requests / self.total_requests * 100) 
                           if self.total_requests > 0 else 0,
            "throughput_rps": self.total_requests / duration if duration > 0 else 0,
        }
        
        # Response time statistics
        if self.response_times:
            summary.update({
                "response_time_min": min(self.response_times),
                "response_time_max": max(self.response_times),
                "response_time_mean": statistics.mean(self.response_times),
                "response_time_median": statistics.median(self.response_times),
                "response_time_p95": self._percentile(self.response_times, 95),
                "response_time_p99": self._percentile(self.response_times, 99),
            })
        
        # Memory statistics
        if self.memory_samples:
            summary.update({
                "memory_min_mb": min(self.memory_samples),
                "memory_max_mb": max(self.memory_samples),
                "memory_mean_mb": statistics.mean(self.memory_samples),
                "memory_leaked_mb": self.memory_leaked_mb,
            })
        
        # CPU statistics
        if self.cpu_usage_samples:
            summary.update({
                "cpu_mean_percent": statistics.mean(self.cpu_usage_samples),
                "cpu_max_percent": max(self.cpu_usage_samples),
            })
        
        # Tool-specific timings
        tool_summary = {}
        for tool, timings in self.tool_timings.items():
            if timings:
                tool_summary[tool] = {
                    "count": len(timings),
                    "mean": statistics.mean(timings),
                    "min": min(timings),
                    "max": max(timings),
                    "p95": self._percentile(timings, 95),
                }
        summary["tool_timings"] = tool_summary
        
        # Error summary
        if self.errors_by_type:
            summary["errors"] = dict(self.errors_by_type)
        
        return summary
    
    @staticmethod
    def _percentile(data: List[float], percentile: float) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]


@contextmanager
def measure_time(metrics: PerformanceMetrics, operation: str):
    """Context manager to measure operation time"""
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        metrics.add_tool_timing(operation, duration)
        metrics.add_response_time(duration)


class PerformanceTestBase:
    """Base class for performance tests"""
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test"""
        # Clear scores before test
        scores.clear()
        gc.collect()
        
        # Reset rate limiter
        rate_limiter.tokens = rate_limiter.burst
        
        yield
        
        # Cleanup after test
        scores.clear()
        gc.collect()
    
    def generate_large_score(self, num_parts: int = 100, num_measures: int = 300) -> stream.Score:
        """Generate a large orchestral score for testing"""
        score = stream.Score()
        
        # Add metadata
        score.metadata = stream.Metadata()
        score.metadata.title = f"Test Symphony - {num_parts} instruments"
        score.metadata.composer = "Performance Test"
        
        # Create parts
        instrument_families = ['strings', 'woodwinds', 'brass', 'percussion']
        
        for i in range(num_parts):
            part = stream.Part()
            part.id = f"Part-{i+1}"
            
            # Add tempo marking
            if i == 0:
                part.append(tempo.MetronomeMark(number=120))
            
            # Add time signature
            part.append(meter.TimeSignature('4/4'))
            
            # Generate measures
            for m in range(num_measures):
                measure = stream.Measure(number=m + 1)
                
                # Create different patterns for different instrument families
                family_idx = i % len(instrument_families)
                
                if family_idx == 0:  # Strings - flowing melodies
                    for beat in range(4):
                        n = note.Note(60 + (i % 12) + random.randint(-3, 3))
                        n.quarterLength = 0.5
                        measure.append(n)
                        n = note.Note(64 + (i % 12) + random.randint(-3, 3))
                        n.quarterLength = 0.5
                        measure.append(n)
                
                elif family_idx == 1:  # Woodwinds - sustained notes
                    n = note.Note(67 + (i % 7))
                    n.quarterLength = 4
                    measure.append(n)
                
                elif family_idx == 2:  # Brass - rhythmic patterns
                    for beat in range(4):
                        if beat % 2 == 0:
                            n = note.Note(48 + (i % 5))
                            n.quarterLength = 1
                            measure.append(n)
                        else:
                            r = note.Rest()
                            r.quarterLength = 1
                            measure.append(r)
                
                else:  # Percussion - complex rhythms
                    durations = [0.25, 0.25, 0.5, 0.25, 0.25, 0.5, 1, 0.5]
                    for dur in durations:
                        if random.random() > 0.3:
                            n = note.Note(60)
                            n.quarterLength = dur
                            measure.append(n)
                        else:
                            r = note.Rest()
                            r.quarterLength = dur
                            measure.append(r)
                
                part.append(measure)
            
            score.append(part)
        
        return score
    
    async def simulate_client_request(self, client_id: int, metrics: PerformanceMetrics):
        """Simulate a single client making requests"""
        # Create a unique score ID for this client
        score_id = f"client_{client_id}_score"
        
        try:
            # Import a score
            with measure_time(metrics, "import_score"):
                tool = ImportScoreTool(scores)
                result = await tool.execute(
                    score_id=score_id,
                    source="bach/bwv65.2.xml",
                    source_type="corpus"
                )
                if "error" not in result:
                    metrics.record_success()
                else:
                    metrics.record_error("import_error")
            
            # Perform various analyses
            if score_id in scores:
                # Key analysis
                with measure_time(metrics, "key_analysis"):
                    tool = KeyAnalysisTool(scores)
                    result = await tool.execute(score_id=score_id)
                    if "error" not in result:
                        metrics.record_success()
                    else:
                        metrics.record_error("key_analysis_error")
                
                # Chord analysis
                with measure_time(metrics, "chord_analysis"):
                    tool = ChordAnalysisTool(scores)
                    result = await tool.execute(score_id=score_id, roman_numerals=True)
                    if "error" not in result:
                        metrics.record_success()
                    else:
                        metrics.record_error("chord_analysis_error")
                
                # Export score
                with measure_time(metrics, "export_score"):
                    tool = ExportScoreTool(scores)
                    result = await tool.execute(score_id=score_id, format="musicxml")
                    if "error" not in result:
                        metrics.record_success()
                    else:
                        metrics.record_error("export_error")
                
                # Cleanup
                with measure_time(metrics, "delete_score"):
                    tool = DeleteScoreTool(scores)
                    result = await tool.execute(score_id=score_id)
                    if "error" not in result:
                        metrics.record_success()
                    else:
                        metrics.record_error("delete_error")
        
        except Exception as e:
            logger.error(f"Client {client_id} error: {e}")
            metrics.record_error(type(e).__name__)


class TestConcurrentClients(PerformanceTestBase):
    """Test concurrent client handling"""
    
    @pytest.mark.parametrize("num_clients", [10, 50, 100])
    @pytest.mark.asyncio
    async def test_concurrent_clients(self, num_clients: int):
        """Test server handling multiple concurrent clients"""
        metrics = PerformanceMetrics(f"concurrent_clients_{num_clients}")
        
        # Start resource monitoring
        monitor_task = asyncio.create_task(self._monitor_resources(metrics))
        
        try:
            # Create client tasks
            tasks = []
            for i in range(num_clients):
                task = asyncio.create_task(
                    self.simulate_client_request(i, metrics)
                )
                tasks.append(task)
                # Stagger client starts slightly
                await asyncio.sleep(0.01)
            
            # Wait for all clients to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            
        finally:
            # Stop monitoring
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
        
        # Calculate and log results
        summary = metrics.calculate_summary()
        logger.info(f"Test {metrics.test_name} completed:")
        logger.info(f"  Total requests: {summary['total_requests']}")
        logger.info(f"  Success rate: {summary['success_rate']:.2f}%")
        logger.info(f"  Throughput: {summary['throughput_rps']:.2f} req/s")
        logger.info(f"  Response time p95: {summary.get('response_time_p95', 0):.3f}s")
        logger.info(f"  Peak memory: {summary['memory_max_mb']:.1f} MB")
        
        # Assertions
        assert summary['success_rate'] >= 90.0, f"Success rate too low: {summary['success_rate']:.2f}%"
        assert summary['memory_max_mb'] < 2048, f"Memory usage too high: {summary['memory_max_mb']:.1f} MB"
    
    async def _monitor_resources(self, metrics: PerformanceMetrics):
        """Monitor resources during test"""
        while True:
            metrics.sample_resources()
            await asyncio.sleep(0.5)


class TestLargeScoreProcessing(PerformanceTestBase):
    """Test processing of large orchestral scores"""
    
    @pytest.mark.parametrize("num_instruments,num_measures", [
        (20, 100),    # Chamber orchestra
        (50, 200),    # Full orchestra
        (100, 300),   # Large orchestra
        (150, 500),   # Extreme case
    ])
    @pytest.mark.asyncio
    async def test_large_score_processing(self, num_instruments: int, num_measures: int):
        """Test processing large orchestral scores"""
        metrics = PerformanceMetrics(f"large_score_{num_instruments}x{num_measures}")
        
        # Generate large score
        logger.info(f"Generating score with {num_instruments} instruments, {num_measures} measures")
        with measure_time(metrics, "score_generation"):
            large_score = self.generate_large_score(num_instruments, num_measures)
        
        # Store the score
        score_id = "large_symphony"
        scores[score_id] = large_score
        
        try:
            # Test various operations on large score
            
            # 1. Score info (should handle size well)
            with measure_time(metrics, "score_info"):
                tool = ScoreInfoTool(scores)
                result = await tool.execute(score_id=score_id)
                if "error" not in result:
                    metrics.record_success()
                    assert result['parts'] == num_instruments
                else:
                    metrics.record_error("score_info_error")
            
            # 2. Key analysis (computationally intensive)
            with measure_time(metrics, "key_analysis_large"):
                tool = KeyAnalysisTool(scores)
                result = await tool.execute(score_id=score_id)
                if "error" not in result:
                    metrics.record_success()
                else:
                    metrics.record_error("key_analysis_error")
            
            # 3. Harmony analysis (very intensive for large scores)
            if num_instruments <= 50:  # Skip for extremely large scores
                with measure_time(metrics, "harmony_analysis_large"):
                    tool = HarmonyAnalysisTool(scores)
                    result = await tool.execute(score_id=score_id, measures="1-10")
                    if "error" not in result:
                        metrics.record_success()
                    else:
                        metrics.record_error("harmony_analysis_error")
            
            # 4. Export to different formats
            for format_type in ["musicxml", "midi"]:
                with measure_time(metrics, f"export_{format_type}"):
                    tool = ExportScoreTool(scores)
                    result = await tool.execute(score_id=score_id, format=format_type)
                    if "error" not in result:
                        metrics.record_success()
                        # Check file size is reasonable
                        if 'file_size' in result:
                            size_mb = result['file_size'] / 1024 / 1024
                            logger.info(f"Exported {format_type} size: {size_mb:.2f} MB")
                    else:
                        metrics.record_error(f"export_{format_type}_error")
        
        finally:
            # Cleanup
            del scores[score_id]
            gc.collect()
        
        # Calculate results
        summary = metrics.calculate_summary()
        logger.info(f"Large score test completed:")
        logger.info(f"  Score size: {num_instruments}x{num_measures}")
        logger.info(f"  Total operations: {summary['total_requests']}")
        logger.info(f"  Success rate: {summary['success_rate']:.2f}%")
        logger.info(f"  Peak memory: {summary['memory_max_mb']:.1f} MB")
        
        # Log tool-specific timings
        for tool_name, tool_stats in summary['tool_timings'].items():
            logger.info(f"  {tool_name}: {tool_stats['mean']:.3f}s (p95: {tool_stats['p95']:.3f}s)")
        
        # Assertions
        assert summary['success_rate'] >= 80.0, f"Success rate too low for large scores"
        # Memory should scale reasonably with score size
        expected_memory_mb = 500 + (num_instruments * num_measures / 100)
        assert summary['memory_max_mb'] < expected_memory_mb, f"Memory usage excessive"


class TestMemoryUnderSustainedLoad(PerformanceTestBase):
    """Test memory behavior under sustained load"""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_memory_leak_detection(self):
        """Test for memory leaks under sustained load"""
        metrics = PerformanceMetrics("memory_leak_detection")
        
        # Enable tracemalloc for detailed memory tracking
        tracemalloc.start()
        
        # Initial memory snapshot
        gc.collect()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        initial_snapshot = tracemalloc.take_snapshot()
        
        # Run sustained load for 5 minutes
        duration_seconds = 300
        operations_per_second = 10
        
        start_time = time.time()
        operation_count = 0
        
        while time.time() - start_time < duration_seconds:
            # Perform operations
            score_id = f"memory_test_{operation_count}"
            
            try:
                # Import
                tool = ImportScoreTool(scores)
                await tool.execute(
                    score_id=score_id,
                    source="bach/bwv7.7.xml",
                    source_type="corpus"
                )
                
                # Analyze
                tool = KeyAnalysisTool(scores)
                await tool.execute(score_id=score_id)
                
                # Export
                tool = ExportScoreTool(scores)
                await tool.execute(score_id=score_id, format="midi")
                
                # Delete
                tool = DeleteScoreTool(scores)
                await tool.execute(score_id=score_id)
                
                operation_count += 1
                metrics.record_success()
                
            except Exception as e:
                metrics.record_error(type(e).__name__)
            
            # Sample memory periodically
            if operation_count % 10 == 0:
                metrics.sample_resources()
                gc.collect()
            
            # Control rate
            await asyncio.sleep(1.0 / operations_per_second)
        
        # Final memory analysis
        gc.collect()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        final_snapshot = tracemalloc.take_snapshot()
        
        # Calculate memory growth
        memory_growth = final_memory - initial_memory
        metrics.memory_leaked_mb = memory_growth
        
        # Analyze top memory differences
        top_stats = final_snapshot.compare_to(initial_snapshot, 'lineno')
        logger.info("Top 10 memory allocations:")
        for stat in top_stats[:10]:
            logger.info(f"  {stat}")
        
        tracemalloc.stop()
        
        # Results
        summary = metrics.calculate_summary()
        logger.info(f"Memory leak test completed:")
        logger.info(f"  Duration: {duration_seconds}s")
        logger.info(f"  Operations: {operation_count}")
        logger.info(f"  Initial memory: {initial_memory:.1f} MB")
        logger.info(f"  Final memory: {final_memory:.1f} MB")
        logger.info(f"  Memory growth: {memory_growth:.1f} MB")
        logger.info(f"  Growth rate: {memory_growth / (duration_seconds / 60):.2f} MB/min")
        
        # Assert no significant memory leak
        # Allow up to 100MB growth over 5 minutes
        assert memory_growth < 100, f"Excessive memory growth: {memory_growth:.1f} MB"


class TestToolExecutionBenchmarks(PerformanceTestBase):
    """Benchmark individual tool performance"""
    
    @pytest.fixture
    def sample_scores(self):
        """Prepare sample scores for benchmarking"""
        # Import various test scores
        test_scores = {
            "small": "bach/bwv7.7.xml",
            "medium": "beethoven/opus18no1/movement1.xml",
            "large": "verdi/lTraviata/04_iv_coro_di_zingarelle.xml"
        }
        
        for score_id, source in test_scores.items():
            try:
                tool = ImportScoreTool(scores)
                asyncio.run(tool.execute(
                    score_id=score_id,
                    source=source,
                    source_type="corpus"
                ))
            except:
                # If specific score not available, generate one
                if score_id == "small":
                    scores[score_id] = self.generate_large_score(1, 10)
                elif score_id == "medium":
                    scores[score_id] = self.generate_large_score(4, 50)
                else:
                    scores[score_id] = self.generate_large_score(20, 100)
        
        yield
        
        # Cleanup
        scores.clear()
    
    def test_import_tool_benchmark(self, benchmark):
        """Benchmark import tool performance"""
        tool = ImportScoreTool(scores)
        
        async def import_score():
            return await tool.execute(
                score_id=f"bench_{time.time()}",
                source="bach/bwv7.7.xml",
                source_type="corpus"
            )
        
        result = benchmark(lambda: asyncio.run(import_score()))
        assert "error" not in result
    
    def test_key_analysis_benchmark(self, benchmark, sample_scores):
        """Benchmark key analysis performance"""
        tool = KeyAnalysisTool(scores)
        
        async def analyze_key():
            return await tool.execute(score_id="medium")
        
        result = benchmark(lambda: asyncio.run(analyze_key()))
        assert "error" not in result
    
    def test_harmony_analysis_benchmark(self, benchmark, sample_scores):
        """Benchmark harmony analysis performance"""
        tool = HarmonyAnalysisTool(scores)
        
        async def analyze_harmony():
            return await tool.execute(score_id="small", measures="1-4")
        
        result = benchmark(lambda: asyncio.run(analyze_harmony()))
        assert "error" not in result
    
    def test_export_tool_benchmark(self, benchmark, sample_scores):
        """Benchmark export tool performance"""
        tool = ExportScoreTool(scores)
        
        async def export_score():
            return await tool.execute(score_id="medium", format="musicxml")
        
        result = benchmark(lambda: asyncio.run(export_score()))
        assert "error" not in result


class TestRateLimiting(PerformanceTestBase):
    """Test rate limiting effectiveness"""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_enforcement(self):
        """Test that rate limiter properly enforces limits"""
        metrics = PerformanceMetrics("rate_limiter_test")
        
        # Reset rate limiter to known state
        rate_limiter.tokens = rate_limiter.burst
        rate_limiter.last_update = time.time()
        
        # Try to exceed rate limit
        successful_requests = 0
        rate_limited_requests = 0
        
        # Burst phase - should allow 'burst' requests immediately
        for i in range(rate_limiter.burst + 5):
            tool = ListScoresTool(scores)
            result = await tool.execute()
            
            if "error" in result and "rate limit" in result["error"].lower():
                rate_limited_requests += 1
            else:
                successful_requests += 1
            
            metrics.total_requests += 1
        
        # Should have allowed burst requests and blocked the rest
        assert successful_requests == rate_limiter.burst
        assert rate_limited_requests == 5
        
        # Wait for tokens to replenish
        await asyncio.sleep(2)
        
        # Test sustained rate
        start_time = time.time()
        test_duration = 5  # seconds
        
        while time.time() - start_time < test_duration:
            tool = ListScoresTool(scores)
            result = await tool.execute()
            
            if "error" in result and "rate limit" in result["error"].lower():
                rate_limited_requests += 1
            else:
                successful_requests += 1
            
            metrics.total_requests += 1
            await asyncio.sleep(0.01)  # Small delay to test sustained rate
        
        # Calculate actual rate
        actual_duration = time.time() - start_time
        sustained_successful = successful_requests - rate_limiter.burst
        actual_rate = sustained_successful / actual_duration
        
        logger.info(f"Rate limiting test results:")
        logger.info(f"  Burst allowed: {rate_limiter.burst}")
        logger.info(f"  Target rate: {rate_limiter.rate:.2f} req/s")
        logger.info(f"  Actual rate: {actual_rate:.2f} req/s")
        logger.info(f"  Total requests: {metrics.total_requests}")
        logger.info(f"  Rate limited: {rate_limited_requests}")
        
        # Rate should be close to configured rate (within 20%)
        rate_ratio = actual_rate / rate_limiter.rate
        assert 0.8 <= rate_ratio <= 1.2, f"Rate limiting not accurate: {actual_rate:.2f} vs {rate_limiter.rate:.2f}"


class TestResourceCleanup(PerformanceTestBase):
    """Test resource cleanup and recovery"""
    
    @pytest.mark.asyncio
    async def test_resource_cleanup_after_errors(self):
        """Test that resources are properly cleaned up after errors"""
        metrics = PerformanceMetrics("resource_cleanup_test")
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        initial_fds = len(psutil.Process().open_files())
        
        # Create situations that might leak resources
        for i in range(50):
            score_id = f"cleanup_test_{i}"
            
            try:
                # Import score
                tool = ImportScoreTool(scores)
                await tool.execute(
                    score_id=score_id,
                    source="bach/bwv7.7.xml",
                    source_type="corpus"
                )
                
                # Try operations that might fail
                if i % 3 == 0:
                    # Invalid operation
                    tool = HarmonyAnalysisTool(scores)
                    await tool.execute(score_id=score_id, measures="invalid")
                
                if i % 5 == 0:
                    # Export to temporary file
                    tool = ExportScoreTool(scores)
                    await tool.execute(score_id=score_id, format="musicxml")
                
            except Exception as e:
                metrics.record_error(type(e).__name__)
            
            # Always try to clean up
            try:
                tool = DeleteScoreTool(scores)
                await tool.execute(score_id=score_id)
            except:
                pass
        
        # Force cleanup
        scores.clear()
        gc.collect()
        await asyncio.sleep(1)
        
        # Check resources
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        final_fds = len(psutil.Process().open_files())
        
        memory_growth = final_memory - initial_memory
        fd_growth = final_fds - initial_fds
        
        logger.info(f"Resource cleanup test results:")
        logger.info(f"  Memory growth: {memory_growth:.1f} MB")
        logger.info(f"  File descriptor growth: {fd_growth}")
        logger.info(f"  Remaining scores: {len(scores)}")
        
        # Assert minimal resource leakage
        assert memory_growth < 50, f"Excessive memory growth: {memory_growth:.1f} MB"
        assert fd_growth <= 5, f"File descriptor leak: {fd_growth} new FDs"
        assert len(scores) == 0, f"Scores not cleaned up: {len(scores)} remaining"


class TestStabilitySimulation(PerformanceTestBase):
    """Simulate 24-hour stability test in accelerated time"""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_stability_simulation(self):
        """Simulate 24-hour operation in 10 minutes"""
        metrics = PerformanceMetrics("stability_simulation")
        
        # Simulation parameters
        real_duration = 600  # 10 minutes real time
        simulated_hours = 24
        time_acceleration = (simulated_hours * 3600) / real_duration
        
        logger.info(f"Starting stability simulation:")
        logger.info(f"  Real duration: {real_duration}s")
        logger.info(f"  Simulated time: {simulated_hours} hours")
        logger.info(f"  Time acceleration: {time_acceleration:.1f}x")
        
        start_time = time.time()
        
        # Simulate different load patterns throughout the day
        async def simulate_hour(hour: int):
            """Simulate one hour of operation"""
            # Vary load based on time of day
            if 9 <= hour <= 17:  # Business hours
                load_factor = 1.0
            elif 18 <= hour <= 22:  # Evening
                load_factor = 0.7
            else:  # Night
                load_factor = 0.3
            
            requests_per_hour = int(1000 * load_factor)
            
            # Simulate requests for this hour
            tasks = []
            for _ in range(int(requests_per_hour / time_acceleration)):
                task = asyncio.create_task(self._simulate_user_session(metrics))
                tasks.append(task)
                await asyncio.sleep(0.01 / time_acceleration)
            
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Run simulation
        for hour in range(simulated_hours):
            if time.time() - start_time > real_duration:
                break
            
            logger.info(f"Simulating hour {hour}...")
            await simulate_hour(hour)
            
            # Periodic maintenance
            if hour % 6 == 0:
                gc.collect()
                metrics.sample_resources()
        
        # Calculate results
        summary = metrics.calculate_summary()
        
        logger.info(f"Stability simulation completed:")
        logger.info(f"  Simulated time: {simulated_hours} hours")
        logger.info(f"  Total requests: {summary['total_requests']}")
        logger.info(f"  Success rate: {summary['success_rate']:.2f}%")
        logger.info(f"  Average response time: {summary.get('response_time_mean', 0):.3f}s")
        logger.info(f"  Peak memory: {summary['memory_max_mb']:.1f} MB")
        logger.info(f"  Memory stability: {summary['memory_leaked_mb']:.1f} MB growth")
        
        # Stability criteria
        assert summary['success_rate'] >= 99.0, "Success rate below 99%"
        assert summary['memory_leaked_mb'] < 200, "Excessive memory growth"
        assert summary.get('response_time_p99', 0) < 5.0, "Response times too slow"
    
    async def _simulate_user_session(self, metrics: PerformanceMetrics):
        """Simulate a typical user session"""
        score_id = f"session_{time.time()}_{random.randint(1000, 9999)}"
        
        try:
            # User workflow
            tool = ImportScoreTool(scores)
            result = await tool.execute(
                score_id=score_id,
                source=random.choice([
                    "bach/bwv7.7.xml",
                    "beethoven/opus59no3/movement1.xml",
                    "mozart/k458/movement1.xml"
                ]),
                source_type="corpus"
            )
            
            if "error" not in result:
                metrics.record_success()
                
                # Perform some analysis
                if random.random() > 0.5:
                    tool = KeyAnalysisTool(scores)
                    await tool.execute(score_id=score_id)
                
                # Cleanup
                tool = DeleteScoreTool(scores)
                await tool.execute(score_id=score_id)
        
        except Exception as e:
            metrics.record_error(type(e).__name__)


# Performance test runner
if __name__ == "__main__":
    """Run performance tests with detailed reporting"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Music21 MCP Server Performance Tests")
    parser.add_argument("--test", help="Specific test to run")
    parser.add_argument("--clients", type=int, default=50, help="Number of concurrent clients")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    # Run tests
    if args.test:
        pytest.main([__file__, f"-k {args.test}", "-v"])
    else:
        pytest.main([__file__, "-v", "--tb=short"])