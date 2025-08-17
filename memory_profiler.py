#!/usr/bin/env python3
"""
Memory Profiler for Music21 MCP Server

Identifies memory leaks, excessive memory usage, and optimization opportunities.
Tracks memory usage patterns across different operations and identifies the
memory-heavy components that users will notice.
"""

import asyncio
import gc
import logging
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List

import psutil

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from music21_mcp.services import MusicAnalysisService
from music21_mcp.resource_manager import ResourceManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MemoryProfiler:
    """Memory profiling for music21 operations"""
    
    def __init__(self):
        self.service = None
        self.resource_manager = None
        self.baseline_memory = None
        self.snapshots = {}
        self.process = psutil.Process()
        
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        memory_info = self.process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "percent": self.process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024
        }
    
    def take_snapshot(self, name: str) -> None:
        """Take a memory snapshot"""
        if tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()
            self.snapshots[name] = {
                "tracemalloc": snapshot,
                "system": self.get_memory_stats(),
                "timestamp": time.time()
            }
            
    def analyze_snapshot_diff(self, start_name: str, end_name: str) -> Dict[str, Any]:
        """Analyze memory difference between two snapshots"""
        if start_name not in self.snapshots or end_name not in self.snapshots:
            return {"error": "Snapshots not found"}
            
        start_snap = self.snapshots[start_name]
        end_snap = self.snapshots[end_name]
        
        # System memory diff
        start_mem = start_snap["system"]
        end_mem = end_snap["system"]
        
        memory_diff = {
            "rss_diff_mb": end_mem["rss_mb"] - start_mem["rss_mb"],
            "vms_diff_mb": end_mem["vms_mb"] - start_mem["vms_mb"],
            "percent_diff": end_mem["percent"] - start_mem["percent"],
            "duration_seconds": end_snap["timestamp"] - start_snap["timestamp"]
        }
        
        # Tracemalloc diff for detailed analysis
        if "tracemalloc" in start_snap and "tracemalloc" in end_snap:
            top_stats = end_snap["tracemalloc"].compare_to(
                start_snap["tracemalloc"], 'lineno'
            )
            
            # Get top memory consuming lines
            top_lines = []
            for stat in top_stats[:10]:  # Top 10
                top_lines.append({
                    "size_mb": stat.size / 1024 / 1024,
                    "size_diff_mb": stat.size_diff / 1024 / 1024,
                    "count": stat.count,
                    "count_diff": stat.count_diff,
                    "traceback": str(stat.traceback)
                })
                
            memory_diff["detailed_analysis"] = top_lines
            
        return memory_diff
    
    async def setup(self):
        """Setup test environment with memory tracking"""
        print("🔧 Setting up memory profiling environment...")
        
        # Start memory tracing
        tracemalloc.start()
        
        # Get baseline memory
        self.baseline_memory = self.get_memory_stats()
        self.take_snapshot("baseline")
        
        # Initialize service with resource management
        self.service = MusicAnalysisService(max_memory_mb=256, max_scores=50)
        self.resource_manager = self.service.resource_manager
        
        self.take_snapshot("after_init")
        
        # Import multiple test scores to simulate real usage
        test_scores = [
            ("bach1", "bach/bwv66.6", "corpus"),
            ("bach2", "bach/bwv7.7", "corpus"),
            ("mozart1", "mozart/k545", "corpus"),
        ]
        
        for score_id, source, source_type in test_scores:
            try:
                await self.service.import_score(score_id, source, source_type)
            except Exception as e:
                logger.warning(f"Could not import {score_id}: {e}")
                
        self.take_snapshot("after_imports")
        print("✅ Test environment setup complete")
    
    async def profile_memory_usage(self):
        """Profile memory usage across different operations"""
        print("\n🧠 Memory Usage Profiling")
        print("=" * 50)
        
        # Test 1: Multiple chord analyses (common operation)
        print("\n📊 Test 1: Multiple Chord Analyses")
        self.take_snapshot("before_chord_analyses")
        
        for i in range(5):
            try:
                await self.service.analyze_chords("bach1")
                if i == 0:
                    self.take_snapshot("after_first_chord_analysis")
            except Exception as e:
                logger.warning(f"Chord analysis {i} failed: {e}")
                
        self.take_snapshot("after_multiple_chord_analyses")
        
        # Analyze memory growth
        diff = self.analyze_snapshot_diff("before_chord_analyses", "after_multiple_chord_analyses")
        print(f"   Memory growth: {diff['rss_diff_mb']:.2f}MB RSS, {diff['vms_diff_mb']:.2f}MB VMS")
        print(f"   Duration: {diff['duration_seconds']:.2f}s")
        
        # Test 2: Memory usage with caching vs without
        print("\n📊 Test 2: Cache Effectiveness")
        
        # Clear caches first
        gc.collect()
        self.take_snapshot("before_cache_test")
        
        # First analysis (cache miss)
        await self.service.analyze_harmony("bach1", "roman")
        self.take_snapshot("after_first_harmony")
        
        # Second analysis (cache hit)
        await self.service.analyze_harmony("bach1", "roman")
        self.take_snapshot("after_second_harmony")
        
        cache_miss_diff = self.analyze_snapshot_diff("before_cache_test", "after_first_harmony")
        cache_hit_diff = self.analyze_snapshot_diff("after_first_harmony", "after_second_harmony")
        
        print(f"   Cache miss memory: {cache_miss_diff['rss_diff_mb']:.2f}MB")
        print(f"   Cache hit memory: {cache_hit_diff['rss_diff_mb']:.2f}MB")
        print(f"   Cache efficiency: {((cache_miss_diff['rss_diff_mb'] - cache_hit_diff['rss_diff_mb']) / cache_miss_diff['rss_diff_mb'] * 100):.1f}% memory saved")
        
        # Test 3: Resource manager effectiveness
        print("\n📊 Test 3: Resource Manager Impact")
        self.take_snapshot("before_resource_test")
        
        # Import many scores to test limits
        for i in range(10):
            try:
                await self.service.import_score(f"test_score_{i}", "bach/bwv66.6", "corpus")
            except Exception as e:
                print(f"   Resource limit reached at score {i}: {e}")
                break
                
        self.take_snapshot("after_resource_test")
        
        # Check resource manager stats
        stats = self.resource_manager.get_system_stats()
        print(f"   Scores in memory: {stats['storage']['total_scores']}")
        print(f"   Memory utilization: {stats['storage']['memory_utilization_percent']:.1f}%")
        print(f"   System memory: {stats['system']['process_memory_mb']:.1f}MB")
        
        # Test 4: Memory leak detection
        print("\n📊 Test 4: Memory Leak Detection")
        await self.detect_memory_leaks()
        
    async def detect_memory_leaks(self):
        """Detect potential memory leaks"""
        print("   Running memory leak detection...")
        
        # Baseline
        gc.collect()
        self.take_snapshot("leak_baseline")
        baseline_mem = self.get_memory_stats()
        
        # Run operations multiple times
        for cycle in range(3):
            for operation in ["analyze_chords", "analyze_harmony", "key_analysis"]:
                try:
                    if operation == "analyze_chords":
                        await self.service.analyze_chords("bach1")
                    elif operation == "analyze_harmony":
                        await self.service.analyze_harmony("bach1", "roman")
                    elif operation == "key_analysis":
                        await self.service.analyze_key("bach1")
                except Exception as e:
                    logger.warning(f"Operation {operation} failed: {e}")
            
            # Force garbage collection
            gc.collect()
            
            # Check memory after each cycle
            current_mem = self.get_memory_stats()
            growth = current_mem["rss_mb"] - baseline_mem["rss_mb"]
            print(f"   Cycle {cycle + 1}: {growth:.2f}MB growth")
            
            # If growth is significant, investigate
            if growth > 5.0:  # 5MB threshold
                print(f"   ⚠️  Potential memory leak detected: {growth:.2f}MB growth")
                
        self.take_snapshot("leak_final")
        
        # Final analysis
        leak_diff = self.analyze_snapshot_diff("leak_baseline", "leak_final")
        print(f"   Total memory growth: {leak_diff['rss_diff_mb']:.2f}MB")
        
        if leak_diff['rss_diff_mb'] > 10.0:
            print("   ❌ MEMORY LEAK DETECTED")
            print("   Top memory consumers:")
            if "detailed_analysis" in leak_diff:
                for item in leak_diff["detailed_analysis"][:5]:
                    print(f"     {item['size_diff_mb']:.2f}MB: {item['traceback'][:100]}...")
        else:
            print("   ✅ No significant memory leaks detected")
    
    async def profile_concurrent_memory(self):
        """Profile memory usage under concurrent load"""
        print("\n🚀 Concurrent Memory Profiling")
        print("=" * 50)
        
        self.take_snapshot("before_concurrent")
        
        # Simulate concurrent requests
        async def concurrent_operation(op_id: int):
            try:
                # Mix of operations
                if op_id % 3 == 0:
                    return await self.service.analyze_chords("bach1")
                elif op_id % 3 == 1:
                    return await self.service.analyze_harmony("bach1", "roman")
                else:
                    return await self.service.analyze_key("bach1")
            except Exception as e:
                return {"error": str(e)}
        
        # Run 10 concurrent operations
        tasks = [concurrent_operation(i) for i in range(10)]
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        duration = time.time() - start_time
        
        self.take_snapshot("after_concurrent")
        
        # Analyze results
        successful = sum(1 for r in results if isinstance(r, dict) and "error" not in r)
        concurrent_diff = self.analyze_snapshot_diff("before_concurrent", "after_concurrent")
        
        print(f"   Concurrent operations: 10")
        print(f"   Successful operations: {successful}")
        print(f"   Total duration: {duration:.2f}s")
        print(f"   Average per operation: {duration/10:.2f}s")
        print(f"   Memory usage: {concurrent_diff['rss_diff_mb']:.2f}MB")
        print(f"   Memory per operation: {concurrent_diff['rss_diff_mb']/10:.2f}MB")
        
    def generate_memory_report(self):
        """Generate comprehensive memory analysis report"""
        print("\n📋 Memory Analysis Report")
        print("=" * 50)
        
        # Current system stats
        current_stats = self.get_memory_stats()
        baseline_stats = self.baseline_memory
        
        total_growth = current_stats["rss_mb"] - baseline_stats["rss_mb"]
        
        print(f"\n📊 Overall Memory Usage:")
        print(f"   Baseline memory: {baseline_stats['rss_mb']:.1f}MB")
        print(f"   Current memory: {current_stats['rss_mb']:.1f}MB")
        print(f"   Total growth: {total_growth:.1f}MB")
        print(f"   Memory utilization: {current_stats['percent']:.1f}%")
        
        # Resource manager stats
        if self.resource_manager:
            rm_stats = self.resource_manager.get_system_stats()
            print(f"\n🗃️  Resource Manager Stats:")
            print(f"   Scores stored: {rm_stats['storage']['total_scores']}")
            print(f"   Cache hit rate: {rm_stats['storage']['hit_rate_percent']:.1f}%")
            print(f"   Storage memory: {rm_stats['storage']['memory_usage_mb']:.1f}MB")
            print(f"   Memory efficiency: {rm_stats['storage']['memory_utilization_percent']:.1f}%")
        
        # Recommendations
        print(f"\n💡 Memory Optimization Recommendations:")
        
        if total_growth > 50:
            print("   ❌ HIGH MEMORY USAGE - Critical optimizations needed:")
            print("     - Implement more aggressive score cleanup")
            print("     - Reduce cache TTL times")
            print("     - Add memory pressure monitoring")
        elif total_growth > 20:
            print("   ⚠️  MODERATE MEMORY USAGE - Optimizations recommended:")
            print("     - Monitor cache effectiveness")
            print("     - Consider score compression")
        else:
            print("   ✅ MEMORY USAGE ACCEPTABLE")
            
        # Cache optimization recommendations
        if self.resource_manager:
            hit_rate = rm_stats['storage']['hit_rate_percent']
            if hit_rate < 50:
                print("     - Cache hit rate is low, consider warming cache")
            elif hit_rate > 90:
                print("     - Excellent cache performance")
        
        print(f"\n🎯 User Impact Assessment:")
        if total_growth > 100:
            print("   ❌ SEVERE IMPACT - Users will experience slowdowns")
        elif total_growth > 50:
            print("   ⚠️  MODERATE IMPACT - May affect performance under load")
        else:
            print("   ✅ MINIMAL IMPACT - Good for production use")
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.service:
                await self.service.delete_score("bach1")
                await self.service.delete_score("bach2") 
                await self.service.delete_score("mozart1")
                # Clean up test scores
                for i in range(10):
                    try:
                        await self.service.delete_score(f"test_score_{i}")
                    except:
                        pass
        except:
            pass
        
        if self.resource_manager:
            self.resource_manager.shutdown()
            
        tracemalloc.stop()
    
    async def run_memory_profiling(self):
        """Run complete memory profiling suite"""
        print("🧠 Starting Memory Profiling Analysis")
        print("=" * 60)
        
        try:
            await self.setup()
            await self.profile_memory_usage()
            await self.profile_concurrent_memory()
            self.generate_memory_report()
        finally:
            await self.cleanup()
        
        print("\n🏁 Memory Profiling Complete!")


async def main():
    """Main entry point"""
    profiler = MemoryProfiler()
    await profiler.run_memory_profiling()


if __name__ == "__main__":
    asyncio.run(main())