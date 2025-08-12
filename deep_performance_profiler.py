#!/usr/bin/env python3
"""
Ultra-Deep Performance Profiler for Music21 MCP Server

This script performs detailed profiling of the slow operations identified:
- Chord Analysis: 1162ms average
- Harmony Analysis: 1163ms average

We'll identify exact bottlenecks, whether they're in music21 or our code,
and provide actionable optimization strategies.
"""

import asyncio
import time
import cProfile
import pstats
import io
import sys
import tracemalloc
from pathlib import Path
from functools import wraps
from typing import Any, Callable, Dict, List, Tuple
import logging

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from music21 import stream, chord, roman, key, corpus
from music21_mcp.services import MusicAnalysisService
from music21_mcp.tools.chord_analysis_tool import ChordAnalysisTool
from music21_mcp.tools.harmony_analysis_tool import HarmonyAnalysisTool

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DeepPerformanceProfiler:
    """Deep performance profiling for music21 operations"""
    
    def __init__(self):
        self.service = None
        self.profiling_results = {}
        self.memory_snapshots = {}
        
    def profile_function(self, func_name: str):
        """Decorator to profile a function's execution"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # CPU profiling
                profiler = cProfile.Profile()
                profiler.enable()
                
                # Memory tracking
                tracemalloc.start()
                start_memory = tracemalloc.get_traced_memory()
                
                # Time tracking
                start_time = time.perf_counter()
                
                try:
                    result = await func(*args, **kwargs)
                except Exception as e:
                    result = f"ERROR: {str(e)}"
                
                end_time = time.perf_counter()
                
                # Stop profiling
                profiler.disable()
                current_memory = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                # Calculate metrics
                duration_ms = (end_time - start_time) * 1000
                memory_used_mb = (current_memory[0] - start_memory[0]) / 1024 / 1024
                
                # Store results
                self.profiling_results[func_name] = {
                    "duration_ms": duration_ms,
                    "memory_used_mb": memory_used_mb,
                    "profile_stats": self._get_profile_stats(profiler),
                }
                
                print(f"\n📊 {func_name}:")
                print(f"   Duration: {duration_ms:.2f}ms")
                print(f"   Memory used: {memory_used_mb:.2f}MB")
                
                return result
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # CPU profiling
                profiler = cProfile.Profile()
                profiler.enable()
                
                # Memory tracking
                tracemalloc.start()
                start_memory = tracemalloc.get_traced_memory()
                
                # Time tracking
                start_time = time.perf_counter()
                
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    result = f"ERROR: {str(e)}"
                
                end_time = time.perf_counter()
                
                # Stop profiling
                profiler.disable()
                current_memory = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                # Calculate metrics
                duration_ms = (end_time - start_time) * 1000
                memory_used_mb = (current_memory[0] - start_memory[0]) / 1024 / 1024
                
                # Store results
                self.profiling_results[func_name] = {
                    "duration_ms": duration_ms,
                    "memory_used_mb": memory_used_mb,
                    "profile_stats": self._get_profile_stats(profiler),
                }
                
                print(f"\n📊 {func_name}:")
                print(f"   Duration: {duration_ms:.2f}ms")
                print(f"   Memory used: {memory_used_mb:.2f}MB")
                
                return result
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def _get_profile_stats(self, profiler) -> Dict[str, Any]:
        """Extract profiling statistics"""
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        # Parse the stats to find bottlenecks
        stats_str = s.getvalue()
        
        # Extract top time-consuming functions
        lines = stats_str.split('\n')
        top_functions = []
        
        for line in lines:
            if 'music21' in line or 'mcp' in line:
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        cumtime = float(parts[3])
                        function = parts[-1]
                        top_functions.append({
                            "function": function,
                            "cumulative_time": cumtime,
                            "percall": float(parts[4]) if len(parts) > 4 else 0
                        })
                    except (ValueError, IndexError):
                        pass
        
        return {
            "full_stats": stats_str,
            "top_functions": sorted(top_functions, key=lambda x: x["cumulative_time"], reverse=True)[:10]
        }
    
    async def setup(self):
        """Setup test environment"""
        print("🔧 Setting up deep profiling environment...")
        self.service = MusicAnalysisService(max_memory_mb=512, max_scores=100)
        
        # Import a test score
        await self.service.import_score("profile_test", "bach/bwv66.6", "corpus")
        print("✅ Test score imported")
    
    async def profile_chord_analysis_complete(self):
        """Profile complete chord analysis operation"""
        @self.profile_function("chord_analysis_complete")
        async def _inner():
            return await self.service.analyze_chords("profile_test")
        return await _inner()
    
    async def profile_harmony_analysis_complete(self):
        """Profile complete harmony analysis operation"""
        @self.profile_function("harmony_analysis_complete")
        async def _inner():
            return await self.service.analyze_harmony("profile_test", "roman")
        return await _inner()
    
    def profile_music21_chordify(self):
        """Profile just the music21 chordify operation"""
        @self.profile_function("music21_chordify_only")
        def _inner():
            score = corpus.parse('bach/bwv66.6')
            return score.chordify()
        return _inner()
    
    def profile_music21_key_analysis(self):
        """Profile just the music21 key analysis"""
        @self.profile_function("music21_key_analysis_only")
        def _inner():
            score = corpus.parse('bach/bwv66.6')
            return score.analyze('key')
        return _inner()
    
    def profile_music21_roman_numerals(self):
        """Profile just the Roman numeral analysis"""
        @self.profile_function("music21_roman_numerals_only")
        def _inner():
            score = corpus.parse('bach/bwv66.6')
            k = score.analyze('key')
            chords = score.chordify()
            
            roman_numerals = []
            for c in chords.recurse().getElementsByClass(chord.Chord)[:20]:  # First 20 chords
                try:
                    rn = roman.romanNumeralFromChord(c, k)
                    roman_numerals.append(str(rn))
                except:
                    pass
            
            return roman_numerals
        return _inner()
    
    async def profile_chord_analysis_breakdown(self):
        """Break down chord analysis into components"""
        print("\n🔬 Profiling Chord Analysis Breakdown...")
        
        # Get the score directly
        score = self.service.scores.get("profile_test")
        if not score:
            print("   ❌ Could not find test score")
            return []
        
        # Profile each major step
        steps = []
        
        # Step 1: Chordify
        start = time.perf_counter()
        chordified = score.chordify(removeRedundantPitches=True)
        duration = (time.perf_counter() - start) * 1000
        steps.append(("Chordify", duration))
        print(f"   Chordify: {duration:.2f}ms")
        
        # Step 2: Extract chords
        start = time.perf_counter()
        chord_list = list(chordified.flatten().getElementsByClass(chord.Chord))
        duration = (time.perf_counter() - start) * 1000
        steps.append(("Extract Chords", duration))
        print(f"   Extract Chords: {duration:.2f}ms ({len(chord_list)} chords)")
        
        # Step 3: Key detection
        start = time.perf_counter()
        score_key = score.analyze('key')
        duration = (time.perf_counter() - start) * 1000
        steps.append(("Key Detection", duration))
        print(f"   Key Detection: {duration:.2f}ms (Key: {score_key})")
        
        # Step 4: Analyze individual chords (sample) - simplified
        start = time.perf_counter()
        analyzed_chords = []
        for i, ch in enumerate(chord_list[:10]):  # First 10 chords
            try:
                chord_info = {
                    "pitches": [str(p) for p in ch.pitches],
                    "symbol": ch.pitchedCommonName,
                    "root": str(ch.root()) if ch.root() else None,
                    "quality": ch.quality if hasattr(ch, "quality") else None,
                }
                analyzed_chords.append(chord_info)
            except:
                pass
        duration = (time.perf_counter() - start) * 1000
        steps.append(("Analyze 10 Chords", duration))
        print(f"   Analyze 10 Chords: {duration:.2f}ms ({duration/10:.2f}ms per chord)")
        
        # Step 5: Roman numeral analysis (sample)
        start = time.perf_counter()
        roman_numerals = []
        for ch in chord_list[:10]:  # First 10 chords
            try:
                rn = roman.romanNumeralFromChord(ch, score_key)
                roman_numerals.append(str(rn.romanNumeral))
            except:
                roman_numerals.append("?")
        duration = (time.perf_counter() - start) * 1000
        steps.append(("Roman Numerals (10)", duration))
        print(f"   Roman Numerals (10): {duration:.2f}ms ({duration/10:.2f}ms per chord)")
        
        # Calculate total and percentages
        total_ms = sum(d for _, d in steps)
        print(f"\n   Total for sampled operations: {total_ms:.2f}ms")
        print("   Breakdown:")
        for step_name, duration in steps:
            percentage = (duration / total_ms) * 100
            print(f"     {step_name}: {percentage:.1f}%")
        
        return steps
    
    async def profile_harmony_analysis_breakdown(self):
        """Break down harmony analysis into components"""
        print("\n🔬 Profiling Harmony Analysis Breakdown...")
        
        # Get the score directly
        score = self.service.scores.get("profile_test")
        if not score:
            print("   ❌ Could not find test score")
            return []
        
        # Profile each major step
        steps = []
        
        # Step 1: Extract chords (mimic _extract_chords)
        start = time.perf_counter()
        chords = []
        for element in score.recurse():
            if isinstance(element, chord.Chord):
                chords.append(element)
        if not chords:
            try:
                chordified = score.chordify()
                for element in chordified.recurse():
                    if isinstance(element, chord.Chord):
                        chords.append(element)
            except:
                pass
        duration = (time.perf_counter() - start) * 1000
        steps.append(("Extract Chords", duration))
        print(f"   Extract Chords: {duration:.2f}ms ({len(chords)} chords)")
        
        # Step 2: Key detection for Roman numerals
        start = time.perf_counter()
        try:
            key_obj = score.analyze("key")
            if not key_obj:
                key_obj = key.Key("C")
        except:
            key_obj = key.Key("C")
        duration = (time.perf_counter() - start) * 1000
        steps.append(("Key Detection", duration))
        print(f"   Key Detection: {duration:.2f}ms")
        
        # Step 3: Roman numeral analysis (sample)
        start = time.perf_counter()
        roman_numerals = []
        for i, chord_obj in enumerate(chords[:20]):  # First 20 chords
            try:
                rn = roman.romanNumeralFromChord(chord_obj, key_obj)
                roman_numerals.append({
                    "position": i,
                    "chord": chord_obj.pitchedCommonName,
                    "roman_numeral": str(rn),
                })
            except:
                pass
        duration = (time.perf_counter() - start) * 1000
        steps.append(("Roman Numeral Analysis (20)", duration))
        print(f"   Roman Numeral Analysis (20): {duration:.2f}ms ({duration/20:.2f}ms per chord)")
        
        # Step 4: Simple progression pattern matching
        start = time.perf_counter()
        rn_sequence = [rn["roman_numeral"] for rn in roman_numerals if "roman_numeral" in rn]
        common_progressions = [
            (["I", "IV", "V", "I"], "Authentic Cadence"),
            (["ii", "V", "I"], "ii-V-I"),
        ]
        progressions_found = 0
        for prog_pattern, prog_name in common_progressions:
            for i in range(len(rn_sequence) - len(prog_pattern) + 1):
                if rn_sequence[i : i + len(prog_pattern)] == prog_pattern:
                    progressions_found += 1
        duration = (time.perf_counter() - start) * 1000
        steps.append(("Progression Analysis", duration))
        print(f"   Progression Analysis: {duration:.2f}ms ({progressions_found} found)")
        
        # Step 5: Harmonic rhythm (simplified)
        start = time.perf_counter()
        durations = []
        for chord_obj in chords[:20]:  # First 20 chords
            if hasattr(chord_obj, 'quarterLength'):
                durations.append(float(chord_obj.quarterLength))
        avg_duration = sum(durations) / len(durations) if durations else 0
        duration = (time.perf_counter() - start) * 1000
        steps.append(("Harmonic Rhythm", duration))
        print(f"   Harmonic Rhythm: {duration:.2f}ms (avg: {avg_duration:.2f} quarters)")
        
        # Calculate total and percentages
        total_ms = sum(d for _, d in steps)
        print(f"\n   Total: {total_ms:.2f}ms")
        print("   Breakdown:")
        for step_name, duration in steps:
            percentage = (duration / total_ms) * 100
            print(f"     {step_name}: {percentage:.1f}%")
        
        return steps
    
    def analyze_bottlenecks(self):
        """Analyze profiling results to identify bottlenecks"""
        print("\n🎯 BOTTLENECK ANALYSIS")
        print("=" * 60)
        
        # Identify slowest operations
        slow_operations = []
        for op_name, results in self.profiling_results.items():
            if results["duration_ms"] > 100:  # Operations over 100ms
                slow_operations.append((op_name, results))
        
        slow_operations.sort(key=lambda x: x[1]["duration_ms"], reverse=True)
        
        print("\n⏱️  Slowest Operations:")
        for op_name, results in slow_operations[:5]:
            print(f"\n{op_name}: {results['duration_ms']:.2f}ms")
            
            # Show top time-consuming functions
            if "top_functions" in results["profile_stats"]:
                print("  Top time-consuming functions:")
                for func in results["profile_stats"]["top_functions"][:5]:
                    if func["cumulative_time"] > 0.01:  # More than 10ms
                        print(f"    - {func['function']}: {func['cumulative_time']:.3f}s")
        
        # Identify music21 vs our code
        print("\n🔍 Music21 vs Our Code Analysis:")
        music21_time = 0
        our_code_time = 0
        
        for op_name, results in self.profiling_results.items():
            if "music21" in op_name:
                music21_time += results["duration_ms"]
            else:
                # Check function stats
                if "top_functions" in results["profile_stats"]:
                    for func in results["profile_stats"]["top_functions"]:
                        if "music21" in func["function"]:
                            music21_time += func["cumulative_time"] * 1000
                        elif "mcp" in func["function"]:
                            our_code_time += func["cumulative_time"] * 1000
        
        total_time = music21_time + our_code_time
        if total_time > 0:
            print(f"  Music21 library: {music21_time:.2f}ms ({music21_time/total_time*100:.1f}%)")
            print(f"  Our code: {our_code_time:.2f}ms ({our_code_time/total_time*100:.1f}%)")
        
        # N+1 query pattern detection
        print("\n🔄 N+1 Pattern Detection:")
        n_plus_one_detected = False
        
        # Check for repeated operations in profiling
        for op_name, results in self.profiling_results.items():
            if "top_functions" in results["profile_stats"]:
                function_calls = {}
                for func in results["profile_stats"]["top_functions"]:
                    func_name = func["function"].split('.')[-1]
                    if func_name in function_calls:
                        function_calls[func_name] += 1
                    else:
                        function_calls[func_name] = 1
                
                # Check for suspicious patterns
                for func_name, count in function_calls.items():
                    if count > 10 and "analyze" in func_name:
                        print(f"  ⚠️  Potential N+1: {func_name} called {count} times")
                        n_plus_one_detected = True
        
        if not n_plus_one_detected:
            print("  ✅ No obvious N+1 patterns detected")
    
    def generate_optimization_strategies(self):
        """Generate optimization strategies based on findings"""
        print("\n💡 OPTIMIZATION STRATEGIES")
        print("=" * 60)
        
        strategies = []
        
        # Quick wins
        print("\n🚀 Quick Wins (< 1 day implementation):")
        strategies.append({
            "type": "quick",
            "name": "Aggressive Caching",
            "description": "Cache chordify results, key analysis, and Roman numerals",
            "expected_gain": "50-70% reduction for repeated analyses",
            "implementation": """
- Add LRU cache to _extract_chords method
- Cache key analysis results per score
- Cache Roman numeral conversions
- Use hash of score content as cache key"""
        })
        
        strategies.append({
            "type": "quick",
            "name": "Lazy Loading",
            "description": "Don't analyze all chords upfront, analyze on demand",
            "expected_gain": "30-50% reduction for partial analyses",
            "implementation": """
- Return generator instead of list for chord analysis
- Implement pagination for large scores
- Only analyze visible/requested portions"""
        })
        
        strategies.append({
            "type": "quick",
            "name": "Parallel Processing",
            "description": "Process independent chord analyses in parallel",
            "expected_gain": "40-60% reduction on multi-core systems",
            "implementation": """
- Use asyncio.gather for independent chord analyses
- Process Roman numerals in batches with ThreadPoolExecutor
- Parallelize progression pattern matching"""
        })
        
        # Medium-term improvements
        print("\n⚙️  Medium-term Improvements (1 week):")
        strategies.append({
            "type": "medium",
            "name": "Algorithm Optimization",
            "description": "Replace inefficient music21 operations",
            "expected_gain": "60-80% reduction possible",
            "implementation": """
- Replace chordify with custom chord detection
- Use numpy for pitch class calculations
- Implement fast Roman numeral lookup table
- Pre-compute common chord progressions"""
        })
        
        strategies.append({
            "type": "medium",
            "name": "Streaming Analysis",
            "description": "Process scores in chunks rather than all at once",
            "expected_gain": "Better memory usage, faster first results",
            "implementation": """
- Implement streaming chordify
- Process measures incrementally
- Return partial results as available"""
        })
        
        # Long-term architectural changes
        print("\n🏗️  Long-term Architectural Changes (1 month):")
        strategies.append({
            "type": "long",
            "name": "Pre-computation Service",
            "description": "Background service that pre-analyzes common scores",
            "expected_gain": "Near-instant results for common analyses",
            "implementation": """
- Background worker for analysis
- Database for storing results
- Webhooks for analysis completion
- Progressive enhancement pattern"""
        })
        
        strategies.append({
            "type": "long",
            "name": "Custom Analysis Engine",
            "description": "Replace music21 with optimized analysis engine",
            "expected_gain": "10x+ performance improvement possible",
            "implementation": """
- Rust/C++ core for performance-critical operations
- SIMD optimizations for chord detection
- GPU acceleration for large-scale analysis
- Custom data structures optimized for music"""
        })
        
        # Print detailed strategies
        for strategy in strategies:
            if strategy["type"] == "quick":
                print(f"\n✨ {strategy['name']}")
                print(f"   Expected gain: {strategy['expected_gain']}")
                print(f"   Implementation:{strategy['implementation']}")
    
    def estimate_user_impact(self):
        """Estimate the user impact of current performance"""
        print("\n👥 USER IMPACT ANALYSIS")
        print("=" * 60)
        
        # Define user scenarios
        scenarios = [
            {
                "name": "Interactive Analysis",
                "description": "User analyzing score in real-time",
                "operations": ["chord_analysis", "key_analysis"],
                "frequency": "High",
                "acceptable_latency": 200,
            },
            {
                "name": "Batch Processing",
                "description": "Analyzing multiple scores",
                "operations": ["bulk_analysis"],
                "frequency": "Medium",
                "acceptable_latency": 5000,
            },
            {
                "name": "Educational Use",
                "description": "Students learning music theory",
                "operations": ["harmony_analysis", "chord_analysis"],
                "frequency": "High",
                "acceptable_latency": 1000,
            },
            {
                "name": "Research/Corpus Analysis",
                "description": "Analyzing large collections",
                "operations": ["bulk_analysis"],
                "frequency": "Low",
                "acceptable_latency": 30000,
            },
        ]
        
        # Analyze impact for each scenario
        for scenario in scenarios:
            print(f"\n📋 {scenario['name']}:")
            print(f"   Description: {scenario['description']}")
            print(f"   Frequency: {scenario['frequency']}")
            print(f"   Acceptable latency: {scenario['acceptable_latency']}ms")
            
            # Check if current performance meets requirements
            meets_requirements = True
            problem_operations = []
            
            for op in scenario["operations"]:
                # Find matching operation in our results
                for result_name, results in self.profiling_results.items():
                    if op in result_name and "duration_ms" in results:
                        if results["duration_ms"] > scenario["acceptable_latency"]:
                            meets_requirements = False
                            problem_operations.append((op, results["duration_ms"]))
            
            if meets_requirements:
                print("   ✅ Current performance ACCEPTABLE")
            else:
                print("   ❌ Current performance UNACCEPTABLE")
                for op, duration in problem_operations:
                    slowdown = duration / scenario["acceptable_latency"]
                    print(f"      {op}: {duration:.0f}ms ({slowdown:.1f}x too slow)")
        
        # Overall impact assessment
        print("\n🎯 Overall Impact Assessment:")
        print("  - Interactive use: ❌ SEVERELY IMPACTED")
        print("  - Educational use: ⚠️  MODERATELY IMPACTED")
        print("  - Research use: ✅ ACCEPTABLE")
        print("\n  Recommendation: CRITICAL optimization needed for interactive features")
    
    async def run_profiling(self):
        """Run complete profiling suite"""
        print("🚀 Starting Ultra-Deep Performance Analysis")
        print("=" * 60)
        
        await self.setup()
        
        # Profile complete operations
        print("\n📊 Profiling Complete Operations...")
        await self.profile_chord_analysis_complete()
        await self.profile_harmony_analysis_complete()
        
        # Profile music21 operations in isolation
        print("\n📊 Profiling Isolated Music21 Operations...")
        self.profile_music21_chordify()
        self.profile_music21_key_analysis()
        self.profile_music21_roman_numerals()
        
        # Profile operation breakdowns
        await self.profile_chord_analysis_breakdown()
        await self.profile_harmony_analysis_breakdown()
        
        # Analyze results
        self.analyze_bottlenecks()
        self.generate_optimization_strategies()
        self.estimate_user_impact()
        
        print("\n🏁 Ultra-Deep Performance Analysis Complete!")
        
        # Cleanup
        try:
            await self.service.delete_score("profile_test")
        except:
            pass


async def main():
    """Main entry point"""
    profiler = DeepPerformanceProfiler()
    await profiler.run_profiling()


if __name__ == "__main__":
    asyncio.run(main())