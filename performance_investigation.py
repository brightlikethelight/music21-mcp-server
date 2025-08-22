#!/usr/bin/env python3
"""
Performance Investigation for Music21 MCP Server

This script investigates performance claims and measures actual response times
for key operations to identify any performance bottlenecks.
"""

import asyncio
import time
import statistics
import sys
import os
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from music21_mcp.services import MusicAnalysisService
from music21_mcp.adapters.python_adapter import create_sync_analyzer


class PerformanceInvestigator:
    """Investigate performance claims and measure actual response times"""

    def __init__(self):
        self.results = {}
        self.service = None
        self.sync_analyzer = None

    async def setup(self):
        """Setup test environment"""
        print("🔧 Setting up performance investigation environment...")

        # Initialize async service
        self.service = MusicAnalysisService(max_memory_mb=256, max_scores=50)

        # Initialize sync analyzer
        self.sync_analyzer = create_sync_analyzer()

        print("✅ Environment setup complete")

    async def time_operation(
        self, operation_name: str, operation_func, iterations: int = 10
    ):
        """Time an operation multiple times and calculate statistics"""
        print(f"\n⏱️  Testing {operation_name} ({iterations} iterations)...")

        times = []
        for i in range(iterations):
            start_time = time.perf_counter()
            try:
                result = (
                    await operation_func()
                    if asyncio.iscoroutinefunction(operation_func)
                    else operation_func()
                )
                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000
                times.append(duration_ms)
                print(f"  Iteration {i + 1}: {duration_ms:.2f}ms")
            except Exception as e:
                print(f"  Iteration {i + 1}: FAILED - {str(e)}")
                continue

        if not times:
            return {
                "operation": operation_name,
                "status": "FAILED",
                "error": "All iterations failed",
            }

        stats = {
            "operation": operation_name,
            "status": "SUCCESS",
            "iterations": len(times),
            "mean_ms": statistics.mean(times),
            "median_ms": statistics.median(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "std_dev_ms": statistics.stdev(times) if len(times) > 1 else 0,
            "times": times,
        }

        # Check if under claimed "sub-200ms" performance
        under_200ms = sum(1 for t in times if t < 200) / len(times) * 100
        stats["under_200ms_percent"] = under_200ms

        print(
            f"  📊 Results: Mean={stats['mean_ms']:.2f}ms, Median={stats['median_ms']:.2f}ms"
        )
        print(f"     Min={stats['min_ms']:.2f}ms, Max={stats['max_ms']:.2f}ms")
        print(f"     {under_200ms:.1f}% under 200ms")

        return stats

    async def test_import_operations(self):
        """Test score import performance"""
        print("\n🎵 Testing Score Import Operations")

        # Test corpus import (Bach chorale)
        async def import_bach_chorale():
            return await self.service.import_score(
                "perf_test_bach", "bach/bwv66.6", "corpus"
            )

        bach_stats = await self.time_operation(
            "Import Bach Chorale (Corpus)", import_bach_chorale, 5
        )
        self.results["import_bach_chorale"] = bach_stats

        # Test text notation import
        async def import_text_melody():
            return await self.service.import_score(
                "perf_test_melody", "C4 E4 G4 C5 B4 G4 E4 C4", "text"
            )

        text_stats = await self.time_operation(
            "Import Text Melody", import_text_melody, 10
        )
        self.results["import_text_melody"] = text_stats

        # Cleanup
        try:
            await self.service.delete_score("perf_test_bach")
            await self.service.delete_score("perf_test_melody")
        except:
            pass

    async def test_analysis_operations(self):
        """Test analysis operation performance"""
        print("\n🔍 Testing Analysis Operations")

        # Import a test score first
        await self.service.import_score("analysis_test", "bach/bwv66.6", "corpus")

        # Test key analysis
        async def analyze_key():
            return await self.service.analyze_key("analysis_test")

        key_stats = await self.time_operation("Key Analysis", analyze_key, 10)
        self.results["key_analysis"] = key_stats

        # Test chord analysis
        async def analyze_chords():
            return await self.service.analyze_chords("analysis_test")

        chord_stats = await self.time_operation("Chord Analysis", analyze_chords, 5)
        self.results["chord_analysis"] = chord_stats

        # Test harmony analysis
        async def analyze_harmony():
            return await self.service.analyze_harmony("analysis_test", "roman")

        harmony_stats = await self.time_operation(
            "Harmony Analysis", analyze_harmony, 5
        )
        self.results["harmony_analysis"] = harmony_stats

        # Cleanup
        try:
            await self.service.delete_score("analysis_test")
        except:
            pass

    async def test_resource_operations(self):
        """Test resource management performance"""
        print("\n💾 Testing Resource Management Operations")

        # Test score listing
        async def list_scores():
            return await self.service.list_scores()

        list_stats = await self.time_operation("List Scores", list_scores, 20)
        self.results["list_scores"] = list_stats

        # Test resource stats
        def get_resource_stats():
            return self.service.get_resource_stats()

        resource_stats = await self.time_operation(
            "Get Resource Stats", get_resource_stats, 20
        )
        self.results["resource_stats"] = resource_stats

        # Test performance metrics
        def get_performance_metrics():
            return self.service.get_performance_metrics()

        perf_stats = await self.time_operation(
            "Get Performance Metrics", get_performance_metrics, 20
        )
        self.results["performance_metrics"] = perf_stats

    async def test_bulk_operations(self):
        """Test performance under load"""
        print("\n🔄 Testing Bulk Operations")

        # Import multiple scores
        scores_to_import = [
            ("bulk_bach1", "bach/bwv66.6", "corpus"),
            ("bulk_bach2", "bach/bwv67.7", "corpus"),
            ("bulk_melody1", "C4 E4 G4 C5", "text"),
            ("bulk_melody2", "D4 F#4 A4 D5", "text"),
            ("bulk_melody3", "E4 G#4 B4 E5", "text"),
        ]

        async def import_multiple_scores():
            for score_id, source, source_type in scores_to_import:
                try:
                    await self.service.import_score(score_id, source, source_type)
                except:
                    pass  # Some corpus files might not exist

        bulk_import_stats = await self.time_operation(
            "Bulk Import (5 scores)", import_multiple_scores, 3
        )
        self.results["bulk_import"] = bulk_import_stats

        # Test analysis on multiple scores
        async def analyze_all_keys():
            results = []
            scores = await self.service.list_scores()
            for score_info in scores.get("scores", []):
                try:
                    result = await self.service.analyze_key(score_info["score_id"])
                    results.append(result)
                except:
                    pass
            return results

        bulk_analysis_stats = await self.time_operation(
            "Bulk Key Analysis", analyze_all_keys, 3
        )
        self.results["bulk_analysis"] = bulk_analysis_stats

        # Cleanup
        for score_id, _, _ in scores_to_import:
            try:
                await self.service.delete_score(score_id)
            except:
                pass

    def search_for_performance_claims(self):
        """Search codebase for performance claims"""
        print("\n🔍 Searching for Performance Claims in Codebase...")

        claims_found = []

        # Check documentation files
        doc_files = [
            "README.md",
            "docs/simplified-api.md",
            "docs/architecture.md",
            "INTERFACES.md",
            "SIMPLIFIED.md",
        ]

        for doc_file in doc_files:
            file_path = Path(__file__).parent / doc_file
            if file_path.exists():
                content = file_path.read_text()

                # Look for performance-related claims
                performance_terms = [
                    "sub-200ms",
                    "200ms",
                    "milliseconds",
                    "ms",
                    "fast",
                    "speed",
                    "performance",
                    "benchmark",
                    "latency",
                    "response time",
                ]

                for term in performance_terms:
                    if term.lower() in content.lower():
                        # Find the line containing the term
                        lines = content.split("\n")
                        for i, line in enumerate(lines):
                            if term.lower() in line.lower():
                                claims_found.append(
                                    {
                                        "file": str(file_path),
                                        "line": i + 1,
                                        "content": line.strip(),
                                        "term": term,
                                    }
                                )

        if claims_found:
            print("📄 Performance Claims Found:")
            for claim in claims_found:
                print(f"  {claim['file']}:{claim['line']} - {claim['content']}")
        else:
            print("❌ No explicit performance claims found in documentation")

        return claims_found

    def analyze_results(self):
        """Analyze performance test results"""
        print("\n📊 Performance Analysis Results")
        print("=" * 60)

        # Overall statistics
        all_operations = []
        failed_operations = []

        for op_name, stats in self.results.items():
            if stats["status"] == "SUCCESS":
                all_operations.extend(stats["times"])

                # Check if operation meets "sub-200ms" claim
                under_200_pct = stats["under_200ms_percent"]
                status = (
                    "✅ FAST"
                    if under_200_pct >= 90
                    else "⚠️ SLOW"
                    if under_200_pct >= 50
                    else "❌ VERY SLOW"
                )

                print(f"\n{op_name}:")
                print(f"  Mean: {stats['mean_ms']:.2f}ms")
                print(f"  Under 200ms: {under_200_pct:.1f}% {status}")

                # Identify bottlenecks
                if stats["mean_ms"] > 500:
                    print(f"  🚨 BOTTLENECK: Mean response time > 500ms")
                elif stats["mean_ms"] > 200:
                    print(f"  ⚠️ WARNING: Mean response time > 200ms")
            else:
                failed_operations.append(op_name)
                print(f"\n{op_name}: ❌ FAILED")

        # Overall summary
        if all_operations:
            overall_mean = statistics.mean(all_operations)
            under_200_overall = (
                sum(1 for t in all_operations if t < 200) / len(all_operations) * 100
            )

            print(f"\n🎯 OVERALL PERFORMANCE SUMMARY:")
            print(f"  Total operations tested: {len(all_operations)}")
            print(f"  Overall mean response time: {overall_mean:.2f}ms")
            print(f"  Operations under 200ms: {under_200_overall:.1f}%")

            # Performance verdict
            if under_200_overall >= 90:
                verdict = "✅ EXCELLENT - Meets sub-200ms performance claims"
            elif under_200_overall >= 70:
                verdict = "⚠️ GOOD - Most operations are fast, some optimization needed"
            elif under_200_overall >= 50:
                verdict = "⚠️ FAIR - Significant performance improvements needed"
            else:
                verdict = "❌ POOR - Major performance bottlenecks identified"

            print(f"  Verdict: {verdict}")

        if failed_operations:
            print(f"\n❌ Failed Operations: {', '.join(failed_operations)}")

        # Performance recommendations
        print(f"\n💡 PERFORMANCE RECOMMENDATIONS:")

        # Find slowest operations
        slow_ops = [
            (name, stats)
            for name, stats in self.results.items()
            if stats["status"] == "SUCCESS" and stats["mean_ms"] > 200
        ]
        slow_ops.sort(key=lambda x: x[1]["mean_ms"], reverse=True)

        if slow_ops:
            print("  Priority optimizations needed:")
            for name, stats in slow_ops[:3]:  # Top 3 slowest
                print(f"    - {name}: {stats['mean_ms']:.2f}ms avg")
        else:
            print("  ✅ All operations performing well!")

        # Check for high variability
        variable_ops = [
            (name, stats)
            for name, stats in self.results.items()
            if stats["status"] == "SUCCESS" and stats["std_dev_ms"] > 50
        ]

        if variable_ops:
            print("  High variability operations (inconsistent performance):")
            for name, stats in variable_ops:
                print(f"    - {name}: ±{stats['std_dev_ms']:.2f}ms std dev")

    async def run_investigation(self):
        """Run complete performance investigation"""
        print("🚀 Starting Music21 MCP Server Performance Investigation")
        print("=" * 60)

        # Search for performance claims first
        self.search_for_performance_claims()

        # Setup environment
        await self.setup()

        # Run performance tests
        try:
            await self.test_import_operations()
            await self.test_analysis_operations()
            await self.test_resource_operations()
            await self.test_bulk_operations()
        except Exception as e:
            print(f"❌ Error during testing: {e}")

        # Analyze results
        self.analyze_results()

        print("\n🏁 Performance investigation complete!")


async def main():
    """Main entry point"""
    investigator = PerformanceInvestigator()
    await investigator.run_investigation()


if __name__ == "__main__":
    asyncio.run(main())
