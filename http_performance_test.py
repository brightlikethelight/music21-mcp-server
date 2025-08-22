#!/usr/bin/env python3
"""
HTTP API Performance Test for Music21 MCP Server
"""

import asyncio
import time
import requests
import json
import statistics
from pathlib import Path
import sys

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from music21_mcp.adapters.http_adapter import HTTPAdapter
from music21_mcp.services import MusicAnalysisService


class HTTPPerformanceTester:
    """Test HTTP API performance"""

    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.service = None
        self.adapter = None

    async def setup_server(self):
        """Setup HTTP server for testing"""
        print("🔧 Setting up HTTP server...")
        self.service = MusicAnalysisService(max_memory_mb=256, max_scores=50)
        self.adapter = HTTPAdapter(self.service)

        # Note: In a real test, we'd start the server in a separate process
        # For now, we'll test the adapter directly
        print("✅ HTTP adapter ready")

    def time_http_request(
        self,
        operation_name: str,
        method: str,
        endpoint: str,
        data: dict = None,
        iterations: int = 5,
    ):
        """Time HTTP requests"""
        print(f"\n⏱️  Testing HTTP {operation_name} ({iterations} iterations)...")

        times = []
        for i in range(iterations):
            start_time = time.perf_counter()
            try:
                if method.upper() == "POST":
                    response = requests.post(
                        f"{self.base_url}{endpoint}", json=data, timeout=30
                    )
                else:
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=30)

                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000

                if response.status_code == 200:
                    times.append(duration_ms)
                    print(f"  Iteration {i + 1}: {duration_ms:.2f}ms ✅")
                else:
                    print(f"  Iteration {i + 1}: HTTP {response.status_code} ❌")

            except requests.RequestException as e:
                print(f"  Iteration {i + 1}: ERROR - {str(e)} ❌")
                continue

        if not times:
            return {"status": "FAILED", "error": "All requests failed"}

        return {
            "operation": operation_name,
            "status": "SUCCESS",
            "mean_ms": statistics.mean(times),
            "median_ms": statistics.median(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "times": times,
        }

    async def test_direct_adapter_performance(self):
        """Test adapter performance directly (simulating HTTP calls)"""
        print("\n🚀 Testing HTTP Adapter Performance (Direct)")
        results = {}

        # Test import
        start_time = time.perf_counter()
        try:
            result = await self.service.import_score(
                "http_test", "bach/bwv66.6", "corpus"
            )
            duration = (time.perf_counter() - start_time) * 1000
            print(f"Import Bach Chorale: {duration:.2f}ms")
            results["import"] = duration
        except Exception as e:
            print(f"Import failed: {e}")
            results["import"] = None

        # Test key analysis
        start_time = time.perf_counter()
        try:
            result = await self.service.analyze_key("http_test")
            duration = (time.perf_counter() - start_time) * 1000
            print(f"Key Analysis: {duration:.2f}ms")
            results["key_analysis"] = duration
        except Exception as e:
            print(f"Key analysis failed: {e}")
            results["key_analysis"] = None

        # Test list scores
        start_time = time.perf_counter()
        try:
            result = await self.service.list_scores()
            duration = (time.perf_counter() - start_time) * 1000
            print(f"List Scores: {duration:.2f}ms")
            results["list_scores"] = duration
        except Exception as e:
            print(f"List scores failed: {e}")
            results["list_scores"] = None

        return results

    async def run_test(self):
        """Run HTTP performance test"""
        print("🌐 HTTP API Performance Test")
        print("=" * 50)

        await self.setup_server()

        # Test direct adapter performance
        direct_results = await self.test_direct_adapter_performance()

        print("\n📊 HTTP Adapter Performance Results:")
        for operation, duration in direct_results.items():
            if duration is not None:
                status = (
                    "✅ FAST"
                    if duration < 200
                    else "⚠️ SLOW"
                    if duration < 500
                    else "❌ VERY SLOW"
                )
                print(f"  {operation}: {duration:.2f}ms {status}")
            else:
                print(f"  {operation}: FAILED ❌")

        print("\n🏁 HTTP performance test complete!")


async def main():
    """Main entry point"""
    tester = HTTPPerformanceTester()
    await tester.run_test()


if __name__ == "__main__":
    asyncio.run(main())
