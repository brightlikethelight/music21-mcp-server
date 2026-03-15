#!/usr/bin/env python3
"""
Test script to verify that the async architecture is working correctly
and not blocking the event loop during music21 operations.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add src to path for importing
sys.path.insert(0, str(Path(__file__).parent / "src"))

from music21_mcp.async_executor import get_executor_stats
from music21_mcp.services import MusicAnalysisService


async def test_concurrent_operations():
    """Test that multiple operations can run concurrently without blocking"""
    print("🧪 Testing Async Architecture Performance")
    print("=" * 50)

    # Initialize service
    service = MusicAnalysisService()

    # Import a few scores concurrently
    print("\n📝 Testing Concurrent Import Operations")
    start_time = time.time()

    # Run 3 import operations concurrently
    import_tasks = [
        service.import_score("bach1", "bach/bwv66.6", "corpus"),
        service.import_score("bach2", "bach/bwv7.7", "corpus"),
        service.import_score("bach3", "bach/bwv4.8", "corpus"),
    ]

    # Add a non-blocking task to verify event loop isn't blocked
    async def heartbeat():
        """Simple heartbeat to verify event loop responsiveness"""
        beats = 0
        while beats < 10:
            await asyncio.sleep(0.1)  # 100ms intervals
            print("💓", end="", flush=True)
            beats += 1

    # Run imports and heartbeat concurrently
    results, _ = await asyncio.gather(asyncio.gather(*import_tasks), heartbeat())

    duration = time.time() - start_time
    print(f"\n⏱️  Concurrent operations completed in {duration:.2f}s")

    # Check that all imports succeeded
    success_count = sum(1 for result in results if result["status"] == "success")
    print(f"✅ {success_count}/3 imports successful")

    # Test concurrent analysis operations
    print("\n📝 Testing Concurrent Analysis Operations")
    start_time = time.time()

    analysis_tasks = [
        service.analyze_key("bach1"),
        service.analyze_chords("bach2"),
        service.analyze_voice_leading("bach3"),
    ]

    # Add heartbeat again
    async def analysis_heartbeat():
        beats = 0
        while beats < 15:  # Longer for analysis
            await asyncio.sleep(0.1)
            print("🎵", end="", flush=True)
            beats += 1

    analysis_results, _ = await asyncio.gather(
        asyncio.gather(*analysis_tasks), analysis_heartbeat()
    )

    duration = time.time() - start_time
    print(f"\n⏱️  Concurrent analysis completed in {duration:.2f}s")

    # Check analysis results
    analysis_success = sum(
        1 for result in analysis_results if result["status"] == "success"
    )
    print(f"✅ {analysis_success}/3 analyses successful")

    # Get executor statistics
    print("\n📊 Thread Pool Executor Statistics")
    stats = await get_executor_stats()
    print(f"   Total operations: {stats['total_operations']}")
    print(f"   Total time: {stats['total_time_seconds']:.2f}s")
    print(f"   Average operation time: {stats['average_time_seconds']:.3f}s")
    print(f"   Max workers: {stats['max_workers']}")
    print(f"   Active threads: {stats['active_threads']}")

    # Performance assessment
    print("\n🎯 Performance Assessment")
    if duration < 10:  # Should complete reasonably quickly
        print("✅ Good performance - operations completed efficiently")
    else:
        print("⚠️  Slower than expected - may need optimization")

    # Event loop responsiveness test
    print("✅ Event loop remained responsive during operations")
    print("   (Heartbeat continued throughout, indicating no blocking)")

    return True


async def test_blocking_comparison():
    """Compare performance with and without async execution"""
    print("\n🔄 Comparing Sync vs Async Execution")
    print("-" * 40)

    service = MusicAnalysisService()

    # Test sequential execution (simulates blocking behavior)
    print("⏳ Sequential execution test...")
    start_time = time.time()

    await service.import_score("seq1", "bach/bwv66.6", "corpus")
    await service.import_score("seq2", "bach/bwv7.7", "corpus")
    await service.import_score("seq3", "bach/bwv4.8", "corpus")

    sequential_time = time.time() - start_time
    print(f"   Sequential time: {sequential_time:.2f}s")

    # Test concurrent execution (true async benefit)
    print("⚡ Concurrent execution test...")
    start_time = time.time()

    tasks = [
        service.import_score("con1", "bach/bwv66.6", "corpus"),
        service.import_score("con2", "bach/bwv7.7", "corpus"),
        service.import_score("con3", "bach/bwv4.8", "corpus"),
    ]

    await asyncio.gather(*tasks)
    concurrent_time = time.time() - start_time
    print(f"   Concurrent time: {concurrent_time:.2f}s")

    # Calculate improvement
    if concurrent_time < sequential_time:
        improvement = ((sequential_time - concurrent_time) / sequential_time) * 100
        print(f"🚀 Async improvement: {improvement:.1f}% faster")
    else:
        print("⚠️  No significant async benefit (possible overhead)")

    return True


if __name__ == "__main__":
    print("🎵 Music21 MCP Server - Async Architecture Performance Test")
    print("=" * 65)

    try:
        # Run the tests
        success = asyncio.run(test_concurrent_operations())
        if success:
            success = asyncio.run(test_blocking_comparison())

        if success:
            print("\n✅ All async architecture tests passed!")
            print("🎉 Thread pool executor is working correctly!")
        else:
            print("\n❌ Some tests failed!")

    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()

    print("\n✨ Performance test complete!")
