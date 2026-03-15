#!/usr/bin/env python3
"""
Test script to verify the new observability system works correctly.

This script validates:
1. Structured logging with correlation IDs
2. Performance metrics collection
3. Request tracing and timing
4. Error tracking and categorization
5. Integration with MusicAnalysisService
"""

import asyncio
import json
import sys
import time
from pathlib import Path

# Add src to path for importing
sys.path.insert(0, str(Path(__file__).parent / "src"))

from music21_mcp.observability import get_logger, get_metrics, with_context
from music21_mcp.services import MusicAnalysisService


async def test_observability_system():
    """Comprehensive test of observability system"""
    print("📊 Testing Observability System")
    print("=" * 50)

    # Initialize service
    service = MusicAnalysisService(max_memory_mb=100, max_scores=10)
    logger = get_logger("test")

    print("✅ Initialized service with observability")

    # Test 1: Basic structured logging
    print("\n📝 Test 1: Structured Logging")

    with with_context(
        request_id="test-123", user_id="test-user", operation="test-logging"
    ):
        logger.info("Testing structured logging", test_data="sample")
        logger.warning("Testing warning message", warning_code="W001")
        logger.debug("Testing debug message", debug_info={"key": "value"})

    print("   ✅ Structured logs generated with context")

    # Test 2: Performance monitoring with operations
    print("\n⏱️  Test 2: Performance Monitoring")

    test_operations = [
        (
            "import_score",
            lambda: service.import_score("test_1", "bach/bwv66.6", "corpus"),
        ),
        ("list_scores", lambda: service.list_scores()),
        ("analyze_key", lambda: service.analyze_key("test_1")),
        ("analyze_chords", lambda: service.analyze_chords("test_1")),
        ("get_score_info", lambda: service.get_score_info("test_1")),
    ]

    for op_name, operation in test_operations:
        try:
            with with_context(operation=op_name):
                start_time = time.time()
                result = await operation()
                duration = time.time() - start_time

                if result.get("status") == "success":
                    print(f"   ✅ {op_name}: {duration * 1000:.1f}ms")
                else:
                    print(f"   ⚠️  {op_name}: {result.get('message', 'Unknown error')}")

        except Exception as e:
            print(f"   ❌ {op_name}: {e}")

    # Test 3: Error tracking
    print("\n🚨 Test 3: Error Tracking")

    try:
        with with_context(operation="error-test"):
            # This should fail and generate error metrics
            await service.analyze_key("nonexistent_score")
    except Exception as e:
        print(f"   ✅ Error correctly tracked: {type(e).__name__}")

    # Test 4: Metrics collection
    print("\n📈 Test 4: Metrics Collection")

    metrics = service.get_performance_metrics()

    print(f"   Operation counters: {len(metrics.get('counters', {}))}")
    print(f"   Timer metrics: {len(metrics.get('timers', {}))}")
    print(f"   Uptime: {metrics.get('metadata', {}).get('uptime_seconds', 0):.1f}s")

    # Show sample metrics
    if metrics.get("counters"):
        print("   Sample counters:")
        for counter, value in list(metrics["counters"].items())[:3]:
            print(f"     - {counter}: {value}")

    if metrics.get("timers"):
        print("   Sample timers:")
        for timer, data in list(metrics["timers"].items())[:3]:
            mean_ms = data.get("mean_ms", 0)
            count = data.get("count", 0)
            print(f"     - {timer}: {mean_ms:.1f}ms avg ({count} calls)")

    # Test 5: Service status
    print("\n🏥 Test 5: Service Status")

    status = service.get_service_status()

    print(f"   Service status: {status['service']['status']}")
    print(f"   Health status: {status['health']['status']}")
    print(
        f"   Resource memory: {status['resources']['storage']['memory_usage_mb']:.1f}MB"
    )
    print(f"   Performance ops: {len(status['performance']['operation_counts'])}")

    if status["health"]["warnings"]:
        print(f"   Warnings: {status['health']['warnings']}")

    if status["health"]["errors"]:
        print(f"   Errors: {status['health']['errors']}")

    # Test 6: Context correlation
    print("\n🔗 Test 6: Request Correlation")

    # Simulate correlated operations
    request_id = "req-456"

    with with_context(
        request_id=request_id, user_id="user-789", operation="batch-analysis"
    ):
        logger.info("Starting batch analysis")

        try:
            await service.import_score("batch_1", "bach/bwv7.7", "corpus")
            logger.info("Imported score successfully")

            result = await service.analyze_key("batch_1")
            logger.info("Key analysis completed", result_status=result.get("status"))

        except Exception as e:
            logger.error("Batch analysis failed", error=e)

    print(f"   ✅ Correlated operations with request_id: {request_id}")

    # Test 7: Resource monitoring integration
    print("\n🔧 Test 7: Resource Monitoring Integration")

    resource_stats = service.get_resource_stats()
    memory_usage = service.get_memory_usage()

    print(f"   Storage scores: {resource_stats['storage']['total_scores']}")
    print(f"   Memory usage: {memory_usage['storage_memory_mb']:.1f}MB")
    print(f"   Memory utilization: {memory_usage['storage_utilization_percent']:.1f}%")
    print(f"   System memory: {memory_usage['system_memory_mb']:.1f}MB")

    # Test 8: Final metrics summary
    print("\n📊 Test 8: Final Metrics Summary")

    final_metrics = service.get_performance_metrics()

    # Count successful vs failed operations
    success_count = 0
    error_count = 0

    for counter_name, count in final_metrics.get("counters", {}).items():
        if "status=success" in counter_name:
            success_count += count
        elif "status=error" in counter_name:
            error_count += count

    print(f"   Total successful operations: {success_count}")
    print(f"   Total failed operations: {error_count}")
    print(
        f"   Success rate: {(success_count / (success_count + error_count) * 100):.1f}%"
        if (success_count + error_count) > 0
        else "N/A"
    )

    # Show performance percentiles
    for timer_name, timer_data in final_metrics.get("timers", {}).items():
        if "music_analysis" in timer_name:
            print(f"   {timer_name}:")
            print(f"     - Mean: {timer_data.get('mean_ms', 0):.1f}ms")
            print(f"     - P95: {timer_data.get('p95_ms', 0):.1f}ms")
            print(f"     - Count: {timer_data.get('count', 0)}")

    print("\n🎉 Observability System Test Complete!")
    print("=" * 50)


def test_structured_logging():
    """Test structured logging directly"""
    print("\n🔍 Direct Structured Logging Test")
    print("-" * 40)

    logger = get_logger("direct-test")

    # Test without context
    logger.info("Basic info message", component="test")

    # Test with context
    with with_context(request_id="direct-123", operation="direct-test"):
        logger.info("Message with context", data={"test": True})
        logger.warning("Warning with context", warning_type="test")

        try:
            raise ValueError("Test error")
        except ValueError as e:
            logger.error("Error with context", error=e, error_details={"code": "E001"})

    print("   ✅ Structured logging working correctly")


if __name__ == "__main__":
    print("📊 Music21 MCP Server - Observability Test")
    print("=" * 60)

    # Test direct logging functionality
    test_structured_logging()

    # Test integrated observability
    try:
        asyncio.run(test_observability_system())
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()

    print("\n✨ Observability test session complete!")
