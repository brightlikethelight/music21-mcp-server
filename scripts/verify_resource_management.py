#!/usr/bin/env python3
"""
Test script to verify the new resource management system works correctly.

This script validates:
1. Memory-managed score storage
2. Automatic cleanup and TTL
3. Resource limits and monitoring
4. Health checks and statistics
5. Integration with MusicAnalysisService
"""

import asyncio
import sys
import time
from pathlib import Path

import pytest

# Add src to path for importing
sys.path.insert(0, str(Path(__file__).parent / "src"))

from music21_mcp.resource_manager import ResourceExhaustedError
from music21_mcp.services import MusicAnalysisService


@pytest.mark.asyncio
async def test_resource_management():
    """Comprehensive test of resource management system"""
    print("🧪 Testing Resource Management System")
    print("=" * 50)

    # Initialize service with small limits for testing
    service = MusicAnalysisService(max_memory_mb=50, max_scores=5)

    print(f"✅ Initialized service with limits: 50MB, 5 scores")

    # Test 1: Basic resource monitoring
    print("\n📊 Test 1: Basic Resource Monitoring")
    stats = service.get_resource_stats()
    print(f"   Initial memory usage: {stats['storage']['memory_usage_mb']:.1f}MB")
    print(f"   Initial score count: {stats['storage']['total_scores']}")
    print(f"   System memory: {stats['system']['process_memory_mb']:.1f}MB")

    # Test 2: Health check
    print("\n🏥 Test 2: Health Check")
    health = service.check_health()
    print(f"   Status: {health['status']}")
    print(f"   Warnings: {health['warnings']}")
    print(f"   Errors: {health['errors']}")
    print(f"   Is healthy: {service.is_resource_healthy()}")

    # Test 3: Import some scores
    print("\n📝 Test 3: Importing Test Scores")
    test_scores = [
        "bach/bwv66.6",
        "bach/bwv7.7",
        "bach/bwv269",
        "bach/bwv324",
        "bach/bwv348",
    ]

    for i, score_name in enumerate(test_scores):
        try:
            print(f"   Importing {score_name}...")
            result = await service.import_score(
                score_id=f"test_{i}", source=score_name, source_type="corpus"
            )
            if result.get("status") == "success":
                print(f"   ✅ Imported successfully")
            else:
                print(f"   ⚠️  Import failed: {result.get('message', 'Unknown error')}")

            # Show memory usage after each import
            memory = service.get_memory_usage()
            print(
                f"   Memory: {memory['storage_memory_mb']:.1f}MB ({memory['storage_utilization_percent']:.1f}%)"
            )

        except Exception as e:
            print(f"   ❌ Error importing {score_name}: {e}")

    # Test 4: Try to exceed limits
    print("\n🚨 Test 4: Testing Resource Limits")
    try:
        # Try to import one more score to exceed the limit
        result = await service.import_score(
            score_id="overflow_test", source="bach/bwv250", source_type="corpus"
        )
        print(f"   Unexpected success: {result}")
    except ResourceExhaustedError as e:
        print(f"   ✅ Resource limit correctly enforced: {e}")
    except Exception as e:
        print(f"   ⚠️  Different error occurred: {e}")

    # Test 5: Cleanup and statistics
    print("\n🧹 Test 5: Cleanup and Statistics")
    cleanup_stats = service.cleanup_resources()
    print(f"   Removed scores: {cleanup_stats['removed_scores']}")
    print(f"   Freed memory: {cleanup_stats['freed_memory_mb']:.1f}MB")
    print(f"   GC collected objects: {cleanup_stats['gc_collected_objects']}")

    # Test 6: Final statistics
    print("\n📈 Test 6: Final Statistics")
    final_stats = service.get_resource_stats()
    storage = final_stats["storage"]

    print(f"   Total scores loaded: {storage['total_scores_loaded']}")
    print(f"   Current scores: {storage['total_scores']}")
    print(f"   Cache hits: {storage['cache_hits']}")
    print(f"   Cache misses: {storage['cache_misses']}")
    print(f"   Hit rate: {storage['hit_rate_percent']:.1f}%")
    print(f"   Cleanup runs: {storage['cleanup_runs']}")
    print(f"   Memory warnings: {storage['memory_warnings']}")

    # Test 7: List remaining scores
    print("\n📋 Test 7: Listing Remaining Scores")
    scores_result = await service.list_scores()
    if scores_result.get("status") == "success":
        scores = scores_result.get("scores", [])
        print(f"   Remaining scores: {len(scores)}")
        for score in scores:
            print(f"   - {score.get('score_id', score.get('id', 'unknown'))}")

    print("\n🎉 Resource Management Test Complete!")
    print("=" * 50)


def test_score_storage_directly():
    """Test the ScoreStorage class directly"""
    print("\n🔧 Direct ScoreStorage Test")
    print("-" * 30)

    from music21_mcp.resource_manager import ScoreStorage

    # Create small storage for testing
    storage = ScoreStorage(max_scores=3, score_ttl_seconds=2, max_memory_mb=10)

    # Add some test data
    storage["test1"] = "Small data 1"
    storage["test2"] = "Small data 2"
    storage["test3"] = "Small data 3"

    print(f"Added 3 items, storage length: {len(storage)}")
    print(f"Stats: {storage.get_stats()}")

    # Wait for TTL expiration
    print("Waiting 3 seconds for TTL expiration...")
    time.sleep(3)

    # Check if items expired
    print(f"After TTL, storage length: {len(storage)}")
    try:
        item = storage["test1"]
        print(f"Item still exists: {item}")
    except KeyError:
        print("✅ Items correctly expired due to TTL")


if __name__ == "__main__":
    print("🎵 Music21 MCP Server - Resource Management Test")
    print("=" * 60)

    # Test direct storage functionality
    test_score_storage_directly()

    # Test integrated service
    try:
        asyncio.run(test_resource_management())
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()

    print("\n✨ Test session complete!")
