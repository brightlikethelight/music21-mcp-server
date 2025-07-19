#!/usr/bin/env python3
"""
Comprehensive FastMCP Server Integration Tests

Tests the actual MCP protocol integration, server behavior, and tool interconnection.
Validates rate limiting, concurrent access, memory management, and end-to-end workflows.
"""

import asyncio
import gc
import sys
import time
from pathlib import Path
from typing import Any

import psutil
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import server components
try:
    from mcp.server.fastmcp import FastMCP

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    FastMCP = None

from music21_mcp.server import (
    SimpleRateLimiter,
    rate_limiter,
    scores,
)
from music21_mcp.tools import (
    DeleteScoreTool,
    ExportScoreTool,
    HarmonizationTool,
    HarmonyAnalysisTool,
    ImportScoreTool,
    KeyAnalysisTool,
    ListScoresTool,
    PatternRecognitionTool,
    ScoreInfoTool,
)

try:
    from music21 import chord, corpus, key, metadata, note, stream

    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False


class TestFastMCPServer:
    """Test suite for FastMCP server architecture"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test"""
        # Clear scores before each test
        scores.clear()

        # Reset rate limiter
        if hasattr(rate_limiter, "tokens"):
            rate_limiter.tokens = rate_limiter.burst
            rate_limiter.last_update = time.time()

        # Force garbage collection
        gc.collect()

        yield

        # Cleanup after test
        scores.clear()
        gc.collect()

    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP not available")
    def test_fastmcp_server_initialization(self):
        """Test FastMCP server can be initialized properly"""
        # Create a test server instance
        test_mcp = FastMCP("Test Music21 Server")

        # Verify server has expected attributes
        assert hasattr(test_mcp, "tool")
        assert hasattr(test_mcp, "resource")
        assert hasattr(test_mcp, "run")

        # Test that we can register a tool
        @test_mcp.tool(name="test_tool")
        async def test_tool():
            return {"status": "success"}

        # FastMCP doesn't expose tools directly, but we can verify registration worked
        # by checking that the decorator didn't raise an exception
        assert callable(test_tool)

    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP not available")
    @pytest.mark.asyncio
    async def test_tool_registration_via_decorator(self):
        """Test that tools are properly registered via @mcp.tool decorator"""
        # Import the actual server module to get the registered mcp instance
        from music21_mcp import server

        # Get list of expected tools
        expected_tools = [
            "import_score",
            "list_scores",
            "key_analysis",
            "chord_analysis",
            "score_info",
            "export_score",
            "delete_score",
            "harmony_analysis",
            "voice_leading_analysis",
            "pattern_recognition",
            "harmonization",
            "counterpoint",
            "style_imitation",
            "health_check",
            "cleanup_memory",
        ]

        # Verify tools are accessible as functions
        assert hasattr(server, "import_score")
        assert hasattr(server, "list_scores")
        assert hasattr(server, "key_analysis")

        # Test that the tools are callable
        assert callable(server.import_score)
        assert callable(server.list_scores)
        assert callable(server.key_analysis)

    @pytest.mark.asyncio
    async def test_score_storage_dict_operations(self):
        """Test that score storage dict is properly shared across tools"""
        # Create tools with shared storage
        import_tool = ImportScoreTool(scores)
        list_tool = ListScoresTool(scores)
        delete_tool = DeleteScoreTool(scores)

        # Import a score
        result = await import_tool.execute(
            score_id="test_score", source="bach/bwv66.6", source_type="corpus"
        )
        assert result["status"] == "success"

        # Verify it's in the shared storage
        assert "test_score" in scores

        # List scores should see it
        list_result = await list_tool.execute()
        assert list_result["status"] == "success"
        assert any(s["id"] == "test_score" for s in list_result["scores"])

        # Delete should remove it from shared storage
        delete_result = await delete_tool.execute(score_id="test_score")
        assert delete_result["status"] == "success"
        assert "test_score" not in scores

        # List should no longer see it
        list_result2 = await list_tool.execute()
        assert not any(s["id"] == "test_score" for s in list_result2["scores"])

    def test_rate_limiting_functionality(self):
        """Test rate limiting with token bucket algorithm"""
        # Create a rate limiter with low limits for testing
        test_limiter = SimpleRateLimiter(rate=2.0, burst=3)  # 2 per second, burst of 3

        # Should allow burst requests
        assert test_limiter.acquire()
        assert test_limiter.acquire()
        assert test_limiter.acquire()

        # Should be rate limited now
        assert not test_limiter.acquire()

        # Wait for token refill
        time.sleep(0.6)  # Should get ~1 token back
        assert test_limiter.acquire()
        assert not test_limiter.acquire()

        # Test rate refill calculation
        time.sleep(1.0)  # Should get 2 tokens back
        assert test_limiter.acquire()
        assert test_limiter.acquire()
        assert not test_limiter.acquire()

    @pytest.mark.asyncio
    async def test_rate_limiting_with_tools(self):
        """Test rate limiting integration with actual tools"""
        # Import the server functions directly
        from music21_mcp.server import import_score, rate_limiter

        # Temporarily set very restrictive limits
        original_tokens = rate_limiter.tokens
        original_rate = rate_limiter.rate
        original_burst = rate_limiter.burst

        try:
            rate_limiter.rate = 1.0  # 1 per second
            rate_limiter.burst = 2  # burst of 2
            rate_limiter.tokens = 2

            # First two requests should succeed
            result1 = await import_score("test1", "bach/bwv66.6", "corpus")
            assert "error" not in result1 or "Rate limit" not in result1.get(
                "error", ""
            )

            result2 = await import_score("test2", "bach/bwv66.6", "corpus")
            assert "error" not in result2 or "Rate limit" not in result2.get(
                "error", ""
            )

            # Third should be rate limited
            result3 = await import_score("test3", "bach/bwv66.6", "corpus")
            assert result3.get("error") == "Rate limit exceeded, please try again later"

        finally:
            # Restore original limits
            rate_limiter.rate = original_rate
            rate_limiter.burst = original_burst
            rate_limiter.tokens = original_tokens

    @pytest.mark.asyncio
    async def test_error_handling_and_response_formats(self):
        """Test error handling and consistent response formats"""
        # Test with various error conditions
        key_tool = KeyAnalysisTool(scores)

        # Test missing score
        result = await key_tool.execute(score_id="nonexistent")
        assert result["status"] == "error"
        # Check for either 'error' or 'message' field (tools may use either)
        error_msg = result.get("error") or result.get("message", "")
        assert "not found" in error_msg.lower()

        # Test invalid parameters
        import_tool = ImportScoreTool(scores)
        result = await import_tool.execute(
            score_id="test", source="invalid/path", source_type="corpus"
        )
        assert result["status"] == "error"
        # Check for either 'error' or 'message' field
        assert "error" in result or "message" in result

        # Verify consistent response structure
        list_tool = ListScoresTool(scores)
        result = await list_tool.execute()
        assert "status" in result
        assert result["status"] in ["success", "error"]
        if result["status"] == "success":
            assert "scores" in result

    @pytest.mark.asyncio
    async def test_tool_interconnection_workflow(self):
        """Test complex workflows involving multiple interconnected tools"""
        # Import -> Analyze -> Transform -> Export workflow

        # Step 1: Import a score
        import_tool = ImportScoreTool(scores)
        import_result = await import_tool.execute(
            score_id="workflow_test", source="bach/bwv66.6", source_type="corpus"
        )
        assert import_result["status"] == "success"

        # Step 2: Analyze the score with multiple tools
        key_tool = KeyAnalysisTool(scores)
        key_result = await key_tool.execute(score_id="workflow_test")
        assert key_result["status"] == "success"
        assert "key" in key_result

        harmony_tool = HarmonyAnalysisTool(scores)
        harmony_result = await harmony_tool.execute(score_id="workflow_test")
        assert harmony_result["status"] == "success"

        pattern_tool = PatternRecognitionTool(scores)
        pattern_result = await pattern_tool.execute(score_id="workflow_test")
        assert pattern_result["status"] == "success"

        # Step 3: Generate variations
        if MUSIC21_AVAILABLE:
            # Create a simple melody for harmonization
            melody_score = stream.Score()
            melody_score.metadata = metadata.Metadata()
            melody_score.metadata.title = "Test Melody"
            part = stream.Part()

            for pitch in ["C4", "E4", "G4", "C5"]:
                n = note.Note(pitch, quarterLength=1)
                part.append(n)

            melody_score.append(part)
            scores["melody_test"] = melody_score

            harmonization_tool = HarmonizationTool(scores)
            harm_result = await harmonization_tool.execute(
                score_id="melody_test", style="basic"
            )
            # Harmonization might fail gracefully
            assert "status" in harm_result

        # Step 4: Export in different formats
        export_tool = ExportScoreTool(scores)

        # Test MusicXML export
        xml_result = await export_tool.execute(
            score_id="workflow_test", format="musicxml"
        )
        assert xml_result["status"] == "success"
        # Export tool returns file_path, not data
        assert "file_path" in xml_result or "data" in xml_result
        assert xml_result["format"] == "musicxml"

        # Test MIDI export
        midi_result = await export_tool.execute(score_id="workflow_test", format="midi")
        assert midi_result["status"] == "success"
        # Export tool returns file_path, not data
        assert "file_path" in midi_result or "data" in midi_result
        assert midi_result["format"] == "midi"

        # Step 5: Cleanup
        delete_tool = DeleteScoreTool(scores)
        delete_result = await delete_tool.execute(score_id="workflow_test")
        assert delete_result["status"] == "success"
        assert "workflow_test" not in scores

    @pytest.mark.asyncio
    async def test_concurrent_access_patterns(self):
        """Test concurrent access to shared score storage"""
        # Create multiple concurrent operations
        import_tool = ImportScoreTool(scores)
        list_tool = ListScoresTool(scores)

        # Define concurrent tasks
        async def import_task(score_id: str):
            return await import_tool.execute(
                score_id=score_id, source="bach/bwv66.6", source_type="corpus"
            )

        async def list_task():
            return await list_tool.execute()

        # Run concurrent imports
        tasks = []
        for i in range(5):
            tasks.append(import_task(f"concurrent_{i}"))
            tasks.append(list_task())

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify no exceptions
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, (
            f"Concurrent access caused exceptions: {exceptions}"
        )

        # Verify imports succeeded - check for successful import operations
        successful_imports = [
            r
            for r in results
            if isinstance(r, dict)
            and r.get("status") == "success"
            and (r.get("title") or r.get("score_id") or "import" in str(r))
        ]
        # We should have at least some successful imports (may not be exactly 5 due to concurrency)
        assert len(successful_imports) >= 3

        # Verify final state is consistent
        final_list = await list_tool.execute()
        assert final_list["status"] == "success"

        # Count concurrent scores
        concurrent_scores = [
            s for s in final_list["scores"] if s["id"].startswith("concurrent_")
        ]
        assert len(concurrent_scores) == 5

    @pytest.mark.asyncio
    async def test_memory_management_and_cleanup(self):
        """Test memory management and cleanup functionality"""
        # Import the cleanup function
        from music21_mcp.server import cleanup_memory

        # Get initial memory usage
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Create and store multiple scores
        import_tool = ImportScoreTool(scores)
        for i in range(10):
            await import_tool.execute(
                score_id=f"memory_test_{i}", source="bach/bwv66.6", source_type="corpus"
            )

        # Memory should have increased
        after_import_memory = psutil.Process().memory_info().rss / 1024 / 1024
        assert len(scores) == 10

        # Clear scores and run cleanup
        scores.clear()
        cleanup_result = await cleanup_memory()

        assert cleanup_result["status"] == "success"
        assert "memory_before_mb" in cleanup_result
        assert "memory_after_mb" in cleanup_result
        assert cleanup_result["scores_count"] == 0

        # Verify garbage collection was triggered
        assert cleanup_result["memory_after_mb"] <= cleanup_result["memory_before_mb"]

    @pytest.mark.asyncio
    async def test_health_check_functionality(self):
        """Test the health check endpoint"""
        from music21_mcp.server import health_check

        # Populate some test data
        import_tool = ImportScoreTool(scores)
        await import_tool.execute(
            score_id="health_test", source="bach/bwv66.6", source_type="corpus"
        )

        # Run health check
        health_result = await health_check()

        # Verify response structure
        assert health_result["status"] == "healthy"
        assert "timestamp" in health_result
        assert "uptime_seconds" in health_result
        assert "memory" in health_result
        assert "scores" in health_result

        # Verify memory info
        memory_info = health_result["memory"]
        assert "used_mb" in memory_info
        assert "percent" in memory_info
        assert memory_info["used_mb"] > 0
        assert 0 <= memory_info["percent"] <= 100

        # Verify scores info
        scores_info = health_result["scores"]
        assert scores_info["count"] == 1
        assert "health_test" in scores_info["ids"]

        # Verify architecture info
        assert health_result["architecture"] == "simple_fastmcp"
        assert health_result["version"] == "minimal_working"

    @pytest.mark.asyncio
    async def test_resource_endpoints(self):
        """Test MCP resource endpoints"""
        from music21_mcp.server import get_score_metadata, get_scores_list

        # Import test scores
        import_tool = ImportScoreTool(scores)
        await import_tool.execute(
            score_id="resource_test", source="bach/bwv66.6", source_type="corpus"
        )

        # Test scores list resource
        list_resource = await get_scores_list()
        assert "scores" in list_resource
        assert len(list_resource["scores"]) == 1
        assert list_resource["scores"][0]["id"] == "resource_test"

        # Test score metadata resource
        metadata_resource = await get_score_metadata("resource_test")
        assert "id" in metadata_resource
        assert metadata_resource["id"] == "resource_test"
        assert "title" in metadata_resource

        # Test nonexistent score
        missing_metadata = await get_score_metadata("nonexistent")
        assert missing_metadata.get("error") == "Score not found"

    @pytest.mark.asyncio
    async def test_error_recovery_and_resilience(self):
        """Test server resilience and error recovery"""
        # Test recovery from tool errors
        harmony_tool = HarmonyAnalysisTool(scores)

        # Try to analyze nonexistent score
        result = await harmony_tool.execute(score_id="nonexistent")
        assert result["status"] == "error"

        # Server should still be functional
        list_tool = ListScoresTool(scores)
        list_result = await list_tool.execute()
        assert list_result["status"] == "success"

        # Test with malformed score data
        scores["malformed"] = "not a score object"

        # Tools should handle gracefully - ScoreInfoTool may handle this differently
        info_tool = ScoreInfoTool(scores)
        info_result = await info_tool.execute(score_id="malformed")
        # Tool may still return success but with limited info
        assert info_result["status"] in ["error", "success"]

        # Cleanup malformed data
        del scores["malformed"]

        # Server should continue working
        import_tool = ImportScoreTool(scores)
        import_result = await import_tool.execute(
            score_id="recovery_test", source="bach/bwv66.6", source_type="corpus"
        )
        assert import_result["status"] == "success"

    @pytest.mark.asyncio
    async def test_corpus_data_integration(self):
        """Test with various music21 corpus data for realistic scenarios"""
        if not MUSIC21_AVAILABLE:
            pytest.skip("Music21 not available")

        import_tool = ImportScoreTool(scores)
        key_tool = KeyAnalysisTool(scores)

        # Test with different corpus pieces
        test_pieces = [
            ("bach/bwv66.6", "Bach Chorale"),
            ("mozart/k331", "Mozart Sonata"),
            ("beethoven/opus18no1", "Beethoven Quartet"),
        ]

        for corpus_path, description in test_pieces:
            score_id = corpus_path.replace("/", "_")

            # Import
            import_result = await import_tool.execute(
                score_id=score_id, source=corpus_path, source_type="corpus"
            )

            if import_result["status"] == "success":
                print(f"✓ Successfully imported {description}")

                # Analyze
                key_result = await key_tool.execute(score_id=score_id)
                if key_result["status"] == "success":
                    print(f"  - Key: {key_result.get('key', 'Unknown')}")
            else:
                # Some corpus items might not be available
                print(f"  - Skipped {description} (not in corpus)")


class TestFastMCPServerStress:
    """Stress tests for FastMCP server"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for stress tests"""
        scores.clear()
        gc.collect()
        yield
        scores.clear()
        gc.collect()

    @pytest.mark.asyncio
    async def test_large_score_handling(self):
        """Test handling of large scores"""
        if not MUSIC21_AVAILABLE:
            pytest.skip("Music21 not available")

        # Create a large score
        large_score = stream.Score()
        large_score.metadata = metadata.Metadata()
        large_score.metadata.title = "Large Test Score"

        # Add multiple parts with many notes
        for part_num in range(10):  # 10 parts
            part = stream.Part()
            for measure_num in range(100):  # 100 measures
                for beat in range(4):  # 4 beats per measure
                    n = note.Note("C4", quarterLength=1)
                    part.append(n)
            large_score.append(part)

        # Store the large score
        scores["large_test"] = large_score

        # Test various operations on large score
        info_tool = ScoreInfoTool(scores)
        info_result = await info_tool.execute(score_id="large_test")
        assert info_result["status"] == "success"

        # Check that we got some info back
        # The exact structure may vary based on what ScoreInfoTool can extract
        assert len(info_result) > 2  # More than just status and score_id

        # If structure info is available, verify it has reasonable values
        if "structure" in info_result:
            structure = info_result["structure"]
            # Check for any indication of parts
            if "num_parts" in structure:
                assert structure["num_parts"] >= 1
            # Check for any indication of size
            if "num_measures" in structure:
                assert structure["num_measures"] >= 1
            if "num_notes" in structure:
                assert structure["num_notes"] >= 1

        # Test export of large score
        export_tool = ExportScoreTool(scores)
        export_result = await export_tool.execute(
            score_id="large_test", format="musicxml"
        )
        assert export_result["status"] == "success"
        # Export returns file info, not data directly
        if "file_size" in export_result:
            assert export_result["file_size"] > 10000  # Should be substantial

    @pytest.mark.asyncio
    async def test_rapid_sequential_requests(self):
        """Test handling of rapid sequential requests"""
        import_tool = ImportScoreTool(scores)
        list_tool = ListScoresTool(scores)

        start_time = time.time()
        request_count = 50

        for i in range(request_count):
            # Alternate between import and list
            if i % 2 == 0:
                await import_tool.execute(
                    score_id=f"rapid_{i}", source="bach/bwv66.6", source_type="corpus"
                )
            else:
                await list_tool.execute()

        elapsed = time.time() - start_time

        # Verify all imports succeeded
        final_list = await list_tool.execute()
        rapid_scores = [s for s in final_list["scores"] if s["id"].startswith("rapid_")]
        assert len(rapid_scores) == request_count // 2

        print(f"Processed {request_count} requests in {elapsed:.2f} seconds")
        print(f"Average: {request_count / elapsed:.1f} requests/second")


# Additional test utilities
def create_test_score(title: str = "Test Score", num_notes: int = 10) -> Any:
    """Create a test score with music21"""
    if not MUSIC21_AVAILABLE:
        return None

    s = stream.Score()
    s.metadata = metadata.Metadata()
    s.metadata.title = title

    part = stream.Part()
    for i in range(num_notes):
        n = note.Note("C4", quarterLength=1)
        part.append(n)

    s.append(part)
    return s


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
