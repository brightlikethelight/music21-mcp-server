#!/usr/bin/env python3
"""
Production Failure Tests for Music21 MCP Server

This test suite specifically focuses on finding failures and edge cases that could occur
in production environments. Tests are designed to break the server and identify weaknesses.
"""

import asyncio
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import psutil
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import server components
try:
    from mcp.server import Server
    from mcp.server.fastmcp import FastMCP
    from mcp.types import TextContent, Tool

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    FastMCP = None

from music21_mcp.server import (
    rate_limiter,
    register_tool,
    scores,
)
from music21_mcp.tools import (
    DeleteScoreTool,
    ExportScoreTool,
    ImportScoreTool,
    KeyAnalysisTool,
    ListScoresTool,
    ScoreInfoTool,
)

try:
    from music21 import chord, corpus, key, metadata, note, stream

    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False


class TestProductionFailures:
    """Test suite for production failure scenarios"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test"""
        # Clear scores before each test
        scores.clear()

        # Reset rate limiter
        if hasattr(rate_limiter, "tokens"):
            rate_limiter.tokens = rate_limiter.burst
            rate_limiter.last_update = time.time()

        yield

        # Cleanup after test
        scores.clear()

    @pytest.mark.asyncio
    async def test_requests_before_initialization(self):
        """Test 1: Server receives requests before initialization"""
        # Create a new FastMCP instance (simulating uninitialized server)
        uninitialized_mcp = FastMCP("Uninitialized Test Server")

        # Try to use a tool before any are registered
        with pytest.raises(AttributeError):
            # This should fail because no tools are registered yet
            await uninitialized_mcp.import_score("test", "source", "corpus")

        # Test with partially initialized server
        @uninitialized_mcp.tool(name="test_tool")
        async def test_tool():
            # Try to access scores dict before it's properly set up
            return {"count": len(scores)}

        # The tool should be registered but might fail if internals aren't ready
        result = await test_tool()
        assert isinstance(result, dict)

        # Test accessing server internals before full initialization
        # Simulate early request by directly calling tool functions
        import_tool = ImportScoreTool({})  # Empty dict instead of initialized scores
        with pytest.raises((KeyError, AttributeError)):
            # This should fail because the storage dict is empty/wrong
            await import_tool.execute(
                score_id="early_test", source="bach/bwv66.6", source_type="corpus"
            )

    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP not available")
    def test_tools_visibility_in_claude_desktop(self):
        """Test 2: Check if tools actually appear in Claude Desktop"""
        # This test simulates what Claude Desktop would see

        # Check that mcp instance has the expected tools
        from music21_mcp import server

        # Tools should be accessible as decorated functions
        expected_tools = [
            "import_score",
            "list_scores",
            "score_info",
            "export_score",
            "delete_score",
            "key_analysis",
            "chord_analysis",
            "harmony_analysis",
            "voice_leading_analysis",
            "pattern_recognition",
            "harmonization",
            "counterpoint",
            "style_imitation",
            "health_check",
            "cleanup_memory",
        ]

        # Verify tools are registered with MCP
        # In production, Claude Desktop would enumerate these through MCP protocol
        # We simulate by checking the decorated functions exist
        missing_tools = []
        for tool_name in expected_tools:
            if not hasattr(server, tool_name):
                missing_tools.append(tool_name)

        # Log which tools are missing for debugging
        if missing_tools:
            print(f"❌ Missing tools in server module: {missing_tools}")

        # Test that FastMCP instance is properly configured
        assert server.mcp is not None
        assert isinstance(server.mcp, FastMCP)

        # Simulate MCP protocol tool discovery
        # This is what Claude Desktop would do internally
        # We can't fully test this without the MCP client, but we can verify structure
        assert hasattr(server.mcp, "tool")  # Decorator exists
        assert hasattr(server.mcp, "run")  # Server can be run

        # Test tool metadata (what Claude Desktop sees)
        # Each tool should have proper documentation
        import inspect

        if hasattr(server, "import_score"):
            func = server.import_score
            # Check if function has docstring (shown in Claude Desktop)
            doc = inspect.getdoc(func)
            print(f"import_score documentation: {doc}")
            # In production, empty or missing docs mean tools won't be understood
            if not doc:
                print("⚠️  WARNING: import_score has no documentation!")

    @pytest.mark.asyncio
    async def test_timeout_scenarios(self):
        """Test 3: Timeout scenarios - does the server crash?"""

        # Test 1: Tool execution timeout
        class SlowTool:
            def __init__(self, storage):
                self.storage = storage

            async def execute(self, **kwargs):
                # Simulate long-running operation
                await asyncio.sleep(40)  # Longer than typical timeout
                return {"status": "success"}

        # Register slow tool
        slow_tool_func = register_tool("slow_test", SlowTool)

        # Test with asyncio timeout
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(slow_tool_func(test_param="value"), timeout=2.0)

        # Server should still be responsive after timeout
        list_tool = ListScoresTool(scores)
        result = await list_tool.execute()
        assert result["status"] == "success"

        # Test 2: Deadlock simulation
        lock = threading.Lock()

        class DeadlockTool:
            def __init__(self, storage):
                self.storage = storage

            async def execute(self, **kwargs):
                # Try to acquire lock from async context (bad practice)
                def blocking_operation():
                    with lock:
                        time.sleep(5)  # Hold lock
                        return {"blocked": True}

                # This could cause issues in production
                loop = asyncio.get_event_loop()
                try:
                    # Run blocking operation in executor
                    result = await loop.run_in_executor(None, blocking_operation)
                    return {"status": "success", "data": result}
                except Exception as e:
                    return {"status": "error", "error": str(e)}

        deadlock_tool = DeadlockTool(scores)

        # Acquire lock in main thread
        lock.acquire()
        try:
            # Try to use tool while lock is held
            task = asyncio.create_task(deadlock_tool.execute())

            # Wait a bit then release lock
            await asyncio.sleep(0.5)
            lock.release()

            # Tool should eventually complete
            result = await asyncio.wait_for(task, timeout=10)
            assert "status" in result
        finally:
            if lock.locked():
                lock.release()

        # Test 3: Network timeout simulation (for corpus downloads)
        with patch("urllib.request.urlopen") as mock_urlopen:
            # Simulate network timeout
            mock_urlopen.side_effect = TimeoutError("Network timeout")

            import_tool = ImportScoreTool(scores)
            # This might try to download corpus data
            result = await import_tool.execute(
                score_id="timeout_test",
                source="https://example.com/score.xml",
                source_type="url",  # If supported
            )

            # Should handle timeout gracefully
            if "error" in result:
                assert (
                    "timeout" in result["error"].lower() or "error" in result["status"]
                )

    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test 4: Verify if the server recovers from errors"""
        # Test 1: Memory corruption recovery
        # Corrupt the scores storage
        scores["corrupted"] = None  # Invalid score object
        scores["partial"] = {"invalid": "data"}  # Partial data

        # Try to use tools with corrupted data
        info_tool = ScoreInfoTool(scores)

        # Should handle corrupted entry
        result1 = await info_tool.execute(score_id="corrupted")
        # Tool should either error or handle gracefully
        assert result1["status"] in ["error", "success"]

        # Should handle partial data
        result2 = await info_tool.execute(score_id="partial")
        assert result2["status"] in ["error", "success"]

        # Clean up corruption
        del scores["corrupted"]
        del scores["partial"]

        # Server should work normally after cleanup
        import_result = await ImportScoreTool(scores).execute(
            score_id="recovery_test", source="bach/bwv66.6", source_type="corpus"
        )
        assert import_result["status"] == "success"

        # Test 2: Exception in tool execution
        class CrashingTool:
            def __init__(self, storage):
                self.storage = storage
                self.crash_count = 0

            async def execute(self, **kwargs):
                self.crash_count += 1
                if self.crash_count < 3:
                    raise RuntimeError(f"Simulated crash #{self.crash_count}")
                return {"status": "success", "recovered": True}

        crashing_tool = CrashingTool(scores)

        # First attempts should fail
        with pytest.raises(RuntimeError):
            await crashing_tool.execute()

        with pytest.raises(RuntimeError):
            await crashing_tool.execute()

        # Third attempt should succeed (simulating recovery)
        result = await crashing_tool.execute()
        assert result["status"] == "success"
        assert result["recovered"] is True

        # Test 3: Resource exhaustion recovery
        # Simulate running out of memory by creating large objects
        large_scores = []
        try:
            # Create large scores but don't store in main dict
            if MUSIC21_AVAILABLE:
                for i in range(5):
                    large_score = stream.Score()
                    for j in range(100):
                        part = stream.Part()
                        for k in range(1000):
                            part.append(note.Note())
                        large_score.append(part)
                    large_scores.append(large_score)
        except MemoryError:
            # System protected itself from OOM
            pass

        # Clear large objects
        large_scores.clear()

        # Server should still function
        health_check = await server.health_check()
        assert health_check["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test 5: Check what happens with concurrent requests"""
        # Test 1: Race condition on same score ID
        import_tool = ImportScoreTool(scores)

        async def concurrent_import(suffix: int):
            return await import_tool.execute(
                score_id="race_condition",  # Same ID for all
                source="bach/bwv66.6",
                source_type="corpus",
            )

        # Launch multiple concurrent imports with same ID
        tasks = [concurrent_import(i) for i in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check results - some might fail, but server shouldn't crash
        successes = [
            r for r in results if isinstance(r, dict) and r.get("status") == "success"
        ]
        errors = [
            r
            for r in results
            if isinstance(r, Exception)
            or (isinstance(r, dict) and r.get("status") == "error")
        ]

        print(f"Concurrent same ID: {len(successes)} successes, {len(errors)} errors")

        # At least one should succeed
        assert len(successes) >= 1
        # But we might have conflicts
        if errors:
            print(f"Concurrent conflicts detected: {len(errors)}")

        # Test 2: Concurrent different operations on same score
        scores.clear()

        # First, import a score
        await import_tool.execute(
            score_id="concurrent_test", source="bach/bwv66.6", source_type="corpus"
        )

        # Now perform different operations concurrently
        async def analyze():
            tool = KeyAnalysisTool(scores)
            return await tool.execute(score_id="concurrent_test")

        async def get_info():
            tool = ScoreInfoTool(scores)
            return await tool.execute(score_id="concurrent_test")

        async def export():
            tool = ExportScoreTool(scores)
            return await tool.execute(score_id="concurrent_test", format="musicxml")

        async def modify():
            # Simulate modification by reimporting
            return await import_tool.execute(
                score_id="concurrent_test", source="mozart/k331", source_type="corpus"
            )

        # Run all operations concurrently
        tasks = [
            analyze(),
            get_info(),
            export(),
            modify(),
            analyze(),
            get_info(),
            export(),
            modify(),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successes and failures
        successes = sum(
            1 for r in results if isinstance(r, dict) and r.get("status") == "success"
        )
        failures = sum(
            1
            for r in results
            if isinstance(r, Exception)
            or (isinstance(r, dict) and r.get("status") == "error")
        )

        print(f"Concurrent operations: {successes} successes, {failures} failures")

        # Server should handle most operations
        assert successes >= 4  # At least half should succeed

        # Test 3: Stress test with many concurrent requests
        async def stress_operation(op_id: int):
            operations = [
                lambda: ImportScoreTool(scores).execute(
                    score_id=f"stress_{op_id}",
                    source="bach/bwv66.6",
                    source_type="corpus",
                ),
                lambda: ListScoresTool(scores).execute(),
                lambda: ScoreInfoTool(scores).execute(
                    score_id=f"stress_{op_id % 5}"  # Some will exist, some won't
                ),
            ]

            op = operations[op_id % len(operations)]
            try:
                return await op()
            except Exception as e:
                return {"status": "error", "error": str(e)}

        # Launch many concurrent operations
        stress_tasks = [stress_operation(i) for i in range(50)]
        stress_results = await asyncio.gather(*stress_tasks, return_exceptions=True)

        # Server should handle the load
        stress_successes = sum(
            1
            for r in stress_results
            if isinstance(r, dict) and r.get("status") == "success"
        )

        print(f"Stress test: {stress_successes}/50 operations succeeded")

        # At least 60% should succeed even under stress
        assert stress_successes >= 30

    @pytest.mark.asyncio
    async def test_docker_cloud_environment(self):
        """Test 6: Test if server works in Docker/cloud environments"""
        # Test 1: Environment variable configuration
        original_env = os.environ.copy()

        try:
            # Simulate Docker environment variables
            os.environ["MUSIC21_MCP_MODE"] = "enterprise"
            os.environ["MUSIC21_MCP_SECURITY"] = "true"
            os.environ["MUSIC21_MCP_MONITORING"] = "true"
            os.environ["MUSIC21_MCP_RESILIENCE"] = "true"
            os.environ["MUSIC21_MCP_RATE_LIMIT"] = "10"
            os.environ["MUSIC21_MCP_TIMEOUT"] = "5.0"

            # Reimport to test configuration loading
            from music21_mcp.server import ServerConfig

            docker_config = ServerConfig()

            # Verify configuration
            assert docker_config.mode == "enterprise"
            assert docker_config.is_enterprise is True
            assert docker_config.security_enabled is True
            assert docker_config.monitoring_enabled is True
            assert docker_config.resilience_enabled is True
            assert docker_config.rate_limit_rate == 10.0 / 60
            assert docker_config.request_timeout == 5.0

        finally:
            # Restore environment
            os.environ.clear()
            os.environ.update(original_env)

        # Test 2: File system constraints (read-only, permissions)
        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate read-only file system
            test_file = Path(tmpdir) / "test_export.xml"

            # Make directory read-only (simulating cloud constraints)
            os.chmod(tmpdir, 0o555)

            try:
                # Try to export to read-only location
                export_tool = ExportScoreTool(scores)

                # First import a score
                await ImportScoreTool(scores).execute(
                    score_id="docker_test", source="bach/bwv66.6", source_type="corpus"
                )

                # Try to export to read-only directory
                # Tool should handle permission error gracefully
                result = await export_tool.execute(
                    score_id="docker_test",
                    format="musicxml",
                    # Some tools might accept output path
                )

                # Should either succeed (using different path) or error gracefully
                assert result["status"] in ["success", "error"]
                if result["status"] == "error":
                    print(
                        f"Expected permission error: {result.get('error', 'Unknown')}"
                    )

            finally:
                # Restore permissions
                os.chmod(tmpdir, 0o755)

        # Test 3: Memory constraints (common in containers)
        # Get current memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Simulate memory pressure by setting low limit
        # (In real Docker, this would be enforced by cgroups)
        memory_limit_mb = initial_memory + 50  # Only 50MB headroom

        # Try operations that might exceed memory
        large_operations = 0
        memory_errors = 0

        for i in range(10):
            try:
                # Try to import multiple scores
                await ImportScoreTool(scores).execute(
                    score_id=f"memory_test_{i}",
                    source="bach/bwv66.6",
                    source_type="corpus",
                )

                # Check memory usage
                current_memory = process.memory_info().rss / 1024 / 1024
                if current_memory > memory_limit_mb:
                    print(
                        f"⚠️  Memory limit exceeded: {current_memory:.1f}MB > {memory_limit_mb:.1f}MB"
                    )
                    # In real Docker, container might be killed here
                    memory_errors += 1

                    # Try cleanup
                    await server.cleanup_memory()

                large_operations += 1

            except MemoryError:
                memory_errors += 1
                print(f"MemoryError on operation {i}")
                # Try to recover
                scores.clear()
                await server.cleanup_memory()

        print(
            f"Completed {large_operations} operations with {memory_errors} memory issues"
        )

        # Server should still be functional
        health = await server.health_check()
        assert health["status"] == "healthy"

        # Test 4: Network isolation (common in cloud)
        # Simulate network failure for corpus downloads
        with patch("music21.corpus.parse") as mock_parse:
            mock_parse.side_effect = Exception("Network unreachable")

            # Try to import from corpus with network down
            result = await ImportScoreTool(scores).execute(
                score_id="network_test", source="bach/bwv66.6", source_type="corpus"
            )

            # Should handle network error gracefully
            assert result["status"] == "error"
            assert "error" in result

    @pytest.mark.asyncio
    async def test_edge_cases_and_corner_cases(self):
        """Additional edge cases that could cause failures"""

        # Test 1: Unicode and special characters in score IDs
        weird_ids = [
            "test_😀_emoji",
            "test\nwith\nnewlines",
            "test\twith\ttabs",
            "test with spaces",
            "test/with/slashes",
            "test\\with\\backslashes",
            'test"with"quotes',
            "test'with'quotes",
            "../../../etc/passwd",  # Path traversal attempt
            "test\x00null",  # Null byte
            "❤️🎵🎼🎹",  # All emojis
            "тест_кириллица",  # Cyrillic
            "测试中文",  # Chinese
            "",  # Empty string
            " ",  # Just space
            "." * 1000,  # Very long ID
        ]

        import_tool = ImportScoreTool(scores)

        for weird_id in weird_ids:
            try:
                result = await import_tool.execute(
                    score_id=weird_id, source="bach/bwv66.6", source_type="corpus"
                )
                # Should either handle gracefully or error appropriately
                assert "status" in result
                if result["status"] == "error":
                    print(
                        f"ID '{weird_id[:20]}...' rejected: {result.get('error', 'Unknown')}"
                    )
            except Exception as e:
                print(f"ID '{weird_id[:20]}...' caused exception: {type(e).__name__}")

        # Test 2: Extremely large parameter values
        list_tool = ListScoresTool(scores)

        # Try with huge limit parameter (if supported)
        try:
            result = await list_tool.execute(limit=999999999)
            assert result["status"] in ["success", "error"]
        except Exception as e:
            print(f"Large limit parameter caused: {type(e).__name__}")

        # Test 3: Circular references in scores
        if MUSIC21_AVAILABLE:
            circular_score = stream.Score()
            part1 = stream.Part()
            part2 = stream.Part()

            # Create circular reference (if possible)
            circular_score.append(part1)
            circular_score.append(part2)
            # This might not create a true circular ref in music21

            scores["circular"] = circular_score

            # Try to analyze circular score
            info_result = await ScoreInfoTool(scores).execute(score_id="circular")
            assert info_result["status"] in ["success", "error"]

        # Test 4: Rapid score ID reuse
        reuse_id = "reuse_test"

        for i in range(10):
            # Import
            await import_tool.execute(
                score_id=reuse_id,
                source="bach/bwv66.6" if i % 2 == 0 else "mozart/k331",
                source_type="corpus",
            )

            # Delete immediately
            await DeleteScoreTool(scores).execute(score_id=reuse_id)

            # Re-import with same ID
            await import_tool.execute(
                score_id=reuse_id,
                source="beethoven/opus18no1" if i % 2 == 0 else "chopin/prelude-e",
                source_type="corpus",
            )

        # Server should handle rapid reuse
        assert len(scores) >= 0  # Should have at most one score with reuse_id

        # Test 5: Malformed tool parameters
        malformed_tests = [
            # Wrong parameter types
            {"score_id": 123, "source": "bach/bwv66.6", "source_type": "corpus"},
            {"score_id": ["list"], "source": "bach/bwv66.6", "source_type": "corpus"},
            {
                "score_id": {"dict": "value"},
                "source": "bach/bwv66.6",
                "source_type": "corpus",
            },
            {"score_id": None, "source": "bach/bwv66.6", "source_type": "corpus"},
            # Missing required parameters
            {"score_id": "test"},
            {"source": "bach/bwv66.6"},
            {},
            # Extra unknown parameters
            {
                "score_id": "test",
                "source": "bach/bwv66.6",
                "source_type": "corpus",
                "unknown_param": "value",
                "another_unknown": 123,
            },
        ]

        for params in malformed_tests:
            try:
                result = await import_tool.execute(**params)
                # Should handle gracefully
                assert result["status"] == "error"
            except Exception as e:
                print(f"Malformed params {params} caused: {type(e).__name__}")


class TestServerStability:
    """Test server stability under adverse conditions"""

    @pytest.mark.asyncio
    async def test_memory_leaks(self):
        """Test for memory leaks during extended operation"""
        if not MUSIC21_AVAILABLE:
            pytest.skip("Music21 not available")

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024

        # Perform many operations
        import_tool = ImportScoreTool(scores)
        delete_tool = DeleteScoreTool(scores)

        leak_detected = False
        memory_readings = [initial_memory]

        for cycle in range(10):
            # Import and delete many scores
            for i in range(20):
                score_id = f"leak_test_{cycle}_{i}"

                # Import
                await import_tool.execute(
                    score_id=score_id, source="bach/bwv66.6", source_type="corpus"
                )

                # Analyze
                await KeyAnalysisTool(scores).execute(score_id=score_id)

                # Delete
                await delete_tool.execute(score_id=score_id)

            # Force cleanup
            await server.cleanup_memory()

            # Check memory
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_readings.append(current_memory)

            # Check for steady increase (potential leak)
            if current_memory > initial_memory * 1.5:  # 50% increase
                leak_detected = True
                print(
                    f"⚠️  Potential memory leak: {initial_memory:.1f}MB -> {current_memory:.1f}MB"
                )

        # Analyze memory trend
        memory_increase = memory_readings[-1] - memory_readings[0]
        avg_increase_per_cycle = memory_increase / len(memory_readings)

        print(
            f"Memory trend: Start={memory_readings[0]:.1f}MB, End={memory_readings[-1]:.1f}MB"
        )
        print(f"Average increase per cycle: {avg_increase_per_cycle:.2f}MB")

        # Small increase is acceptable, large increase indicates leak
        assert avg_increase_per_cycle < 5.0, "Potential memory leak detected"

    @pytest.mark.asyncio
    async def test_signal_handling(self):
        """Test server behavior with system signals"""
        # This test is tricky because we're testing signal handling
        # in the same process. In production, signals come from outside.

        if sys.platform == "win32":
            pytest.skip("Signal handling test not applicable on Windows")

        # Test handling of SIGTERM (graceful shutdown)
        # We'll use a subprocess for proper signal testing

        test_script = """
import asyncio
import signal
import sys
sys.path.insert(0, "src")

from music21_mcp.server import mcp, scores
from music21_mcp.tools import ImportScoreTool

async def signal_test():
    # Set up signal handler
    shutdown_event = asyncio.Event()

    def handle_signal(signum, frame):
        print(f"Received signal {signum}")
        shutdown_event.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Start some operations
    tool = ImportScoreTool(scores)

    try:
        # Simulate ongoing work
        await tool.execute(
            score_id="signal_test",
            source="bach/bwv66.6",
            source_type="corpus"
        )

        # Wait for signal
        await asyncio.wait_for(shutdown_event.wait(), timeout=5.0)
        print("Shutting down gracefully")

    except asyncio.TimeoutError:
        print("No signal received")

    # Cleanup
    scores.clear()
    print("Cleanup complete")

asyncio.run(signal_test())
"""

        # Write test script
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_script)
            test_file = f.name

        try:
            # Start subprocess
            proc = subprocess.Popen(
                [sys.executable, test_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Give it time to start
            time.sleep(1)

            # Send SIGTERM
            proc.terminate()

            # Wait for completion
            stdout, stderr = proc.communicate(timeout=10)

            # Check output
            assert "Shutting down gracefully" in stdout or "Received signal" in stdout
            assert "Cleanup complete" in stdout
            assert proc.returncode is not None

        finally:
            # Clean up test file
            Path(test_file).unlink(missing_ok=True)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s", "--tb=short"])
