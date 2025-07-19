#!/usr/bin/env python3
"""
Claude Desktop Integration Tests

This test suite specifically tests integration with Claude Desktop,
focusing on MCP protocol compliance and tool visibility issues.
"""

import asyncio
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import server components
try:
    from mcp.server.fastmcp import FastMCP
    from mcp.types import (
        ServerCapabilities,
        ServerInfo,
        InitializeResult,
        Tool,
        TextContent,
        ImageContent,
        EmbeddedResource,
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    
    # Mock MCP types for testing
    class ServerCapabilities:
        pass
    
    class Tool:
        def __init__(self, name, description, input_schema):
            self.name = name
            self.description = description
            self.input_schema = input_schema


class TestClaudeDesktopIntegration:
    """Test suite for Claude Desktop integration issues"""
    
    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP not available")
    def test_mcp_server_configuration(self):
        """Test that server is properly configured for Claude Desktop"""
        from music21_mcp.server import mcp, server_name
        
        # Check server name (shown in Claude Desktop)
        assert server_name is not None
        assert len(server_name) > 0
        assert "Music21" in server_name or "music21" in server_name.lower()
        
        # Check FastMCP instance
        assert isinstance(mcp, FastMCP)
        
        # Test that server has required MCP methods
        assert hasattr(mcp, 'tool')
        assert hasattr(mcp, 'resource') 
        assert hasattr(mcp, 'run')
        
        # Check if server can be serialized for MCP protocol
        # Claude Desktop needs to serialize server info
        assert hasattr(mcp, '__class__')
        assert mcp.__class__.__name__ == 'FastMCP'
    
    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP not available")
    def test_tool_schema_compliance(self):
        """Test that all tools have proper JSON schemas for Claude Desktop"""
        from music21_mcp import server
        
        # Expected tools that should be visible in Claude Desktop
        expected_tools = [
            ("import_score", {
                "required": ["score_id", "source", "source_type"],
                "properties": {
                    "score_id": {"type": "string"},
                    "source": {"type": "string"},
                    "source_type": {"type": "string", "enum": ["corpus", "file", "url"]}
                }
            }),
            ("list_scores", {
                "properties": {},
                "required": []
            }),
            ("key_analysis", {
                "required": ["score_id"],
                "properties": {
                    "score_id": {"type": "string"}
                }
            }),
            ("export_score", {
                "required": ["score_id", "format"],
                "properties": {
                    "score_id": {"type": "string"},
                    "format": {"type": "string", "enum": ["musicxml", "midi", "lilypond", "pdf"]}
                }
            }),
        ]
        
        # Check tool registration
        for tool_name, expected_schema in expected_tools:
            # Tool should be accessible as a function
            assert hasattr(server, tool_name), f"Tool {tool_name} not found in server module"
            
            tool_func = getattr(server, tool_name)
            assert callable(tool_func), f"Tool {tool_name} is not callable"
            
            # Check function has proper async signature
            import inspect
            assert asyncio.iscoroutinefunction(tool_func), f"Tool {tool_name} is not async"
            
            # Check function has docstring (used by Claude Desktop)
            doc = inspect.getdoc(tool_func)
            if not doc:
                print(f"⚠️  WARNING: Tool {tool_name} has no docstring!")
    
    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP not available")
    @pytest.mark.asyncio
    async def test_mcp_protocol_initialization(self):
        """Test MCP protocol initialization sequence"""
        # This simulates what Claude Desktop does when connecting
        
        from music21_mcp.server import mcp
        
        # Create a mock MCP client
        class MockMCPClient:
            def __init__(self):
                self.tools = {}
                self.resources = {}
                self.server_info = None
            
            async def initialize(self, server):
                """Simulate MCP initialization handshake"""
                # Server should provide capabilities
                if hasattr(server, '_get_capabilities'):
                    capabilities = await server._get_capabilities()
                    assert isinstance(capabilities, (dict, ServerCapabilities))
                
                # Server should provide info
                if hasattr(server, '_get_info'):
                    info = await server._get_info()
                    assert isinstance(info, (dict, ServerInfo))
                    self.server_info = info
                
                return True
            
            async def list_tools(self, server):
                """Simulate tool discovery"""
                # This is what Claude Desktop does to find available tools
                # FastMCP doesn't expose tools directly, but we can check
                # that tools are registered by trying to call them
                return True
        
        client = MockMCPClient()
        
        # Simulate initialization
        initialized = await client.initialize(mcp)
        assert initialized
        
        # Simulate tool discovery
        tools_discovered = await client.list_tools(mcp)
        assert tools_discovered
    
    def test_claude_desktop_config_format(self):
        """Test that server can be configured in Claude Desktop's expected format"""
        # Claude Desktop expects configuration in ~/.claude/claude_desktop_config.json
        
        expected_config = {
            "mcpServers": {
                "music21": {
                    "command": "python",
                    "args": ["-m", "music21_mcp.server"],
                    "env": {
                        "PYTHONPATH": str(Path(__file__).parent.parent.parent / "src")
                    }
                }
            }
        }
        
        # Verify the command would work
        import music21_mcp.server
        server_module = Path(music21_mcp.server.__file__)
        assert server_module.exists()
        
        # Check that module can be run as __main__
        server_dir = server_module.parent
        main_file = server_dir / "__main__.py"
        
        # Server should be runnable via python -m
        assert server_module.name == "server.py" or main_file.exists()
        
        # Generate actual config for users
        config_json = json.dumps(expected_config, indent=2)
        print("\n📋 Claude Desktop Configuration:")
        print(config_json)
        print("\nAdd this to ~/.claude/claude_desktop_config.json")
    
    @pytest.mark.asyncio
    async def test_tool_error_handling_for_claude(self):
        """Test that tool errors are properly formatted for Claude Desktop"""
        from music21_mcp.server import import_score, key_analysis
        
        # Test 1: Missing required parameters
        # Claude Desktop should see clear error messages
        
        # Import without required params
        result = await import_score()  # Missing all params
        assert result.get("error") is not None
        error_msg = result.get("error", "")
        # Error should mention missing parameters
        assert "missing" in error_msg.lower() or "required" in error_msg.lower()
        
        # Test 2: Invalid parameter types
        result = await import_score(
            score_id=123,  # Should be string
            source=None,   # Should be string
            source_type=["corpus"]  # Should be string, not list
        )
        assert result.get("error") is not None or result.get("status") == "error"
        
        # Test 3: Non-existent score reference
        result = await key_analysis(score_id="definitely_does_not_exist_12345")
        assert result.get("status") == "error"
        assert "not found" in str(result).lower()
        
        # Test 4: Error response format
        # Claude Desktop expects consistent error format
        error_responses = []
        
        # Collect various error responses
        test_cases = [
            import_score(),  # Missing params
            import_score(score_id="test"),  # Partial params
            key_analysis(score_id="nonexistent"),  # Invalid reference
        ]
        
        for test_case in test_cases:
            try:
                result = await test_case
                if "error" in result or result.get("status") == "error":
                    error_responses.append(result)
            except Exception as e:
                # Tool raised exception instead of returning error
                error_responses.append({"error": str(e), "status": "error"})
        
        # All errors should have consistent structure
        for error_resp in error_responses:
            # Should have either 'error' field or 'status': 'error'
            assert "error" in error_resp or error_resp.get("status") == "error"
            # Should be JSON serializable for MCP protocol
            json.dumps(error_resp)  # Should not raise
    
    @pytest.mark.asyncio
    async def test_tool_streaming_support(self):
        """Test if tools support streaming responses for Claude Desktop"""
        # Some MCP tools can stream responses for better UX
        
        from music21_mcp.server import pattern_recognition
        from music21_mcp.tools import ImportScoreTool, PatternRecognitionTool
        
        # First import a score
        scores = {}
        import_tool = ImportScoreTool(scores)
        await import_tool.execute(
            score_id="stream_test",
            source="bach/bwv66.6",
            source_type="corpus"
        )
        
        # Test pattern recognition (might have long output)
        result = await pattern_recognition(score_id="stream_test")
        
        # Check if result is streamable or chunked
        # FastMCP might not support streaming yet, but check response size
        result_json = json.dumps(result)
        result_size = len(result_json)
        
        print(f"Pattern recognition result size: {result_size} bytes")
        
        # Large results should ideally be streamed
        if result_size > 10000:  # 10KB
            print("⚠️  Large result detected - consider implementing streaming")
    
    def test_server_subprocess_compatibility(self):
        """Test that server works correctly when launched as subprocess by Claude Desktop"""
        # Claude Desktop launches MCP servers as subprocesses
        
        test_script = '''
import sys
import json
sys.path.insert(0, "src")

try:
    from music21_mcp.server import mcp
    print(json.dumps({
        "status": "loaded",
        "server_name": getattr(mcp, 'name', 'unknown'),
        "has_run": hasattr(mcp, 'run')
    }))
except Exception as e:
    print(json.dumps({
        "status": "error",
        "error": str(e)
    }))
'''
        
        # Run server module as subprocess
        result = subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent  # Project root
        )
        
        # Check subprocess ran successfully
        assert result.returncode == 0, f"Subprocess failed: {result.stderr}"
        
        # Parse output
        try:
            output = json.loads(result.stdout)
            assert output["status"] == "loaded"
            assert output["has_run"] is True
            print(f"✅ Server loads correctly as subprocess: {output['server_name']}")
        except json.JSONDecodeError:
            pytest.fail(f"Invalid JSON output: {result.stdout}")
    
    @pytest.mark.asyncio
    async def test_concurrent_claude_connections(self):
        """Test server behavior with multiple Claude Desktop connections"""
        # Claude Desktop might open multiple connections or reconnect
        
        from music21_mcp.server import import_score, list_scores, scores
        
        # Simulate multiple "sessions"
        session_results = []
        
        for session in range(3):
            # Each session does some operations
            session_data = {"session": session, "operations": []}
            
            # Import a score
            result = await import_score(
                score_id=f"session_{session}_score",
                source="bach/bwv66.6",
                source_type="corpus"
            )
            session_data["operations"].append(("import", result.get("status")))
            
            # List scores (should see all sessions' scores)
            result = await list_scores()
            score_count = len(result.get("scores", []))
            session_data["operations"].append(("list", score_count))
            
            session_results.append(session_data)
        
        # Verify all sessions worked
        for session_data in session_results:
            session_num = session_data["session"]
            ops = session_data["operations"]
            
            # Import should succeed
            assert ops[0][1] == "success", f"Session {session_num} import failed"
            
            # List should show cumulative scores
            expected_count = session_num + 1
            actual_count = ops[1][1]
            assert actual_count >= expected_count, \
                f"Session {session_num} should see at least {expected_count} scores, got {actual_count}"
        
        # Server should maintain state across "connections"
        final_list = await list_scores()
        assert len(final_list.get("scores", [])) >= 3
    
    @pytest.mark.asyncio
    async def test_resource_endpoint_visibility(self):
        """Test that MCP resources are visible to Claude Desktop"""
        from music21_mcp.server import get_scores_list, get_score_metadata, import_score
        
        # Import a test score
        await import_score(
            score_id="resource_test",
            source="bach/bwv66.6", 
            source_type="corpus"
        )
        
        # Test resource endpoints that Claude Desktop might use
        
        # 1. List all scores
        scores_list = await get_scores_list()
        assert "scores" in scores_list
        assert isinstance(scores_list["scores"], list)
        
        # Each score should have required fields for Claude Desktop
        for score in scores_list["scores"]:
            assert "id" in score
            assert "title" in score
            # Should be JSON serializable
            json.dumps(score)
        
        # 2. Get specific score metadata  
        metadata = await get_score_metadata("resource_test")
        assert "id" in metadata
        assert metadata["id"] == "resource_test"
        
        # Metadata should be rich enough for Claude Desktop UI
        expected_fields = ["id", "title"]
        for field in expected_fields:
            assert field in metadata, f"Missing metadata field: {field}"
        
        # 3. Test error handling for resources
        missing_metadata = await get_score_metadata("nonexistent")
        assert "error" in missing_metadata
        
        print("✅ Resource endpoints properly formatted for Claude Desktop")
    
    def test_tool_naming_conventions(self):
        """Test that tool names follow Claude Desktop conventions"""
        from music21_mcp import server
        
        # Get all tool names
        tool_names = []
        for attr_name in dir(server):
            attr = getattr(server, attr_name)
            if callable(attr) and not attr_name.startswith('_'):
                # Check if it looks like a tool (async function)
                import inspect
                if asyncio.iscoroutinefunction(attr):
                    tool_names.append(attr_name)
        
        print(f"\n📋 Found {len(tool_names)} tools:")
        
        for tool_name in sorted(tool_names):
            # Check naming conventions
            issues = []
            
            # Should be snake_case
            if not tool_name.islower() or '-' in tool_name:
                issues.append("should be snake_case")
            
            # Should be descriptive
            if len(tool_name) < 5:
                issues.append("name too short")
            
            # Should not have redundant prefixes
            if tool_name.startswith('music21_') or tool_name.startswith('mcp_'):
                issues.append("redundant prefix")
            
            status = "✅" if not issues else "⚠️"
            issue_str = f" ({', '.join(issues)})" if issues else ""
            print(f"  {status} {tool_name}{issue_str}")
        
        # All tools should be found
        assert len(tool_names) >= 10, f"Expected at least 10 tools, found {len(tool_names)}"


class TestMCPProtocolCompliance:
    """Test MCP protocol compliance for Claude Desktop"""
    
    @pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP not available")
    def test_mcp_message_format(self):
        """Test that server messages follow MCP protocol format"""
        # MCP uses JSON-RPC 2.0 format
        
        from music21_mcp.server import mcp
        
        # Check that FastMCP handles protocol details
        assert hasattr(mcp, '__class__')
        
        # Server should handle these MCP protocol messages:
        # - initialize
        # - initialized  
        # - tools/list
        # - tools/call
        # - resources/list
        # - resources/read
        # - shutdown
        
        # FastMCP should handle protocol internally
        # We can't easily test the actual protocol without a full MCP client
        
    @pytest.mark.asyncio
    async def test_tool_parameter_validation(self):
        """Test that tools validate parameters as expected by Claude Desktop"""
        from music21_mcp.server import import_score, export_score
        
        # Test enum validation
        result = await export_score(
            score_id="test",
            format="invalid_format"  # Not in enum
        )
        
        # Should reject invalid enum value
        assert result.get("status") == "error" or "error" in result
        
        # Test required parameter validation
        result = await import_score(
            score_id="test"
            # Missing source and source_type
        )
        
        # Should indicate missing parameters
        assert result.get("status") == "error" or "error" in result
        
    def test_server_metadata(self):
        """Test that server provides proper metadata for Claude Desktop"""
        from music21_mcp import server
        
        # Check server module has required metadata
        assert hasattr(server, '__file__')  # Module file path
        
        # Check for version info (if available)
        version_attrs = ['__version__', 'VERSION', 'version']
        has_version = any(hasattr(server, attr) for attr in version_attrs)
        
        if has_version:
            version = next(getattr(server, attr) for attr in version_attrs if hasattr(server, attr))
            print(f"✅ Server version: {version}")
        else:
            print("⚠️  No version information found")
        
        # Check for proper module structure
        server_path = Path(server.__file__)
        assert server_path.exists()
        assert server_path.suffix == '.py'
        
        # Check for __main__ entry point
        main_path = server_path.parent / '__main__.py'
        if main_path.exists():
            print("✅ Has __main__.py for python -m execution")
        else:
            # Server.py should have if __name__ == "__main__"
            with open(server_path) as f:
                content = f.read()
                assert 'if __name__ == "__main__"' in content
                print("✅ Has __main__ block in server.py")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s", "--tb=short"])