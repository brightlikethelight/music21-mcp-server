"""
FastMCP Server Integration Tests
Tests the actual FastMCP server functionality with all MCP tools
"""
import pytest
from unittest.mock import Mock, patch
import asyncio
import sys
from pathlib import Path
import json

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from music21_mcp.server import scores, mcp
from music21 import stream, note, chord, key


class MockMCPClient:
    """Mock MCP client for testing server interactions"""
    
    def __init__(self, server):
        self.server = server
        self.request_id = 0
    
    def next_id(self):
        self.request_id += 1
        return self.request_id
    
    async def call_tool(self, tool_name, arguments=None):
        """Simulate calling a tool via MCP protocol"""
        if arguments is None:
            arguments = {}
        
        # Find the tool function in the server
        tool_func = None
        for tool in self.server.tools:
            if tool.name == tool_name:
                tool_func = tool.func
                break
        
        if tool_func is None:
            return {"error": f"Tool {tool_name} not found"}
        
        try:
            result = await tool_func(**arguments)
            return result
        except Exception as e:
            return {"error": str(e)}


class TestFastMCPIntegration:
    """Integration tests for FastMCP server"""
    
    def setup_method(self):
        """Setup for each test"""
        # Clear scores before each test
        scores.clear()
        
        # Create mock client
        self.client = MockMCPClient(mcp)
    
    def teardown_method(self):
        """Cleanup after each test"""
        scores.clear()
    
    @pytest.mark.asyncio
    async def test_server_initialization(self):
        """Test that server initializes correctly"""
        assert mcp is not None
        assert mcp.name == "Music21 MCP Server"
        assert len(mcp.tools) == 13  # All tools should be registered
    
    @pytest.mark.asyncio
    async def test_tool_registration(self):
        """Test that all tools are properly registered"""
        tool_names = [tool.name for tool in mcp.tools]
        
        expected_tools = [
            "import_score",
            "list_scores", 
            "analyze_key",
            "analyze_chords",
            "get_score_info",
            "export_score",
            "delete_score",
            "analyze_harmony",
            "analyze_voice_leading",
            "recognize_patterns",
            "generate_harmonization",
            "generate_counterpoint",
            "imitate_style"
        ]
        
        for tool in expected_tools:
            assert tool in tool_names
    
    @pytest.mark.asyncio
    async def test_score_storage_management(self):
        """Test that score storage works correctly"""
        # Import a test score
        result = await self.client.call_tool("import_score", {
            "score_id": "test_score",
            "source": "corpus:bach/bwv66.6",
            "source_type": "corpus"
        })
        
        assert result["status"] == "success"
        assert "test_score" in scores
        assert scores["test_score"] is not None
    
    @pytest.mark.asyncio
    async def test_basic_workflow(self):
        """Test basic workflow: import -> analyze -> export"""
        # Step 1: Import score
        import_result = await self.client.call_tool("import_score", {
            "score_id": "workflow_test",
            "source": "corpus:bach/bwv66.6",
            "source_type": "corpus"
        })
        assert import_result["status"] == "success"
        
        # Step 2: Analyze key
        key_result = await self.client.call_tool("analyze_key", {
            "score_id": "workflow_test",
            "algorithm": "krumhansl"
        })
        assert key_result["status"] == "success"
        assert "key" in key_result
        
        # Step 3: Get score info
        info_result = await self.client.call_tool("get_score_info", {
            "score_id": "workflow_test"
        })
        assert info_result["status"] == "success"
        assert "metadata" in info_result
        
        # Step 4: Export score
        export_result = await self.client.call_tool("export_score", {
            "score_id": "workflow_test",
            "format": "xml"
        })
        assert export_result["status"] == "success"
        assert export_result["format"] == "xml"
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling for invalid operations"""
        # Test with non-existent score
        result = await self.client.call_tool("analyze_key", {
            "score_id": "nonexistent",
            "algorithm": "krumhansl"
        })
        assert result["status"] == "error"
        assert "not found" in result["message"].lower()
        
        # Test with invalid parameters
        result = await self.client.call_tool("analyze_key", {
            "score_id": "test",
            "algorithm": "invalid_algorithm"
        })
        assert result["status"] == "error"
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent tool executions"""
        # Import a test score first
        await self.client.call_tool("import_score", {
            "score_id": "concurrent_test",
            "source": "corpus:bach/bwv66.6",
            "source_type": "corpus"
        })
        
        # Run multiple analyses concurrently
        tasks = [
            self.client.call_tool("analyze_key", {
                "score_id": "concurrent_test",
                "algorithm": "krumhansl"
            }),
            self.client.call_tool("get_score_info", {
                "score_id": "concurrent_test"
            }),
            self.client.call_tool("analyze_harmony", {
                "score_id": "concurrent_test"
            })
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(r["status"] == "success" for r in results)
    
    @pytest.mark.asyncio
    async def test_memory_management(self):
        """Test memory management with multiple scores"""
        # Import multiple scores
        for i in range(5):
            result = await self.client.call_tool("import_score", {
                "score_id": f"memory_test_{i}",
                "source": "corpus:bach/bwv66.6",
                "source_type": "corpus"
            })
            assert result["status"] == "success"
        
        # Check all scores are stored
        assert len(scores) == 5
        
        # Delete some scores
        for i in range(3):
            result = await self.client.call_tool("delete_score", {
                "score_id": f"memory_test_{i}"
            })
            assert result["status"] == "success"
        
        # Check remaining scores
        assert len(scores) == 2
        assert "memory_test_3" in scores
        assert "memory_test_4" in scores
    
    @pytest.mark.asyncio
    async def test_pattern_recognition_workflow(self):
        """Test pattern recognition workflow"""
        # Create a simple score with patterns
        s = stream.Stream()
        s.append(key.Key('C'))
        
        # Add melodic sequence
        for i in range(3):
            base_pitch = 60 + i * 2  # C, D, E
            s.append(note.Note(midi=base_pitch, quarterLength=0.5))
            s.append(note.Note(midi=base_pitch + 2, quarterLength=0.5)) 
            s.append(note.Note(midi=base_pitch + 4, quarterLength=0.5))
        
        # Add to scores directly
        scores["pattern_test"] = s
        
        # Analyze patterns
        result = await self.client.call_tool("recognize_patterns", {
            "score_id": "pattern_test"
        })
        
        assert result["status"] == "success"
        assert "melodic_patterns" in result
    
    @pytest.mark.asyncio
    async def test_harmony_analysis_workflow(self):
        """Test harmony analysis workflow"""
        # Create chord progression
        s = stream.Stream()
        s.append(key.Key('C'))
        
        # I - IV - V - I progression
        chords = [
            chord.Chord(['C4', 'E4', 'G4'], quarterLength=1),
            chord.Chord(['F4', 'A4', 'C5'], quarterLength=1),
            chord.Chord(['G4', 'B4', 'D5'], quarterLength=1),
            chord.Chord(['C4', 'E4', 'G4'], quarterLength=1),
        ]
        
        for c in chords:
            s.append(c)
        
        scores["harmony_test"] = s
        
        # Analyze harmony
        result = await self.client.call_tool("analyze_harmony", {
            "score_id": "harmony_test"
        })
        
        assert result["status"] == "success"
        assert "roman_numerals" in result
    
    @pytest.mark.asyncio 
    async def test_tool_input_validation(self):
        """Test input validation for all tools"""
        # Test missing required parameters
        result = await self.client.call_tool("analyze_key", {})
        assert result["status"] == "error"
        
        # Test invalid score_id format
        result = await self.client.call_tool("analyze_key", {
            "score_id": "",
            "algorithm": "krumhansl"
        })
        assert result["status"] == "error"
    
    @pytest.mark.asyncio
    async def test_list_scores_functionality(self):
        """Test listing scores functionality"""
        # Initially empty
        result = await self.client.call_tool("list_scores")
        assert result["status"] == "success"
        assert len(result["scores"]) == 0
        
        # Add some scores
        for i in range(3):
            await self.client.call_tool("import_score", {
                "score_id": f"list_test_{i}",
                "source": "corpus:bach/bwv66.6",
                "source_type": "corpus"
            })
        
        # List scores
        result = await self.client.call_tool("list_scores")
        assert result["status"] == "success"
        assert len(result["scores"]) == 3
        assert all(f"list_test_{i}" in result["scores"] for i in range(3))
    
    @pytest.mark.asyncio
    async def test_export_formats(self):
        """Test different export formats"""
        # Import test score
        await self.client.call_tool("import_score", {
            "score_id": "export_test",
            "source": "corpus:bach/bwv66.6",
            "source_type": "corpus"
        })
        
        # Test different export formats
        formats = ["xml", "midi", "abc"]
        for fmt in formats:
            result = await self.client.call_tool("export_score", {
                "score_id": "export_test",
                "format": fmt
            })
            assert result["status"] == "success"
            assert result["format"] == fmt
            assert "data" in result
    
    @pytest.mark.asyncio
    async def test_advanced_analysis_tools(self):
        """Test advanced analysis tools"""
        # Import test score
        await self.client.call_tool("import_score", {
            "score_id": "advanced_test",
            "source": "corpus:bach/bwv66.6",
            "source_type": "corpus"
        })
        
        # Test voice leading analysis
        result = await self.client.call_tool("analyze_voice_leading", {
            "score_id": "advanced_test"
        })
        assert result["status"] == "success"
        
        # Test harmonization
        result = await self.client.call_tool("generate_harmonization", {
            "score_id": "advanced_test",
            "style": "bach"
        })
        assert result["status"] == "success"
        
        # Test counterpoint generation
        result = await self.client.call_tool("generate_counterpoint", {
            "score_id": "advanced_test",
            "species": "first"
        })
        assert result["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_style_imitation(self):
        """Test style imitation tool"""
        # Import test score
        await self.client.call_tool("import_score", {
            "score_id": "style_test",
            "source": "corpus:bach/bwv66.6",
            "source_type": "corpus"
        })
        
        # Test style imitation
        result = await self.client.call_tool("imitate_style", {
            "score_id": "style_test",
            "target_style": "bach"
        })
        assert result["status"] == "success"
        assert "generated_score" in result
    
    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test error recovery and graceful handling"""
        # Test with corrupted score data
        scores["corrupted"] = "not_a_score"
        
        result = await self.client.call_tool("analyze_key", {
            "score_id": "corrupted",
            "algorithm": "krumhansl"
        })
        
        # Should handle gracefully
        assert result["status"] == "error"
        assert "error" in result or "message" in result
    
    @pytest.mark.asyncio
    async def test_large_score_handling(self):
        """Test handling of large scores"""
        # Create a large score
        large_score = stream.Stream()
        large_score.append(key.Key('C'))
        
        # Add 1000 notes
        for i in range(1000):
            pitch = 60 + (i % 12)  # Chromatic scale
            large_score.append(note.Note(midi=pitch, quarterLength=0.25))
        
        scores["large_test"] = large_score
        
        # Test key analysis on large score
        result = await self.client.call_tool("analyze_key", {
            "score_id": "large_test",
            "algorithm": "temperley"
        })
        
        assert result["status"] == "success"
        assert "key" in result
    
    @pytest.mark.asyncio
    async def test_server_health_check(self):
        """Test server health and status"""
        # All tools should be available
        tool_names = [tool.name for tool in mcp.tools]
        assert len(tool_names) == 13
        
        # Score storage should be working
        assert isinstance(scores, dict)
        
        # Server should be properly initialized
        assert mcp.name == "Music21 MCP Server"