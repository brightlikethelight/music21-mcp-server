"""
Unit tests for KeyAnalysisTool
"""
import pytest
from music21_mcp.tools.key_analysis_tool import KeyAnalysisTool


class TestKeyAnalysisTool:
    """Test KeyAnalysisTool functionality"""
    
    def test_tool_initialization(self, clean_score_storage):
        """Test tool can be initialized with score storage"""
        tool = KeyAnalysisTool(clean_score_storage)
        assert tool.scores == clean_score_storage
    
    @pytest.mark.asyncio
    async def test_analyze_key_success(self, populated_score_storage):
        """Test successful key analysis"""
        tool = KeyAnalysisTool(populated_score_storage)
        
        result = await tool.execute(score_id="bach_test")
        
        assert result["status"] == "success"
        assert "key_analysis" in result
        assert "tonic" in result["key_analysis"]
        assert "mode" in result["key_analysis"]
    
    @pytest.mark.asyncio
    async def test_analyze_nonexistent_score(self, clean_score_storage):
        """Test key analysis with non-existent score"""
        tool = KeyAnalysisTool(clean_score_storage)
        
        result = await tool.execute(score_id="nonexistent")
        
        assert result["status"] == "error"
        assert "not found" in result["message"]
    
    @pytest.mark.asyncio
    async def test_analyze_key_with_confidence(self, populated_score_storage):
        """Test key analysis includes confidence score"""
        tool = KeyAnalysisTool(populated_score_storage)
        
        result = await tool.execute(score_id="bach_test")
        
        assert result["status"] == "success"
        key_analysis = result["key_analysis"]
        assert "confidence" in key_analysis
        assert isinstance(key_analysis["confidence"], (int, float))
        assert 0 <= key_analysis["confidence"] <= 1
    
    @pytest.mark.asyncio
    async def test_analyze_key_alternative_keys(self, populated_score_storage):
        """Test key analysis provides alternative key suggestions"""
        tool = KeyAnalysisTool(populated_score_storage)
        
        result = await tool.execute(score_id="bach_test")
        
        assert result["status"] == "success"
        key_analysis = result["key_analysis"]
        assert "alternative_keys" in key_analysis
        assert isinstance(key_analysis["alternative_keys"], list)
    
    @pytest.mark.asyncio
    async def test_analyze_key_metadata(self, populated_score_storage):
        """Test key analysis includes metadata"""
        tool = KeyAnalysisTool(populated_score_storage)
        
        result = await tool.execute(score_id="bach_test")
        
        assert result["status"] == "success"
        key_analysis = result["key_analysis"]
        assert "tonic" in key_analysis
        assert "mode" in key_analysis
        assert key_analysis["mode"] in ["major", "minor"]
    
    @pytest.mark.asyncio
    async def test_key_analysis_handles_empty_score(self, clean_score_storage):
        """Test key analysis handles empty or invalid scores gracefully"""
        from music21 import stream
        
        empty_score = stream.Stream()
        clean_score_storage["empty_score"] = empty_score
        
        tool = KeyAnalysisTool(clean_score_storage)
        result = await tool.execute(score_id="empty_score")
        
        # Should handle gracefully, either succeed with default or fail informatively
        assert result["status"] in ["success", "error"]
        if result["status"] == "error":
            assert "analysis failed" in result["message"].lower() or "empty" in result["message"].lower()