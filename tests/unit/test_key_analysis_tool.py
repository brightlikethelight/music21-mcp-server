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
        assert "key" in result
        assert "confidence" in result
        assert "alternatives" in result
        assert isinstance(result["key"], str)
        assert isinstance(result["confidence"], (int, float))
        assert isinstance(result["alternatives"], list)
        # Key should be in format "<tonic> major" or "<tonic> minor"
        assert " major" in result["key"] or " minor" in result["key"]
    
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
        assert "confidence" in result
        assert isinstance(result["confidence"], (int, float))
        assert 0 <= result["confidence"] <= 1
    
    @pytest.mark.asyncio
    async def test_analyze_key_alternative_keys(self, populated_score_storage):
        """Test key analysis provides alternative key suggestions"""
        tool = KeyAnalysisTool(populated_score_storage)
        
        result = await tool.execute(score_id="bach_test")
        
        assert result["status"] == "success"
        assert "alternatives" in result
        assert isinstance(result["alternatives"], list)
        
        # Check alternative structure if any exist
        if len(result["alternatives"]) > 0:
            alt = result["alternatives"][0]
            assert "key" in alt
            assert "confidence" in alt
            assert isinstance(alt["key"], str)
            assert isinstance(alt["confidence"], (int, float))
    
    @pytest.mark.asyncio
    async def test_analyze_key_algorithm_results(self, populated_score_storage):
        """Test key analysis includes algorithm results when using 'all'"""
        tool = KeyAnalysisTool(populated_score_storage)
        
        result = await tool.execute(score_id="bach_test", algorithm="all")
        
        assert result["status"] == "success"
        assert "algorithm_results" in result
        assert isinstance(result["algorithm_results"], dict)
        
        # Check that we have results from multiple algorithms
        expected_algorithms = ["krumhansl", "aarden", "temperley", "bellman"]
        for alg in expected_algorithms:
            if alg in result["algorithm_results"]:
                alg_result = result["algorithm_results"][alg]
                assert "key" in alg_result
                assert "confidence" in alg_result
                assert isinstance(alg_result["key"], str)
                assert isinstance(alg_result["confidence"], (int, float))
    
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
            assert "analysis failed" in result["message"].lower() or "empty" in result["message"].lower() or "not found" in result["message"].lower()