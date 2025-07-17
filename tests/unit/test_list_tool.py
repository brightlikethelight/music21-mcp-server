"""
Unit tests for ListScoresTool
"""
import pytest
from music21_mcp.tools.list_tool import ListScoresTool


class TestListScoresTool:
    """Test ListScoresTool functionality"""
    
    def test_tool_initialization(self, clean_score_storage):
        """Test tool can be initialized with score storage"""
        tool = ListScoresTool(clean_score_storage)
        assert tool.scores == clean_score_storage
    
    @pytest.mark.asyncio
    async def test_list_empty_storage(self, clean_score_storage):
        """Test listing scores from empty storage"""
        tool = ListScoresTool(clean_score_storage)
        
        result = await tool.execute()
        
        assert result["status"] == "success"
        assert "scores" in result
        assert result["scores"] == []
        assert "total_count" in result
        assert result["total_count"] == 0
    
    @pytest.mark.asyncio
    async def test_list_populated_storage(self, populated_score_storage):
        """Test listing scores from populated storage"""
        tool = ListScoresTool(populated_score_storage)
        
        result = await tool.execute()
        
        assert result["status"] == "success"
        assert "scores" in result
        assert "total_count" in result
        assert result["total_count"] > 0
        assert len(result["scores"]) == result["total_count"]
        
        # Check score entry structure
        if len(result["scores"]) > 0:
            score_entry = result["scores"][0]
            assert "id" in score_entry
            assert "title" in score_entry
            assert "parts" in score_entry
    
    @pytest.mark.asyncio
    async def test_list_with_multiple_scores(self, clean_score_storage, sample_bach_score):
        """Test listing multiple scores"""
        # Add multiple scores to storage
        clean_score_storage["score1"] = sample_bach_score
        clean_score_storage["score2"] = sample_bach_score
        clean_score_storage["score3"] = sample_bach_score
        
        tool = ListScoresTool(clean_score_storage)
        result = await tool.execute()
        
        assert result["status"] == "success"
        assert result["total_count"] == 3
        assert len(result["scores"]) == 3
        
        # Check all scores are listed
        score_ids = [score["id"] for score in result["scores"]]
        assert "score1" in score_ids
        assert "score2" in score_ids
        assert "score3" in score_ids
    
    @pytest.mark.asyncio
    async def test_list_score_metadata(self, populated_score_storage):
        """Test that score metadata is included in listing"""
        tool = ListScoresTool(populated_score_storage)
        
        result = await tool.execute()
        
        assert result["status"] == "success"
        if len(result["scores"]) > 0:
            score = result["scores"][0]
            assert "id" in score
            assert "title" in score
            assert "parts" in score
            assert isinstance(score["parts"], int)
            assert score["parts"] >= 0
    
    @pytest.mark.asyncio
    async def test_list_handles_invalid_scores(self, clean_score_storage):
        """Test listing handles invalid score objects gracefully"""
        # Add invalid score object
        clean_score_storage["invalid"] = "not a score"
        
        tool = ListScoresTool(clean_score_storage)
        result = await tool.execute()
        
        # Should handle gracefully and continue with other scores
        assert result["status"] == "success"
        assert "scores" in result
        assert "total_count" in result