"""
Unit tests for DeleteScoreTool
"""
import pytest
from music21_mcp.tools.delete_tool import DeleteScoreTool


class TestDeleteScoreTool:
    """Test DeleteScoreTool functionality"""
    
    def test_tool_initialization(self, clean_score_storage):
        """Test tool can be initialized with score storage"""
        tool = DeleteScoreTool(clean_score_storage)
        assert tool.scores == clean_score_storage
    
    @pytest.mark.asyncio
    async def test_delete_existing_score(self, populated_score_storage):
        """Test deleting an existing score"""
        tool = DeleteScoreTool(populated_score_storage)
        
        # Verify score exists before deletion
        assert "bach_test" in populated_score_storage
        
        result = await tool.execute(score_id="bach_test")
        
        assert result["status"] == "success"
        assert "deleted" in result["message"].lower()
        assert "bach_test" not in populated_score_storage
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_score(self, clean_score_storage):
        """Test deleting a non-existent score"""
        tool = DeleteScoreTool(clean_score_storage)
        
        result = await tool.execute(score_id="nonexistent")
        
        assert result["status"] == "error"
        assert "not found" in result["message"]
    
    @pytest.mark.asyncio
    async def test_delete_all_scores(self, clean_score_storage, sample_bach_score):
        """Test deleting all scores one by one"""
        # Add multiple scores
        clean_score_storage["score1"] = sample_bach_score
        clean_score_storage["score2"] = sample_bach_score
        clean_score_storage["score3"] = sample_bach_score
        
        tool = DeleteScoreTool(clean_score_storage)
        
        # Delete all scores
        result1 = await tool.execute(score_id="score1")
        result2 = await tool.execute(score_id="score2")
        result3 = await tool.execute(score_id="score3")
        
        assert result1["status"] == "success"
        assert result2["status"] == "success"
        assert result3["status"] == "success"
        assert len(clean_score_storage) == 0
    
    @pytest.mark.asyncio
    async def test_delete_score_case_sensitive(self, populated_score_storage):
        """Test that score deletion is case sensitive"""
        tool = DeleteScoreTool(populated_score_storage)
        
        # Try to delete with wrong case
        result = await tool.execute(score_id="BACH_TEST")
        
        assert result["status"] == "error"
        assert "not found" in result["message"]
        # Original score should still exist
        assert "bach_test" in populated_score_storage
    
    @pytest.mark.asyncio
    async def test_delete_score_empty_id(self, clean_score_storage):
        """Test deleting with empty score ID"""
        tool = DeleteScoreTool(clean_score_storage)
        
        result = await tool.execute(score_id="")
        
        assert result["status"] == "error"
        assert "not found" in result["message"]
    
    @pytest.mark.asyncio
    async def test_delete_score_maintains_other_scores(self, clean_score_storage, sample_bach_score):
        """Test that deleting one score doesn't affect others"""
        # Add multiple scores
        clean_score_storage["score1"] = sample_bach_score
        clean_score_storage["score2"] = sample_bach_score
        clean_score_storage["score3"] = sample_bach_score
        
        tool = DeleteScoreTool(clean_score_storage)
        
        # Delete one score
        result = await tool.execute(score_id="score2")
        
        assert result["status"] == "success"
        assert "score2" not in clean_score_storage
        assert "score1" in clean_score_storage
        assert "score3" in clean_score_storage
        assert len(clean_score_storage) == 2