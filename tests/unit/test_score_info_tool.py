"""
Unit tests for ScoreInfoTool
"""
import pytest
from music21_mcp.tools.score_info_tool import ScoreInfoTool


class TestScoreInfoTool:
    """Test ScoreInfoTool functionality"""
    
    def test_tool_initialization(self, clean_score_storage):
        """Test tool can be initialized with score storage"""
        tool = ScoreInfoTool(clean_score_storage)
        assert tool.scores == clean_score_storage
    
    @pytest.mark.asyncio
    async def test_get_score_info_success(self, populated_score_storage):
        """Test successful score info retrieval"""
        tool = ScoreInfoTool(populated_score_storage)
        
        result = await tool.execute(score_id="bach_test")
        
        assert result["status"] == "success"
        assert "score_info" in result
        info = result["score_info"]
        assert "title" in info
        assert "parts" in info
        assert "measures" in info
        assert "time_signature" in info
        assert "key_signature" in info
    
    @pytest.mark.asyncio
    async def test_get_score_info_nonexistent(self, clean_score_storage):
        """Test score info with non-existent score"""
        tool = ScoreInfoTool(clean_score_storage)
        
        result = await tool.execute(score_id="nonexistent")
        
        assert result["status"] == "error"
        assert "not found" in result["message"]
    
    @pytest.mark.asyncio
    async def test_score_info_structure(self, populated_score_storage):
        """Test that score info has expected structure"""
        tool = ScoreInfoTool(populated_score_storage)
        
        result = await tool.execute(score_id="bach_test")
        
        assert result["status"] == "success"
        info = result["score_info"]
        
        # Check basic metadata
        assert "title" in info
        assert "parts" in info
        assert "measures" in info
        assert isinstance(info["parts"], int)
        assert isinstance(info["measures"], int)
        assert info["parts"] >= 0
        assert info["measures"] >= 0
    
    @pytest.mark.asyncio
    async def test_score_info_time_signature(self, populated_score_storage):
        """Test time signature information"""
        tool = ScoreInfoTool(populated_score_storage)
        
        result = await tool.execute(score_id="bach_test")
        
        assert result["status"] == "success"
        info = result["score_info"]
        assert "time_signature" in info
        
        # Time signature should be a string like "4/4"
        time_sig = info["time_signature"]
        assert isinstance(time_sig, str)
        if time_sig != "Unknown":
            assert "/" in time_sig
    
    @pytest.mark.asyncio
    async def test_score_info_key_signature(self, populated_score_storage):
        """Test key signature information"""
        tool = ScoreInfoTool(populated_score_storage)
        
        result = await tool.execute(score_id="bach_test")
        
        assert result["status"] == "success"
        info = result["score_info"]
        assert "key_signature" in info
        
        # Key signature should be a string
        key_sig = info["key_signature"]
        assert isinstance(key_sig, str)
    
    @pytest.mark.asyncio
    async def test_score_info_duration(self, populated_score_storage):
        """Test duration information"""
        tool = ScoreInfoTool(populated_score_storage)
        
        result = await tool.execute(score_id="bach_test")
        
        assert result["status"] == "success"
        info = result["score_info"]
        
        # Duration info might be included
        if "duration" in info:
            assert isinstance(info["duration"], (int, float))
            assert info["duration"] >= 0
    
    @pytest.mark.asyncio
    async def test_score_info_handles_complex_score(self, clean_score_storage):
        """Test score info handles complex scores with multiple parts"""
        from music21 import stream, note, key, time, part
        
        # Create a complex score with multiple parts
        score = stream.Score()
        score.append(key.Key('C'))
        score.append(time.TimeSignature('4/4'))
        
        # Add multiple parts
        part1 = part.Part()
        part1.append(note.Note('C4', quarterLength=1))
        part1.append(note.Note('D4', quarterLength=1))
        
        part2 = part.Part()
        part2.append(note.Note('E4', quarterLength=1))
        part2.append(note.Note('F4', quarterLength=1))
        
        score.append(part1)
        score.append(part2)
        
        clean_score_storage["complex_score"] = score
        
        tool = ScoreInfoTool(clean_score_storage)
        result = await tool.execute(score_id="complex_score")
        
        assert result["status"] == "success"
        info = result["score_info"]
        assert info["parts"] == 2
        assert info["time_signature"] == "4/4"
        assert info["key_signature"] == "C major"
    
    @pytest.mark.asyncio
    async def test_score_info_handles_empty_score(self, clean_score_storage):
        """Test score info handles empty scores gracefully"""
        from music21 import stream
        
        empty_score = stream.Stream()
        clean_score_storage["empty_score"] = empty_score
        
        tool = ScoreInfoTool(clean_score_storage)
        result = await tool.execute(score_id="empty_score")
        
        # Should handle gracefully
        assert result["status"] in ["success", "error"]
        if result["status"] == "success":
            info = result["score_info"]
            assert info["parts"] == 0
            assert info["measures"] == 0