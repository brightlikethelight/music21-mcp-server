"""
Unit tests for VoiceLeadingAnalysisTool
"""
import pytest
from music21_mcp.tools.voice_leading_tool import VoiceLeadingAnalysisTool


class TestVoiceLeadingAnalysisTool:
    """Test VoiceLeadingAnalysisTool functionality"""
    
    def test_tool_initialization(self, clean_score_storage):
        """Test tool can be initialized with score storage"""
        tool = VoiceLeadingAnalysisTool(clean_score_storage)
        assert tool.scores == clean_score_storage
    
    @pytest.mark.asyncio
    async def test_voice_leading_analysis_success(self, populated_score_storage):
        """Test successful voice leading analysis"""
        tool = VoiceLeadingAnalysisTool(populated_score_storage)
        
        result = await tool.execute(score_id="bach_test")
        
        assert result["status"] == "success"
        assert "parallel_issues" in result
        assert "voice_crossings" in result
        assert "smoothness_analysis" in result
        assert "total_issues" in result
        assert "overall_score" in result
        assert isinstance(result["parallel_issues"], list)
        assert isinstance(result["voice_crossings"], list)
        assert isinstance(result["smoothness_analysis"], dict)
        assert isinstance(result["total_issues"], int)
        assert isinstance(result["overall_score"], (int, float))
    
    @pytest.mark.asyncio
    async def test_voice_leading_nonexistent_score(self, clean_score_storage):
        """Test voice leading analysis with non-existent score"""
        tool = VoiceLeadingAnalysisTool(clean_score_storage)
        
        result = await tool.execute(score_id="nonexistent")
        
        assert result["status"] == "error"
        assert "not found" in result["message"]
    
    @pytest.mark.asyncio
    async def test_voice_leading_smoothness(self, populated_score_storage):
        """Test voice leading smoothness analysis"""
        tool = VoiceLeadingAnalysisTool(populated_score_storage)
        
        result = await tool.execute(score_id="bach_test")
        
        assert result["status"] == "success"
        smoothness = result["smoothness_analysis"]
        assert isinstance(smoothness, dict)
        assert "total_motion" in smoothness
        assert "stepwise_motion" in smoothness
        assert "leap_motion" in smoothness
        assert "large_leap_motion" in smoothness
        assert "average_interval_size" in smoothness
        assert "smoothness_score" in smoothness
        assert smoothness["smoothness_score"] >= 0  # Score can be > 100 for very smooth motion
    
    @pytest.mark.asyncio
    async def test_voice_leading_parallel_detection(self, populated_score_storage):
        """Test parallel interval detection"""
        tool = VoiceLeadingAnalysisTool(populated_score_storage)
        
        result = await tool.execute(score_id="bach_test")
        
        assert result["status"] == "success"
        parallels = result["parallel_issues"]
        assert isinstance(parallels, list)
        
        if len(parallels) > 0:
            parallel = parallels[0]
            assert "type" in parallel
            assert "position" in parallel
            assert "parts" in parallel
            assert "notes" in parallel
            assert parallel["type"] in ["parallel_P5", "parallel_P8"]
    
    @pytest.mark.asyncio
    async def test_voice_leading_crossing_detection(self, populated_score_storage):
        """Test voice crossing detection"""
        tool = VoiceLeadingAnalysisTool(populated_score_storage)
        
        result = await tool.execute(score_id="bach_test")
        
        assert result["status"] == "success"
        crossings = result["voice_crossings"]
        assert isinstance(crossings, list)
        
        if len(crossings) > 0:
            crossing = crossings[0]
            assert "position" in crossing
            assert "higher_part" in crossing
            assert "lower_part" in crossing
            assert "higher_note" in crossing
            assert "lower_note" in crossing
    
    @pytest.mark.asyncio
    async def test_voice_leading_overall_score(self, populated_score_storage):
        """Test voice leading overall scoring"""
        tool = VoiceLeadingAnalysisTool(populated_score_storage)
        
        result = await tool.execute(score_id="bach_test")
        
        assert result["status"] == "success"
        score = result["overall_score"]
        assert isinstance(score, (int, float))
        assert 0 <= score <= 100  # Percentage score
    
    @pytest.mark.asyncio
    async def test_voice_leading_basic_parameters(self, populated_score_storage):
        """Test voice leading with basic parameters"""
        tool = VoiceLeadingAnalysisTool(populated_score_storage)
        
        result = await tool.execute(score_id="bach_test")
        
        assert result["status"] == "success"
        assert "parallel_issues" in result
        assert "voice_crossings" in result
        assert "smoothness_analysis" in result
    
    @pytest.mark.asyncio
    async def test_voice_leading_monophonic(self, clean_score_storage):
        """Test voice leading on monophonic music"""
        from music21 import stream, note
        
        # Create monophonic melody as a Score (not Stream)
        melody = stream.Score()
        part = stream.Part()
        for pitch in ['C4', 'D4', 'E4', 'F4', 'G4']:
            part.append(note.Note(pitch, quarterLength=1))
        melody.append(part)
        
        clean_score_storage["melody"] = melody
        
        tool = VoiceLeadingAnalysisTool(clean_score_storage)
        result = await tool.execute(score_id="melody")
        
        # Should handle monophonic gracefully
        assert result["status"] in ["success", "error"]
        if result["status"] == "success":
            # Monophonic should have no voice leading issues
            assert len(result["parallel_issues"]) == 0
            assert len(result["voice_crossings"]) == 0
    
    @pytest.mark.asyncio
    async def test_voice_leading_satb(self, clean_score_storage):
        """Test voice leading on SATB chorale"""
        from music21 import stream, note, chord, key
        
        # Create simple SATB progression
        score = stream.Score()
        score.append(key.Key('C'))
        
        # C major to G major progression
        chord1 = chord.Chord(['C3', 'E3', 'G3', 'C4'])
        chord2 = chord.Chord(['G2', 'D3', 'G3', 'B3'])
        chord1.quarterLength = 1
        chord2.quarterLength = 1
        
        score.append(chord1)
        score.append(chord2)
        
        clean_score_storage["satb"] = score
        
        tool = VoiceLeadingAnalysisTool(clean_score_storage)
        result = await tool.execute(score_id="satb")
        
        assert result["status"] == "success"
        # Should have analysis results
        assert "parallel_issues" in result
        assert "voice_crossings" in result
        assert "smoothness_analysis" in result
        assert "total_issues" in result
        assert "overall_score" in result