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
        assert "voice_leading_analysis" in result
        analysis = result["voice_leading_analysis"]
        assert "voice_movements" in analysis
        assert "parallel_intervals" in analysis
        assert "voice_crossings" in analysis
        assert "smooth_voice_leading_score" in analysis
    
    @pytest.mark.asyncio
    async def test_voice_leading_nonexistent_score(self, clean_score_storage):
        """Test voice leading analysis with non-existent score"""
        tool = VoiceLeadingAnalysisTool(clean_score_storage)
        
        result = await tool.execute(score_id="nonexistent")
        
        assert result["status"] == "error"
        assert "not found" in result["message"]
    
    @pytest.mark.asyncio
    async def test_voice_leading_movements(self, populated_score_storage):
        """Test voice movement detection"""
        tool = VoiceLeadingAnalysisTool(populated_score_storage)
        
        result = await tool.execute(score_id="bach_test")
        
        assert result["status"] == "success"
        movements = result["voice_leading_analysis"]["voice_movements"]
        assert isinstance(movements, list)
        
        if len(movements) > 0:
            movement = movements[0]
            assert "voice" in movement
            assert "from_pitch" in movement
            assert "to_pitch" in movement
            assert "interval" in movement
            assert "measure" in movement
    
    @pytest.mark.asyncio
    async def test_voice_leading_parallel_detection(self, populated_score_storage):
        """Test parallel interval detection"""
        tool = VoiceLeadingAnalysisTool(populated_score_storage)
        
        result = await tool.execute(score_id="bach_test")
        
        assert result["status"] == "success"
        parallels = result["voice_leading_analysis"]["parallel_intervals"]
        assert isinstance(parallels, list)
        
        if len(parallels) > 0:
            parallel = parallels[0]
            assert "interval_type" in parallel
            assert "voices" in parallel
            assert "measure" in parallel
            assert parallel["interval_type"] in ["fifths", "octaves"]
    
    @pytest.mark.asyncio
    async def test_voice_leading_crossing_detection(self, populated_score_storage):
        """Test voice crossing detection"""
        tool = VoiceLeadingAnalysisTool(populated_score_storage)
        
        result = await tool.execute(score_id="bach_test")
        
        assert result["status"] == "success"
        crossings = result["voice_leading_analysis"]["voice_crossings"]
        assert isinstance(crossings, list)
        
        if len(crossings) > 0:
            crossing = crossings[0]
            assert "voice1" in crossing
            assert "voice2" in crossing
            assert "measure" in crossing
            assert "beat" in crossing
    
    @pytest.mark.asyncio
    async def test_voice_leading_smoothness_score(self, populated_score_storage):
        """Test voice leading smoothness scoring"""
        tool = VoiceLeadingAnalysisTool(populated_score_storage)
        
        result = await tool.execute(score_id="bach_test")
        
        assert result["status"] == "success"
        score = result["voice_leading_analysis"]["smooth_voice_leading_score"]
        assert isinstance(score, (int, float))
        assert 0 <= score <= 100  # Assuming percentage score
    
    @pytest.mark.asyncio
    async def test_voice_leading_custom_parameters(self, populated_score_storage):
        """Test voice leading with custom parameters"""
        tool = VoiceLeadingAnalysisTool(populated_score_storage)
        
        result = await tool.execute(
            score_id="bach_test",
            check_parallel_fifths=True,
            check_parallel_octaves=True,
            check_voice_crossings=True,
            check_hidden_intervals=True
        )
        
        assert result["status"] == "success"
        assert "voice_leading_analysis" in result
    
    @pytest.mark.asyncio
    async def test_voice_leading_monophonic(self, clean_score_storage):
        """Test voice leading on monophonic music"""
        from music21 import stream, note
        
        # Create monophonic melody
        melody = stream.Stream()
        for pitch in ['C4', 'D4', 'E4', 'F4', 'G4']:
            melody.append(note.Note(pitch, quarterLength=1))
        
        clean_score_storage["melody"] = melody
        
        tool = VoiceLeadingAnalysisTool(clean_score_storage)
        result = await tool.execute(score_id="melody")
        
        # Should handle monophonic gracefully
        assert result["status"] in ["success", "error"]
        if result["status"] == "success":
            # Monophonic should have no voice leading issues
            analysis = result["voice_leading_analysis"]
            assert len(analysis["parallel_intervals"]) == 0
            assert len(analysis["voice_crossings"]) == 0
    
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
        analysis = result["voice_leading_analysis"]
        # Should detect voice movements
        assert len(analysis["voice_movements"]) > 0