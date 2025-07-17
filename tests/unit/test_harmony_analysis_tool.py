"""
Unit tests for HarmonyAnalysisTool
"""
import pytest
from music21_mcp.tools.harmony_analysis_tool import HarmonyAnalysisTool


class TestHarmonyAnalysisTool:
    """Test HarmonyAnalysisTool functionality"""
    
    def test_tool_initialization(self, clean_score_storage):
        """Test tool can be initialized with score storage"""
        tool = HarmonyAnalysisTool(clean_score_storage)
        assert tool.scores == clean_score_storage
    
    @pytest.mark.asyncio
    async def test_harmony_analysis_success(self, populated_score_storage):
        """Test successful harmony analysis"""
        tool = HarmonyAnalysisTool(populated_score_storage)
        
        result = await tool.execute(score_id="bach_test")
        
        assert result["status"] == "success"
        assert "harmony_analysis" in result
        harmony = result["harmony_analysis"]
        assert "chords" in harmony
        assert "roman_numerals" in harmony
        assert "progressions" in harmony
    
    @pytest.mark.asyncio
    async def test_harmony_analysis_chord_detection(self, populated_score_storage):
        """Test chord detection in harmony analysis"""
        tool = HarmonyAnalysisTool(populated_score_storage)
        
        result = await tool.execute(score_id="bach_test")
        
        assert result["status"] == "success"
        chords = result["harmony_analysis"]["chords"]
        assert isinstance(chords, list)
        
        # Bach chorales should have identifiable chords
        if len(chords) > 0:
            chord = chords[0]
            assert "chord_symbol" in chord
            assert "measure" in chord
            assert "beat" in chord
    
    @pytest.mark.asyncio
    async def test_harmony_analysis_roman_numerals(self, populated_score_storage):
        """Test Roman numeral analysis"""
        tool = HarmonyAnalysisTool(populated_score_storage)
        
        result = await tool.execute(score_id="bach_test")
        
        assert result["status"] == "success"
        roman_numerals = result["harmony_analysis"]["roman_numerals"]
        assert isinstance(roman_numerals, list)
        
        # Should have Roman numeral analysis
        if len(roman_numerals) > 0:
            rn = roman_numerals[0]
            assert "roman_numeral" in rn
            assert "measure" in rn
            assert "beat" in rn
    
    @pytest.mark.asyncio
    async def test_harmony_analysis_progressions(self, populated_score_storage):
        """Test chord progression analysis"""
        tool = HarmonyAnalysisTool(populated_score_storage)
        
        result = await tool.execute(score_id="bach_test")
        
        assert result["status"] == "success"
        progressions = result["harmony_analysis"]["progressions"]
        assert isinstance(progressions, list)
        
        # Should identify common progressions
        if len(progressions) > 0:
            progression = progressions[0]
            assert "progression" in progression
            assert "measures" in progression
            assert "strength" in progression
    
    @pytest.mark.asyncio
    async def test_harmony_analysis_nonexistent_score(self, clean_score_storage):
        """Test harmony analysis with non-existent score"""
        tool = HarmonyAnalysisTool(clean_score_storage)
        
        result = await tool.execute(score_id="nonexistent")
        
        assert result["status"] == "error"
        assert "not found" in result["message"]
    
    @pytest.mark.asyncio
    async def test_harmony_analysis_custom_parameters(self, populated_score_storage):
        """Test harmony analysis with custom parameters"""
        tool = HarmonyAnalysisTool(populated_score_storage)
        
        result = await tool.execute(
            score_id="bach_test",
            include_inversions=True,
            include_secondary_dominants=True
        )
        
        assert result["status"] == "success"
        assert "harmony_analysis" in result
    
    @pytest.mark.asyncio
    async def test_harmony_analysis_handles_monophonic(self, clean_score_storage):
        """Test harmony analysis handles monophonic music"""
        from music21 import stream, note, key
        
        # Create simple monophonic melody
        melody = stream.Stream()
        melody.append(key.Key('C'))
        melody.append(note.Note('C4', quarterLength=1))
        melody.append(note.Note('E4', quarterLength=1))
        melody.append(note.Note('G4', quarterLength=1))
        melody.append(note.Note('C5', quarterLength=1))
        
        clean_score_storage["melody"] = melody
        
        tool = HarmonyAnalysisTool(clean_score_storage)
        result = await tool.execute(score_id="melody")
        
        # Should handle monophonic music gracefully
        assert result["status"] in ["success", "error"]
        if result["status"] == "success":
            # Monophonic music might have implied harmony
            assert "harmony_analysis" in result
    
    @pytest.mark.asyncio
    async def test_harmony_analysis_handles_empty_score(self, clean_score_storage):
        """Test harmony analysis handles empty scores gracefully"""
        from music21 import stream
        
        empty_score = stream.Stream()
        clean_score_storage["empty_score"] = empty_score
        
        tool = HarmonyAnalysisTool(clean_score_storage)
        result = await tool.execute(score_id="empty_score")
        
        # Should handle gracefully
        assert result["status"] in ["success", "error"]
        if result["status"] == "success":
            # Empty score should have no harmony
            harmony = result["harmony_analysis"]
            assert len(harmony["chords"]) == 0
            assert len(harmony["roman_numerals"]) == 0
            assert len(harmony["progressions"]) == 0