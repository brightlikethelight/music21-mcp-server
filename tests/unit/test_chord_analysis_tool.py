"""
Unit tests for ChordAnalysisTool
"""
import pytest
from music21_mcp.tools.chord_analysis_tool import ChordAnalysisTool


class TestChordAnalysisTool:
    """Test ChordAnalysisTool functionality"""
    
    def test_tool_initialization(self, clean_score_storage):
        """Test tool can be initialized with score storage"""
        tool = ChordAnalysisTool(clean_score_storage)
        assert tool.scores == clean_score_storage
    
    @pytest.mark.asyncio
    async def test_chord_analysis_success(self, populated_score_storage):
        """Test successful chord analysis"""
        tool = ChordAnalysisTool(populated_score_storage)
        
        result = await tool.execute(score_id="bach_test")
        
        assert result["status"] == "success"
        assert "chord_analysis" in result
        analysis = result["chord_analysis"]
        assert "chords" in analysis
        assert "chord_histogram" in analysis
        assert "most_common_chords" in analysis
    
    @pytest.mark.asyncio
    async def test_chord_analysis_nonexistent_score(self, clean_score_storage):
        """Test chord analysis with non-existent score"""
        tool = ChordAnalysisTool(clean_score_storage)
        
        result = await tool.execute(score_id="nonexistent")
        
        assert result["status"] == "error"
        assert "not found" in result["message"]
    
    @pytest.mark.asyncio
    async def test_chord_analysis_detailed_chords(self, populated_score_storage):
        """Test detailed chord information"""
        tool = ChordAnalysisTool(populated_score_storage)
        
        result = await tool.execute(score_id="bach_test")
        
        assert result["status"] == "success"
        chords = result["chord_analysis"]["chords"]
        assert isinstance(chords, list)
        
        if len(chords) > 0:
            chord = chords[0]
            assert "chord_symbol" in chord
            assert "measure" in chord
            assert "beat" in chord
            assert "quality" in chord
            assert "root" in chord
    
    @pytest.mark.asyncio
    async def test_chord_analysis_histogram(self, populated_score_storage):
        """Test chord histogram generation"""
        tool = ChordAnalysisTool(populated_score_storage)
        
        result = await tool.execute(score_id="bach_test")
        
        assert result["status"] == "success"
        histogram = result["chord_analysis"]["chord_histogram"]
        assert isinstance(histogram, dict)
        
        # Histogram should have chord types as keys and counts as values
        for chord_type, count in histogram.items():
            assert isinstance(chord_type, str)
            assert isinstance(count, int)
            assert count >= 0
    
    @pytest.mark.asyncio
    async def test_chord_analysis_common_chords(self, populated_score_storage):
        """Test most common chords analysis"""
        tool = ChordAnalysisTool(populated_score_storage)
        
        result = await tool.execute(score_id="bach_test")
        
        assert result["status"] == "success"
        common_chords = result["chord_analysis"]["most_common_chords"]
        assert isinstance(common_chords, list)
        
        # Should be sorted by frequency
        if len(common_chords) > 1:
            for i in range(len(common_chords) - 1):
                assert common_chords[i]["count"] >= common_chords[i + 1]["count"]
    
    @pytest.mark.asyncio
    async def test_chord_analysis_custom_parameters(self, populated_score_storage):
        """Test chord analysis with custom parameters"""
        tool = ChordAnalysisTool(populated_score_storage)
        
        result = await tool.execute(
            score_id="bach_test",
            include_inversions=True,
            include_seventh_chords=True
        )
        
        assert result["status"] == "success"
        assert "chord_analysis" in result
    
    @pytest.mark.asyncio
    async def test_chord_analysis_monophonic_music(self, clean_score_storage):
        """Test chord analysis on monophonic music"""
        from music21 import stream, note
        
        # Create monophonic melody
        melody = stream.Stream()
        for pitch in ['C4', 'E4', 'G4', 'C5']:
            melody.append(note.Note(pitch, quarterLength=1))
        
        clean_score_storage["melody"] = melody
        
        tool = ChordAnalysisTool(clean_score_storage)
        result = await tool.execute(score_id="melody")
        
        # Should handle monophonic music gracefully
        assert result["status"] in ["success", "error"]
        if result["status"] == "success":
            # Monophonic music should have no or implied chords
            chords = result["chord_analysis"]["chords"]
            assert isinstance(chords, list)
    
    @pytest.mark.asyncio
    async def test_chord_analysis_complex_harmony(self, clean_score_storage):
        """Test chord analysis on complex harmony"""
        from music21 import stream, chord, key
        
        # Create score with complex chords
        score = stream.Score()
        score.append(key.Key('C'))
        
        # Add various chord types
        chords_to_add = [
            chord.Chord(['C4', 'E4', 'G4']),  # C major
            chord.Chord(['D4', 'F4', 'A4']),  # D minor
            chord.Chord(['G3', 'B3', 'D4', 'F4']),  # G7
            chord.Chord(['C4', 'E4', 'G4', 'B4']),  # Cmaj7
            chord.Chord(['F#4', 'A4', 'C5', 'E5']),  # F#dim7
        ]
        
        for ch in chords_to_add:
            ch.quarterLength = 1
            score.append(ch)
        
        clean_score_storage["complex"] = score
        
        tool = ChordAnalysisTool(clean_score_storage)
        result = await tool.execute(score_id="complex")
        
        assert result["status"] == "success"
        analysis = result["chord_analysis"]
        assert len(analysis["chords"]) == 5
        assert len(analysis["chord_histogram"]) >= 3  # At least 3 different chord types