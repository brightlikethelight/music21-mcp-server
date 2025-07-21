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
        assert "chord_progression" in result
        assert "harmonic_rhythm" in result
        assert "summary" in result

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
        chord_progression = result["chord_progression"]
        assert isinstance(chord_progression, list)

        if len(chord_progression) > 0:
            chord = chord_progression[0]
            assert "symbol" in chord
            assert "pitches" in chord
            assert "offset" in chord
            assert "quality" in chord
            assert "root" in chord

    @pytest.mark.asyncio
    async def test_chord_analysis_histogram(self, populated_score_storage):
        """Test chord histogram generation"""
        tool = ChordAnalysisTool(populated_score_storage)

        result = await tool.execute(score_id="bach_test")

        assert result["status"] == "success"
        summary = result["summary"]
        assert isinstance(summary, dict)
        assert "chord_qualities" in summary

        # Check chord qualities structure
        qualities = summary["chord_qualities"]
        assert isinstance(qualities, dict)
        for quality, count in qualities.items():
            assert isinstance(quality, str)
            assert isinstance(count, int)
            assert count >= 0

    @pytest.mark.asyncio
    async def test_chord_analysis_common_chords(self, populated_score_storage):
        """Test most common chords analysis"""
        tool = ChordAnalysisTool(populated_score_storage)

        result = await tool.execute(score_id="bach_test")

        assert result["status"] == "success"
        summary = result["summary"]
        assert "most_common_chords" in summary
        common_chords = summary["most_common_chords"]
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
            score_id="bach_test", include_inversions=True, include_seventh_chords=True
        )

        assert result["status"] == "success"
        assert "chord_progression" in result
        assert "harmonic_rhythm" in result

    @pytest.mark.asyncio
    async def test_chord_analysis_monophonic_music(self, clean_score_storage):
        """Test chord analysis on monophonic music"""
        from music21 import note, stream

        # Create monophonic melody
        melody = stream.Stream()
        for pitch in ["C4", "E4", "G4", "C5"]:
            melody.append(note.Note(pitch, quarterLength=1))

        clean_score_storage["melody"] = melody

        tool = ChordAnalysisTool(clean_score_storage)
        result = await tool.execute(score_id="melody")

        # Should handle monophonic music gracefully
        assert result["status"] in ["success", "error"]
        if result["status"] == "success":
            # Monophonic music should have no or few chords
            chord_progression = result["chord_progression"]
            assert isinstance(chord_progression, list)

    @pytest.mark.asyncio
    async def test_chord_analysis_complex_harmony(self, clean_score_storage):
        """Test chord analysis on complex harmony"""
        from music21 import chord, key, stream

        # Create score with complex chords
        score = stream.Score()
        score.append(key.Key("C"))

        # Add various chord types
        chords_to_add = [
            chord.Chord(["C4", "E4", "G4"]),  # C major
            chord.Chord(["D4", "F4", "A4"]),  # D minor
            chord.Chord(["G3", "B3", "D4", "F4"]),  # G7
            chord.Chord(["C4", "E4", "G4", "B4"]),  # Cmaj7
            chord.Chord(["F#4", "A4", "C5", "E5"]),  # F#dim7
        ]

        for ch in chords_to_add:
            ch.quarterLength = 1
            score.append(ch)

        clean_score_storage["complex"] = score

        tool = ChordAnalysisTool(clean_score_storage)
        result = await tool.execute(score_id="complex")

        assert result["status"] == "success"
        assert "chord_progression" in result
        assert "summary" in result
        # Should have detected all 5 chords
        assert result["total_chords"] == 5
        # Should have multiple chord qualities
        assert len(result["summary"]["chord_qualities"]) >= 2
