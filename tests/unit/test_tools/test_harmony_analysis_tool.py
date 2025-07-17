"""
Comprehensive unit tests for HarmonyAnalysisTool
Tests Roman numeral analysis, chord progressions, functional harmony, and edge cases
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from music21_mcp.tools.harmony_analysis_tool import HarmonyAnalysisTool
from music21 import stream, note, chord, key, roman, meter

class TestHarmonyAnalysisTool:
    """Test suite for HarmonyAnalysisTool"""
    
    @pytest.fixture
    def tool(self):
        """Create tool instance with mock storage"""
        return HarmonyAnalysisTool({})
    
    @pytest.fixture
    def simple_progression_c_major(self):
        """Create I-IV-V-I progression in C major"""
        s = stream.Stream()
        s.append(key.Key('C'))
        s.append(meter.TimeSignature('4/4'))
        
        # I - IV - V - I
        chords = [
            chord.Chord(['C4', 'E4', 'G4'], quarterLength=1),    # I
            chord.Chord(['F4', 'A4', 'C5'], quarterLength=1),    # IV
            chord.Chord(['G4', 'B4', 'D5'], quarterLength=1),    # V
            chord.Chord(['C4', 'E4', 'G4'], quarterLength=1),    # I
        ]
        
        for c in chords:
            s.append(c)
        
        return s
    
    @pytest.fixture
    def ii_v_i_progression(self):
        """Create ii-V-I progression in C major"""
        s = stream.Stream()
        s.append(key.Key('C'))
        
        # ii - V - I
        chords = [
            chord.Chord(['D4', 'F4', 'A4'], quarterLength=1),    # ii
            chord.Chord(['G4', 'B4', 'D5', 'F5'], quarterLength=1),  # V7
            chord.Chord(['C4', 'E4', 'G4'], quarterLength=1),    # I
        ]
        
        for c in chords:
            s.append(c)
        
        return s
    
    @pytest.fixture
    def minor_progression(self):
        """Create i-iv-V-i progression in A minor"""
        s = stream.Stream()
        s.append(key.Key('a'))  # A minor
        
        # i - iv - V - i
        chords = [
            chord.Chord(['A3', 'C4', 'E4'], quarterLength=1),    # i
            chord.Chord(['D4', 'F4', 'A4'], quarterLength=1),    # iv
            chord.Chord(['E4', 'G#4', 'B4'], quarterLength=1),   # V (raised 7th)
            chord.Chord(['A3', 'C4', 'E4'], quarterLength=1),    # i
        ]
        
        for c in chords:
            s.append(c)
        
        return s
    
    @pytest.fixture
    def mixed_content_score(self):
        """Create score with both notes and chords"""
        s = stream.Stream()
        s.append(key.Key('G'))
        
        # Mix of single notes and chords
        s.append(note.Note('G4', quarterLength=0.5))
        s.append(note.Note('D4', quarterLength=0.5))
        s.append(chord.Chord(['G3', 'B3', 'D4'], quarterLength=1))  # I
        s.append(note.Note('C5', quarterLength=0.5))
        s.append(note.Note('A4', quarterLength=0.5))
        s.append(chord.Chord(['D4', 'F#4', 'A4'], quarterLength=1))  # V
        
        return s
    
    @pytest.mark.asyncio
    async def test_basic_roman_numeral_analysis(self, tool, simple_progression_c_major):
        """Test basic Roman numeral analysis"""
        tool.scores = {"test_score": simple_progression_c_major}
        
        result = await tool.execute(
            score_id="test_score"
        )
        
        assert result["status"] == "success"
        assert "roman_numerals" in result
        assert len(result["roman_numerals"]) == 4
        
        # Check Roman numerals
        numerals = [rn["roman_numeral"] for rn in result["roman_numerals"]]
        assert numerals == ["I", "IV", "V", "I"]
        
        # Check all have correct key context
        assert all(rn["key"] == "C major" for rn in result["roman_numerals"])
    
    @pytest.mark.asyncio
    async def test_chord_progression_detection(self, tool, simple_progression_c_major):
        """Test chord progression pattern detection"""
        tool.scores = {"test_score": simple_progression_c_major}
        
        result = await tool.execute(
            score_id="test_score"
        )
        
        assert result["status"] == "success"
        assert "progressions" in result
        assert len(result["progressions"]) > 0
        
        # Should detect I-IV-V-I as a common progression
        progression = result["progressions"][0]
        assert progression["pattern"] == ["I", "IV", "V", "I"]
        assert progression["type"] == "authentic_cadence"
        assert progression["strength"] >= 0.8
    
    @pytest.mark.asyncio
    async def test_functional_harmony_analysis(self, tool, simple_progression_c_major):
        """Test functional harmony categorization"""
        tool.scores = {"test_score": simple_progression_c_major}
        
        result = await tool.execute(
            score_id="test_score"
        )
        
        assert result["status"] == "success"
        assert "functional_analysis" in result
        
        functions = result["functional_analysis"]
        assert functions["tonic_chords"] > 0
        assert functions["dominant_chords"] > 0
        assert functions["subdominant_chords"] > 0
    
    @pytest.mark.asyncio
    async def test_harmonic_rhythm_analysis(self, tool, simple_progression_c_major):
        """Test harmonic rhythm analysis"""
        tool.scores = {"test_score": simple_progression_c_major}
        
        result = await tool.execute(
            score_id="test_score"
        )
        
        assert result["status"] == "success"
        assert "harmonic_rhythm" in result
        
        rhythm = result["harmonic_rhythm"]
        assert rhythm["average_duration"] == 1.0  # Each chord lasts 1 quarter
        assert rhythm["changes_per_measure"] == 4  # 4 chords in 4/4 time
    
    @pytest.mark.asyncio
    async def test_ii_v_i_progression(self, tool, ii_v_i_progression):
        """Test ii-V-I progression detection"""
        tool.scores = {"test_score": ii_v_i_progression}
        
        result = await tool.execute(
            score_id="test_score"
        )
        
        assert result["status"] == "success"
        
        # Check Roman numerals
        numerals = [rn["roman_numeral"] for rn in result["roman_numerals"]]
        assert "ii" in numerals or "ii6" in numerals
        assert "V" in numerals or "V7" in numerals
        assert "I" in numerals
        
        # Check progression detection
        progressions = result["progressions"]
        assert any(p["type"] == "ii_V_I" for p in progressions)
    
    @pytest.mark.asyncio
    async def test_minor_key_analysis(self, tool, minor_progression):
        """Test analysis in minor key"""
        tool.scores = {"test_score": minor_progression}
        
        result = await tool.execute(
            score_id="test_score"
        )
        
        assert result["status"] == "success"
        
        # Check Roman numerals are lowercase for minor
        numerals = [rn["roman_numeral"] for rn in result["roman_numerals"]]
        assert "i" in numerals
        assert "iv" in numerals
        
        # Check key context
        assert all(rn["key"] == "a minor" for rn in result["roman_numerals"])
    
    @pytest.mark.asyncio
    async def test_chord_inversions(self, tool):
        """Test detection of chord inversions"""
        s = stream.Stream()
        s.append(key.Key('F'))
        
        # Root position, first inversion, second inversion
        chords = [
            chord.Chord(['F3', 'A3', 'C4'], quarterLength=1),    # I
            chord.Chord(['A3', 'C4', 'F4'], quarterLength=1),    # I6
            chord.Chord(['C4', 'F4', 'A4'], quarterLength=1),    # I64
        ]
        
        for c in chords:
            s.append(c)
        
        tool.scores = {"test_score": s}
        
        result = await tool.execute(
            score_id="test_score"
        )
        
        assert result["status"] == "success"
        
        # Check inversions are detected
        numerals = result["roman_numerals"]
        assert any("6" in rn["roman_numeral"] for rn in numerals)  # First inversion
        assert any("64" in rn["roman_numeral"] for rn in numerals)  # Second inversion
    
    @pytest.mark.asyncio
    async def test_seventh_chords(self, tool):
        """Test analysis of seventh chords"""
        s = stream.Stream()
        s.append(key.Key('G'))
        
        # Various seventh chords
        chords = [
            chord.Chord(['G3', 'B3', 'D4', 'F#4'], quarterLength=1),  # IMaj7
            chord.Chord(['A3', 'C4', 'E4', 'G4'], quarterLength=1),   # ii7
            chord.Chord(['D4', 'F#4', 'A4', 'C5'], quarterLength=1),  # V7
        ]
        
        for c in chords:
            s.append(c)
        
        tool.scores = {"test_score": s}
        
        result = await tool.execute(
            score_id="test_score"
        )
        
        assert result["status"] == "success"
        
        # Check seventh chords are recognized
        assert any("7" in rn["roman_numeral"] for rn in result["roman_numerals"])
        
        # Check chord qualities
        assert any(rn["quality"] == "major-seventh" for rn in result["roman_numerals"])
        assert any(rn["quality"] == "dominant-seventh" for rn in result["roman_numerals"])
    
    @pytest.mark.asyncio
    async def test_mixed_content_analysis(self, tool, mixed_content_score):
        """Test analysis with mixed notes and chords"""
        tool.scores = {"test_score": mixed_content_score}
        
        result = await tool.execute(
            score_id="test_score"
        )
        
        assert result["status"] == "success"
        
        # Should still extract and analyze chords
        assert len(result["roman_numerals"]) >= 2
        assert "chord_density" in result
        assert result["chord_density"] < 1.0  # Not all elements are chords
    
    @pytest.mark.asyncio
    async def test_chromatic_harmony(self, tool):
        """Test analysis of chromatic chords"""
        s = stream.Stream()
        s.append(key.Key('C'))
        
        # Include Neapolitan and augmented sixth
        chords = [
            chord.Chord(['C4', 'E4', 'G4'], quarterLength=1),     # I
            chord.Chord(['Db4', 'F4', 'Ab4'], quarterLength=1),   # bII (Neapolitan)
            chord.Chord(['G4', 'B4', 'D5'], quarterLength=1),     # V
            chord.Chord(['Ab3', 'C4', 'F#4'], quarterLength=1),   # Aug6
            chord.Chord(['G3', 'B3', 'D4'], quarterLength=1),     # V
            chord.Chord(['C4', 'E4', 'G4'], quarterLength=1),     # I
        ]
        
        for c in chords:
            s.append(c)
        
        tool.scores = {"test_score": s}
        
        result = await tool.execute(
            score_id="test_score"
        )
        
        assert result["status"] == "success"
        
        # Check for chromatic chord detection
        assert any("bII" in rn["roman_numeral"] for rn in result["roman_numerals"])
        assert "chromaticism_level" in result
        assert result["chromaticism_level"] > 0.2  # Some chromatic content
    
    @pytest.mark.asyncio
    async def test_modulation_detection(self, tool):
        """Test detection of key modulation"""
        s = stream.Stream()
        
        # Start in C major
        s.append(key.Key('C'))
        s.append(chord.Chord(['C4', 'E4', 'G4'], quarterLength=1))    # I
        s.append(chord.Chord(['G4', 'B4', 'D5'], quarterLength=1))    # V
        
        # Modulate to G major
        s.append(key.Key('G'))
        s.append(chord.Chord(['G4', 'B4', 'D5'], quarterLength=1))    # I (in G)
        s.append(chord.Chord(['D5', 'F#5', 'A5'], quarterLength=1))   # V (in G)
        s.append(chord.Chord(['G4', 'B4', 'D5'], quarterLength=1))    # I (in G)
        
        tool.scores = {"test_score": s}
        
        result = await tool.execute(
            score_id="test_score"
        )
        
        assert result["status"] == "success"
        
        # Check for modulation detection
        assert "modulations" in result
        modulations = result["modulations"]
        assert len(modulations) > 0
        assert any(m["from_key"] == "C major" and m["to_key"] == "G major" 
                  for m in modulations)
    
    @pytest.mark.asyncio
    async def test_cadence_detection(self, tool):
        """Test cadence detection"""
        s = stream.Stream()
        s.append(key.Key('D'))
        
        # Various cadence types
        # Perfect authentic cadence (V-I)
        s.append(chord.Chord(['A4', 'C#5', 'E5'], quarterLength=1))   # V
        s.append(chord.Chord(['D4', 'F#4', 'A4'], quarterLength=2))   # I
        
        # Plagal cadence (IV-I)
        s.append(chord.Chord(['G4', 'B4', 'D5'], quarterLength=1))    # IV
        s.append(chord.Chord(['D4', 'F#4', 'A4'], quarterLength=2))   # I
        
        # Deceptive cadence (V-vi)
        s.append(chord.Chord(['A4', 'C#5', 'E5'], quarterLength=1))   # V
        s.append(chord.Chord(['B4', 'D5', 'F#5'], quarterLength=2))   # vi
        
        tool.scores = {"test_score": s}
        
        result = await tool.execute(
            score_id="test_score"
        )
        
        assert result["status"] == "success"
        assert "cadences" in result
        
        cadences = result["cadences"]
        assert len(cadences) >= 2
        
        # Check cadence types
        cadence_types = [c["type"] for c in cadences]
        assert "authentic" in cadence_types or "perfect_authentic" in cadence_types
        assert "plagal" in cadence_types
        assert "deceptive" in cadence_types
    
    @pytest.mark.asyncio
    async def test_empty_score(self, tool):
        """Test handling of empty score"""
        empty_score = stream.Stream()
        tool.scores = {"test_score": empty_score}
        
        result = await tool.execute(
            score_id="test_score"
        )
        
        assert result["status"] == "success"
        assert result["roman_numerals"] == []
        assert result["progressions"] == []
        assert result["chord_count"] == 0
    
    @pytest.mark.asyncio
    async def test_missing_score_error(self, tool):
        """Test error handling for missing score"""
        result = await tool.execute(
            score_id="nonexistent"
        )
        
        assert result["status"] == "error"
        assert "not found" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_progress_callback(self, tool, simple_progression_c_major):
        """Test progress reporting"""
        tool.scores = {"test_score": simple_progression_c_major}
        progress_calls = []
        
        async def progress_callback(msg):
            progress_calls.append(msg)
        
        result = await tool.execute(
            score_id="test_score",
            progress_callback=progress_callback
        )
        
        assert result["status"] == "success"
        assert len(progress_calls) > 0
        assert any("Analyzing" in call for call in progress_calls)
    
    @pytest.mark.asyncio
    async def test_non_functional_harmony(self, tool):
        """Test analysis of non-functional harmony"""
        s = stream.Stream()
        # No key signature - ambiguous tonality
        
        # Parallel major chords (non-functional)
        chords = [
            chord.Chord(['C4', 'E4', 'G4'], quarterLength=1),     # C
            chord.Chord(['D4', 'F#4', 'A4'], quarterLength=1),    # D
            chord.Chord(['E4', 'G#4', 'B4'], quarterLength=1),    # E
            chord.Chord(['F4', 'A4', 'C5'], quarterLength=1),     # F
        ]
        
        for c in chords:
            s.append(c)
        
        tool.scores = {"test_score": s}
        
        result = await tool.execute(
            score_id="test_score"
        )
        
        assert result["status"] == "success"
        
        # Should handle non-functional harmony
        assert "tonality_strength" in result
        assert result["tonality_strength"] < 0.5  # Weak tonality
    
    @pytest.mark.asyncio
    async def test_sus_chords(self, tool):
        """Test suspended chord analysis"""
        s = stream.Stream()
        s.append(key.Key('G'))
        
        # Sus4 and sus2 chords
        chords = [
            chord.Chord(['G4', 'C5', 'D5'], quarterLength=1),     # Isus4
            chord.Chord(['G4', 'B4', 'D5'], quarterLength=1),     # I
            chord.Chord(['A4', 'B4', 'E5'], quarterLength=1),     # iisus2
            chord.Chord(['A4', 'C5', 'E5'], quarterLength=1),     # ii
        ]
        
        for c in chords:
            s.append(c)
        
        tool.scores = {"test_score": s}
        
        result = await tool.execute(
            score_id="test_score"
        )
        
        assert result["status"] == "success"
        
        # Check suspended chords are recognized
        assert any("sus" in rn.get("quality", "") for rn in result["roman_numerals"])
    
    @pytest.mark.asyncio
    async def test_performance_large_score(self, tool):
        """Test performance with large score"""
        s = stream.Stream()
        s.append(key.Key('C'))
        
        # Create 100 measures of chord progressions
        progression = [
            chord.Chord(['C4', 'E4', 'G4'], quarterLength=1),
            chord.Chord(['A3', 'C4', 'E4'], quarterLength=1),
            chord.Chord(['F4', 'A4', 'C5'], quarterLength=1),
            chord.Chord(['G4', 'B4', 'D5'], quarterLength=1),
        ]
        
        for _ in range(100):
            for c in progression:
                s.append(c)
        
        tool.scores = {"test_score": s}
        
        import time
        start_time = time.time()
        
        result = await tool.execute(
            score_id="test_score"
        )
        
        execution_time = time.time() - start_time
        
        assert result["status"] == "success"
        assert execution_time < 5.0  # Should complete within 5 seconds
        assert result["chord_count"] == 400