"""Comprehensive unit tests for HarmonizationTool achieving >80% coverage"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from music21 import stream, note, chord, key, pitch, tempo

from music21_mcp.tools.harmonization_tool import HarmonizationTool


# Helper functions
def create_simple_melody():
    """Create a simple melody for testing"""
    melody = stream.Stream()
    melody.append(key.Key("C"))
    for pitch_name in ["C4", "D4", "E4", "F4", "G4"]:
        melody.append(note.Note(pitch_name, quarterLength=1))
    return melody


def create_note(pitch_name):
    """Create a note for testing"""
    return note.Note(pitch_name, quarterLength=1)


def create_basic_analysis():
    """Create basic analysis structure for testing"""
    return {
        "key": key.Key("C"),
        "contour": ["ascending", "ascending", "ascending", "ascending"],
        "implied_harmonies": [["I"], ["ii"], ["iii"], ["IV"], ["V"]],
        "phrase_points": [0, 4],
        "climax": 4,
        "range": 7
    }


def create_simple_score_with_parts():
    """Create a simple score with multiple parts for voice leading tests"""
    score = stream.Score()
    soprano = stream.Part()
    bass = stream.Part()
    
    soprano.append(note.Note("C5", quarterLength=1))
    soprano.append(note.Note("B4", quarterLength=1))
    bass.append(note.Note("C3", quarterLength=1))
    bass.append(note.Note("D3", quarterLength=1))
    
    score.insert(0, soprano)
    score.insert(0, bass)
    return score


class TestHarmonizationToolInitialization:
    """Test tool initialization and setup"""
    
    def test_harmonization_tool_initialization(self, clean_score_storage):
        """Test tool initialization with style vocabularies"""
        tool = HarmonizationTool(clean_score_storage)
        assert tool.scores == clean_score_storage
        assert "classical" in tool.style_vocabularies
        assert "jazz" in tool.style_vocabularies
        assert "pop" in tool.style_vocabularies
        assert "modal" in tool.style_vocabularies
        assert "classical" in tool.common_progressions


class TestInputValidation:
    """Test input validation methods"""
    
    def test_validate_inputs_missing_score(self, clean_score_storage):
        """Test validation with non-existent score"""
        tool = HarmonizationTool(clean_score_storage)
        error = tool.validate_inputs(score_id="nonexistent", style="classical")
        assert error is not None
        assert "not found" in error

    def test_validate_inputs_invalid_style(self, clean_score_storage):
        """Test validation with invalid harmonization style"""
        clean_score_storage["test"] = create_simple_melody()
        tool = HarmonizationTool(clean_score_storage)
        error = tool.validate_inputs(score_id="test", style="invalid_style")
        assert error is not None
        assert "Invalid style" in error

    def test_validate_inputs_invalid_voice_parts(self, clean_score_storage):
        """Test validation with invalid voice parts count"""
        clean_score_storage["test"] = create_simple_melody()
        tool = HarmonizationTool(clean_score_storage)
        error = tool.validate_inputs(score_id="test", style="classical", voice_parts=1)
        assert error is not None
        assert "voice_parts must be between 2 and 4" in error

    def test_validate_inputs_duplicate_output_id(self, clean_score_storage):
        """Test validation with existing output ID"""
        clean_score_storage["test"] = create_simple_melody()
        clean_score_storage["existing"] = create_simple_melody()
        tool = HarmonizationTool(clean_score_storage)
        error = tool.validate_inputs(score_id="test", output_id="existing", style="classical")
        assert error is not None
        assert "already exists" in error

    def test_validate_inputs_valid_parameters(self, clean_score_storage):
        """Test validation with all valid parameters"""
        clean_score_storage["test"] = create_simple_melody()
        tool = HarmonizationTool(clean_score_storage)
        error = tool.validate_inputs(score_id="test", style="classical", voice_parts=4)
        assert error is None


class TestMelodyExtraction:
    """Test melody extraction and analysis"""
    
    def test_extract_melody_from_single_part(self, clean_score_storage):
        """Test melody extraction from single-part score"""
        melody = create_simple_melody()
        tool = HarmonizationTool(clean_score_storage)
        extracted = tool._extract_melody(melody)
        assert len(extracted) > 0
        assert all(hasattr(n, 'pitch') for n in extracted)

    def test_extract_melody_from_multipart_score(self, clean_score_storage):
        """Test melody extraction from multi-part score (gets top part)"""
        score = stream.Score()
        part1 = stream.Part()
        part2 = stream.Part()
        part1.append(note.Note("C5", quarterLength=1))
        part2.append(note.Note("C3", quarterLength=1))
        score.insert(0, part1)
        score.insert(0, part2)
        
        tool = HarmonizationTool(clean_score_storage)
        extracted = tool._extract_melody(score)
        assert len(extracted) == 1
        assert extracted[0].pitch.name == "C"

    def test_extract_melody_empty_score(self, clean_score_storage):
        """Test melody extraction from empty score"""
        empty_score = stream.Stream()
        tool = HarmonizationTool(clean_score_storage)
        extracted = tool._extract_melody(empty_score)
        assert extracted == []

    @pytest.mark.asyncio
    async def test_analyze_melody_success(self, clean_score_storage):
        """Test successful melody analysis"""
        melody = [create_note("C4"), create_note("E4"), create_note("G4")]
        tool = HarmonizationTool(clean_score_storage)
        analysis = await tool._analyze_melody(melody)
        
        assert "key" in analysis
        assert "contour" in analysis
        assert "implied_harmonies" in analysis
        assert "phrase_points" in analysis
        assert "climax" in analysis
        assert "range" in analysis

    @pytest.mark.asyncio
    async def test_analyze_melody_empty(self, clean_score_storage):
        """Test melody analysis with empty melody"""
        tool = HarmonizationTool(clean_score_storage)
        analysis = await tool._analyze_melody([])
        assert "key" in analysis
        assert analysis["range"] == 0

    def test_analyze_contour(self, clean_score_storage):
        """Test contour analysis of pitch sequence"""
        tool = HarmonizationTool(clean_score_storage)
        pitches = [60, 64, 67, 65, 62]  # C, E, G, F, D
        contour = tool._analyze_contour(pitches)
        expected = ["ascending", "ascending", "descending", "descending"]
        assert contour == expected

    def test_get_implied_harmony(self, clean_score_storage):
        """Test getting implied harmonies for scale degrees"""
        tool = HarmonizationTool(clean_score_storage)
        key_obj = key.Key("C")
        note_obj = note.Note("C4")
        
        implied = tool._get_implied_harmony(note_obj, key_obj)
        assert isinstance(implied, list)
        assert len(implied) > 0
        assert "I" in implied


class TestClassicalHarmonization:
    """Test classical harmonization methods"""
    
    @pytest.mark.asyncio
    async def test_harmonize_classical_success(self, clean_score_storage):
        """Test successful classical harmonization"""
        melody = [create_note("C4"), create_note("D4"), create_note("E4")]
        analysis = create_basic_analysis()
        tool = HarmonizationTool(clean_score_storage)
        
        result = await tool._harmonize_classical(melody, analysis, None, 4)
        
        assert "score" in result
        assert "progression" in result
        assert "roman_numerals" in result
        assert "confidence_ratings" in result
        assert len(result["score"].parts) == 4  # SATB

    @pytest.mark.asyncio
    async def test_harmonize_classical_with_constraints(self, clean_score_storage):
        """Test classical harmonization with diatonic constraints"""
        melody = [create_note("C4"), create_note("D4"), create_note("E4")]
        analysis = create_basic_analysis()
        constraints = ["diatonic_only"]
        tool = HarmonizationTool(clean_score_storage)
        
        result = await tool._harmonize_classical(melody, analysis, constraints, 4)
        # All chords should be diatonic
        for chord_symbol in result["progression"]:
            assert chord_symbol in ["I", "ii", "iii", "IV", "V", "vi", "viio"]

    @pytest.mark.asyncio
    async def test_harmonize_classical_voice_parts_3(self, clean_score_storage):
        """Test classical harmonization with 3 voices"""
        melody = [create_note("C4"), create_note("D4")]
        analysis = create_basic_analysis()
        tool = HarmonizationTool(clean_score_storage)
        
        result = await tool._harmonize_classical(melody, analysis, None, 3)
        assert len(result["score"].parts) == 3  # Soprano, Alto, Bass

    @pytest.mark.asyncio
    async def test_harmonize_classical_voice_parts_2(self, clean_score_storage):
        """Test classical harmonization with 2 voices"""
        melody = [create_note("C4")]
        analysis = create_basic_analysis()
        tool = HarmonizationTool(clean_score_storage)
        
        result = await tool._harmonize_classical(melody, analysis, None, 2)
        assert len(result["score"].parts) == 2  # Soprano, Bass

    def test_generate_progression_classical(self, clean_score_storage):
        """Test classical progression generation"""
        melody = [create_note("C4"), create_note("D4"), create_note("E4")]
        analysis = create_basic_analysis()
        tool = HarmonizationTool(clean_score_storage)
        
        progression = tool._generate_progression_classical(melody, analysis, None)
        assert len(progression) == len(melody)
        assert progression[-1] == "I"  # Should end on tonic

    def test_is_chord_compatible(self, clean_score_storage):
        """Test chord compatibility checking with melody notes"""
        tool = HarmonizationTool(clean_score_storage)
        key_obj = key.Key("C")
        note_obj = note.Note("E4")
        
        # E is in C major chord, so should be compatible
        assert tool._is_chord_compatible("I", note_obj, key_obj) is True

    def test_realize_chord_classical(self, clean_score_storage):
        """Test classical chord realization"""
        tool = HarmonizationTool(clean_score_storage)
        key_obj = key.Key("C")
        note_obj = note.Note("C4")
        
        pitches = tool._realize_chord_classical("I", key_obj, note_obj)
        assert len(pitches) >= 3  # At least a triad
        assert note_obj.pitch in pitches

    def test_choose_alto_note(self, clean_score_storage):
        """Test alto voice note selection"""
        tool = HarmonizationTool(clean_score_storage)
        chord_pitches = [pitch.Pitch("C4"), pitch.Pitch("E4"), pitch.Pitch("G4")]
        melody_note = note.Note("G4")
        
        alto_pitch = tool._choose_alto_note(chord_pitches, melody_note)
        assert alto_pitch.midi != melody_note.pitch.midi  # Different from melody

    def test_choose_tenor_note(self, clean_score_storage):
        """Test tenor voice note selection"""
        tool = HarmonizationTool(clean_score_storage)
        chord_pitches = [pitch.Pitch("C4"), pitch.Pitch("E4"), pitch.Pitch("G4")]
        melody_note = note.Note("G4")
        alto_pitch = pitch.Pitch("E4")
        
        tenor_pitch = tool._choose_tenor_note(chord_pitches, melody_note, alto_pitch)
        assert tenor_pitch.midi not in {melody_note.pitch.midi, alto_pitch.midi}


class TestJazzHarmonization:
    """Test jazz harmonization methods"""
    
    @pytest.mark.asyncio
    async def test_harmonize_jazz_success(self, clean_score_storage):
        """Test successful jazz harmonization"""
        melody = [create_note("C4"), create_note("D4")]
        analysis = create_basic_analysis()
        tool = HarmonizationTool(clean_score_storage)
        
        result = await tool._harmonize_jazz(melody, analysis, None)
        
        assert "score" in result
        assert "progression" in result
        assert len(result["score"].parts) == 2  # Piano hands

    def test_generate_progression_jazz(self, clean_score_storage):
        """Test jazz progression generation"""
        melody = [create_note("C4"), create_note("D4")]
        analysis = create_basic_analysis()
        tool = HarmonizationTool(clean_score_storage)
        
        progression = tool._generate_progression_jazz(melody, analysis, None)
        assert len(progression) == len(melody)
        # Should contain jazz chord symbols
        assert any("7" in chord_symbol for chord_symbol in progression)

    def test_generate_progression_jazz_no_substitutions(self, clean_score_storage):
        """Test jazz progression without substitutions"""
        melody = [create_note("C4")]
        analysis = create_basic_analysis()
        constraints = ["no_substitutions"]
        tool = HarmonizationTool(clean_score_storage)
        
        progression = tool._generate_progression_jazz(melody, analysis, constraints)
        # Should not contain tritone substitutions
        assert "bII7" not in progression

    def test_create_jazz_voicing_maj7(self, clean_score_storage):
        """Test jazz major 7 voicing creation"""
        tool = HarmonizationTool(clean_score_storage)
        key_obj = key.Key("C")
        note_obj = note.Note("C4")
        
        voicing = tool._create_jazz_voicing("IMaj7", key_obj, note_obj)
        assert "upper" in voicing
        assert "lower" in voicing
        assert len(voicing["upper"]) >= 1
        assert len(voicing["lower"]) >= 1

    def test_create_jazz_voicing_dominant7(self, clean_score_storage):
        """Test jazz dominant 7 voicing creation"""
        tool = HarmonizationTool(clean_score_storage)
        key_obj = key.Key("C")
        note_obj = note.Note("G4")
        
        voicing = tool._create_jazz_voicing("V7", key_obj, note_obj)
        assert "upper" in voicing
        assert "lower" in voicing

    def test_create_jazz_voicing_fallback(self, clean_score_storage):
        """Test jazz voicing creation fallback for errors"""
        tool = HarmonizationTool(clean_score_storage)
        key_obj = key.Key("C")
        note_obj = note.Note("C4")
        
        # Test with invalid chord symbol to trigger fallback
        voicing = tool._create_jazz_voicing("InvalidChord", key_obj, note_obj)
        assert "upper" in voicing
        assert "lower" in voicing


class TestPopHarmonization:
    """Test pop harmonization methods"""
    
    @pytest.mark.asyncio
    async def test_harmonize_pop_success(self, clean_score_storage):
        """Test successful pop harmonization"""
        melody = [create_note("C4")]
        analysis = create_basic_analysis()
        tool = HarmonizationTool(clean_score_storage)
        
        result = await tool._harmonize_pop(melody, analysis, None)
        
        assert "score" in result
        assert len(result["score"].parts) == 3  # Melody, Guitar, Bass

    def test_generate_progression_pop(self, clean_score_storage):
        """Test pop progression generation"""
        melody = [create_note("C4"), create_note("D4")]
        analysis = create_basic_analysis()
        tool = HarmonizationTool(clean_score_storage)
        
        progression = tool._generate_progression_pop(melody, analysis, None)
        assert len(progression) == len(melody)
        # Should use common pop chords
        assert all(chord_symbol in ["I", "V", "vi", "IV", "ii", "iii", "bVII", "bIII"] 
                   for chord_symbol in progression)

    def test_create_pop_voicing_power_chord(self, clean_score_storage):
        """Test pop power chord voicing creation"""
        tool = HarmonizationTool(clean_score_storage)
        key_obj = key.Key("C")
        
        # Mock random to always return true for power chord
        with patch('random.random', return_value=0.1):
            voicing = tool._create_pop_voicing("I", key_obj)
            # Power chord should have 2 notes (root + fifth)
            assert len(voicing) == 2

    def test_create_pop_voicing_triad(self, clean_score_storage):
        """Test pop triad voicing creation"""
        tool = HarmonizationTool(clean_score_storage)
        key_obj = key.Key("C")
        
        # Mock random to always return false for power chord
        with patch('random.random', return_value=0.5):
            voicing = tool._create_pop_voicing("I", key_obj)
            # Full triad should have 3 notes
            assert len(voicing) == 3

    def test_create_pop_voicing_fallback(self, clean_score_storage):
        """Test pop voicing creation fallback"""
        tool = HarmonizationTool(clean_score_storage)
        key_obj = key.Key("C")
        
        # Test with invalid chord to trigger fallback
        voicing = tool._create_pop_voicing("InvalidChord", key_obj)
        assert len(voicing) == 2  # Fallback to root + fifth


class TestModalHarmonization:
    """Test modal harmonization methods"""
    
    @pytest.mark.asyncio
    async def test_harmonize_modal_success(self, clean_score_storage):
        """Test successful modal harmonization"""
        melody = [create_note("D4")]
        analysis = create_basic_analysis()
        tool = HarmonizationTool(clean_score_storage)
        
        result = await tool._harmonize_modal(melody, analysis, None)
        
        assert "score" in result
        assert "modal_center" in result
        assert len(result["score"].parts) == 2

    def test_detect_mode_mixolydian(self, clean_score_storage):
        """Test detection of mixolydian mode"""
        tool = HarmonizationTool(clean_score_storage)
        melody = [note.Note("C4"), note.Note("Bb4")]  # Contains b7, no 4
        analysis = {}
        
        mode = tool._detect_mode(melody, analysis)
        assert mode == "mixolydian"

    def test_detect_mode_dorian(self, clean_score_storage):
        """Test detection of dorian mode"""
        tool = HarmonizationTool(clean_score_storage)
        melody = [note.Note("D4"), note.Note("B4"), note.Note("E4")]  # Natural 6 and 2 in minor
        analysis = {}
        
        mode = tool._detect_mode(melody, analysis)
        assert mode == "dorian"

    def test_detect_mode_lydian(self, clean_score_storage):
        """Test detection of lydian mode"""
        tool = HarmonizationTool(clean_score_storage)
        melody = [note.Note("F4"), note.Note("F#4")]  # Contains #4
        analysis = {}
        
        mode = tool._detect_mode(melody, analysis)
        assert mode == "lydian"

    def test_detect_mode_default(self, clean_score_storage):
        """Test default mode detection (ionian)"""
        tool = HarmonizationTool(clean_score_storage)
        melody = [note.Note("C4"), note.Note("D4"), note.Note("E4")]
        analysis = {}
        
        mode = tool._detect_mode(melody, analysis)
        assert mode == "ionian"

    def test_generate_progression_modal(self, clean_score_storage):
        """Test modal progression generation"""
        melody = [create_note("D4"), create_note("E4")]
        analysis = create_basic_analysis()
        tool = HarmonizationTool(clean_score_storage)
        
        progression = tool._generate_progression_modal(melody, analysis, "dorian", None)
        assert len(progression) == len(melody)
        assert progression[0] in ["i"]  # Should start on tonic
        assert progression[-1] in ["i"]  # Should end on tonic

    def test_create_modal_voicing(self, clean_score_storage):
        """Test modal chord voicing creation"""
        tool = HarmonizationTool(clean_score_storage)
        key_obj = key.Key("C")
        
        voicing = tool._create_modal_voicing("i", key_obj, "dorian")
        assert len(voicing) >= 1

    def test_create_modal_voicing_fallback(self, clean_score_storage):
        """Test modal voicing creation fallback"""
        tool = HarmonizationTool(clean_score_storage)
        key_obj = key.Key("C")
        
        # Test with invalid chord to trigger fallback
        voicing = tool._create_modal_voicing("InvalidChord", key_obj, "dorian")
        assert len(voicing) == 1  # Fallback to tonic


class TestVoiceLeadingAnalysis:
    """Test voice leading analysis methods"""
    
    def test_check_voice_leading_success(self, clean_score_storage):
        """Test voice leading quality checking"""
        score = stream.Score()
        soprano = stream.Part()
        bass = stream.Part()
        
        # Create simple progression
        soprano.append(note.Note("C5", quarterLength=1))
        soprano.append(note.Note("D5", quarterLength=1))
        bass.append(note.Note("C3", quarterLength=1))
        bass.append(note.Note("D3", quarterLength=1))
        
        score.insert(0, soprano)
        score.insert(0, bass)
        
        tool = HarmonizationTool(clean_score_storage)
        quality = tool._check_voice_leading(score, "classical")
        
        assert "smoothness" in quality
        assert "errors" in quality
        assert "parallel_fifths" in quality
        assert "parallel_octaves" in quality

    def test_check_voice_leading_parallel_fifths(self, clean_score_storage):
        """Test detection of parallel fifths"""
        score = stream.Score()
        soprano = stream.Part()
        bass = stream.Part()
        
        # Create parallel fifths
        soprano.append(note.Note("G4", quarterLength=1))
        soprano.append(note.Note("A4", quarterLength=1))
        bass.append(note.Note("C3", quarterLength=1))
        bass.append(note.Note("D3", quarterLength=1))
        
        score.insert(0, soprano)
        score.insert(0, bass)
        
        tool = HarmonizationTool(clean_score_storage)
        quality = tool._check_voice_leading(score, "classical")
        
        assert quality["parallel_fifths"] > 0

    def test_check_voice_leading_parallel_octaves(self, clean_score_storage):
        """Test detection of parallel octaves"""
        score = stream.Score()
        soprano = stream.Part()
        bass = stream.Part()
        
        # Create parallel octaves
        soprano.append(note.Note("C5", quarterLength=1))
        soprano.append(note.Note("D5", quarterLength=1))
        bass.append(note.Note("C4", quarterLength=1))
        bass.append(note.Note("D4", quarterLength=1))
        
        score.insert(0, soprano)
        score.insert(0, bass)
        
        tool = HarmonizationTool(clean_score_storage)
        quality = tool._check_voice_leading(score, "classical")
        
        assert quality["parallel_octaves"] > 0

    def test_check_voice_leading_single_part(self, clean_score_storage):
        """Test voice leading check with single part (should return defaults)"""
        score = stream.Score()
        soprano = stream.Part()
        soprano.append(note.Note("C5", quarterLength=1))
        score.insert(0, soprano)
        
        tool = HarmonizationTool(clean_score_storage)
        quality = tool._check_voice_leading(score, "classical")
        
        assert quality["smoothness"] == 0.0
        assert quality["parallel_fifths"] == 0
        assert quality["parallel_octaves"] == 0

    def test_check_voice_leading_pop_style_leniency(self, clean_score_storage):
        """Test that pop/jazz styles are more lenient"""
        score = create_simple_score_with_parts()
        
        tool = HarmonizationTool(clean_score_storage)
        classical_quality = tool._check_voice_leading(score, "classical")
        pop_quality = tool._check_voice_leading(score, "pop")
        
        # Pop should be more lenient
        assert pop_quality["smoothness"] >= classical_quality["smoothness"]


class TestUtilityMethods:
    """Test utility methods"""
    
    def test_analyze_harmonic_rhythm(self, clean_score_storage):
        """Test harmonic rhythm analysis"""
        tool = HarmonizationTool(clean_score_storage)
        progression = ["I", "I", "V", "V", "vi", "vi", "IV", "I"]
        
        rhythm = tool._analyze_harmonic_rhythm(progression)
        
        assert "changes_per_measure" in rhythm
        assert "static_percentage" in rhythm
        assert "most_common_chord" in rhythm
        assert rhythm["most_common_chord"] == "I"

    def test_analyze_harmonic_rhythm_empty(self, clean_score_storage):
        """Test harmonic rhythm analysis with empty progression"""
        tool = HarmonizationTool(clean_score_storage)
        rhythm = tool._analyze_harmonic_rhythm([])
        assert rhythm == {}

    def test_generate_explanations(self, clean_score_storage):
        """Test explanation generation"""
        tool = HarmonizationTool(clean_score_storage)
        harmonization = {
            "progression": ["I", "IV", "V", "I"],
            "voice_leading_quality": {"smoothness": 0.8}
        }
        analysis = {"key": "C major"}
        
        explanations = tool._generate_explanations(harmonization, analysis, "classical")
        
        assert len(explanations) > 0
        assert all("aspect" in exp and "explanation" in exp for exp in explanations)

    def test_get_style_characteristics(self, clean_score_storage):
        """Test style characteristic descriptions"""
        tool = HarmonizationTool(clean_score_storage)
        
        for style in ["classical", "jazz", "pop", "modal"]:
            desc = tool._get_style_characteristics(style)
            assert isinstance(desc, str)
            assert len(desc) > 0

    def test_contains_progression(self, clean_score_storage):
        """Test progression pattern matching"""
        tool = HarmonizationTool(clean_score_storage)
        full_progression = ["I", "vi", "IV", "V", "I"]
        pattern = ["vi", "IV", "V"]
        
        assert tool._contains_progression(full_progression, pattern) is True
        assert tool._contains_progression(full_progression, ["ii", "V"]) is False


class TestIntegrationAndEndToEnd:
    """Test integration and end-to-end scenarios"""
    
    @pytest.mark.asyncio
    async def test_execute_complete_flow_classical(self, clean_score_storage):
        """Test complete execution flow for classical style"""
        melody = create_simple_melody()
        clean_score_storage["test_melody"] = melody
        
        tool = HarmonizationTool(clean_score_storage)
        result = await tool.execute(
            score_id="test_melody",
            output_id="harmonized_classical",
            style="classical",
            voice_parts=4,
            include_explanations=True
        )
        
        assert result["status"] == "success"
        assert "harmonized_classical" in clean_score_storage
        assert "harmonization" in result
        assert len(result["harmonization"]["explanations"]) > 0

    @pytest.mark.asyncio
    async def test_execute_complete_flow_all_styles(self, clean_score_storage):
        """Test complete execution flow for all styles"""
        melody = create_simple_melody()
        clean_score_storage["test_melody"] = melody
        
        tool = HarmonizationTool(clean_score_storage)
        
        for style in ["classical", "jazz", "pop", "modal"]:
            result = await tool.execute(
                score_id="test_melody",
                output_id=f"harmonized_{style}",
                style=style
            )
            
            assert result["status"] == "success"
            assert f"harmonized_{style}" in clean_score_storage

    @pytest.mark.asyncio
    async def test_execute_with_constraints(self, clean_score_storage):
        """Test execution with various constraints"""
        melody = create_simple_melody()
        clean_score_storage["test_melody"] = melody
        
        tool = HarmonizationTool(clean_score_storage)
        result = await tool.execute(
            score_id="test_melody",
            style="classical",
            constraints=["diatonic_only"]
        )
        
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_execute_empty_melody_error(self, clean_score_storage):
        """Test execution with empty melody"""
        empty_melody = stream.Stream()
        clean_score_storage["empty"] = empty_melody
        
        tool = HarmonizationTool(clean_score_storage)
        result = await tool.execute(score_id="empty", style="classical")
        
        assert result["status"] == "error"
        assert "No melody found" in result["message"]

    @pytest.mark.asyncio
    async def test_execute_invalid_style_error(self, clean_score_storage):
        """Test execution with invalid style"""
        melody = create_simple_melody()
        clean_score_storage["test_melody"] = melody
        
        tool = HarmonizationTool(clean_score_storage)
        result = await tool.execute(score_id="test_melody", style="invalid_style")
        
        assert result["status"] == "error"
        assert "Invalid style" in result["message"]


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases"""
    
    @pytest.mark.asyncio
    async def test_melody_analysis_error_handling(self, clean_score_storage):
        """Test melody analysis handles errors gracefully"""
        # Create problematic melody that might cause analysis errors
        problematic_melody = [note.Note("C4")]  # Very short melody
        
        tool = HarmonizationTool(clean_score_storage)
        analysis = await tool._analyze_melody(problematic_melody)
        
        # Should complete without throwing exceptions
        assert "key" in analysis
        assert "contour" in analysis

    def test_chord_realization_error_handling(self, clean_score_storage):
        """Test chord realization handles invalid chords"""
        tool = HarmonizationTool(clean_score_storage)
        key_obj = key.Key("C")
        note_obj = note.Note("C4")
        
        # Test with invalid chord symbol
        pitches = tool._realize_chord_classical("InvalidChord", key_obj, note_obj)
        assert len(pitches) >= 3  # Should return fallback triad

    def test_voice_leading_check_error_handling(self, clean_score_storage):
        """Test voice leading check handles malformed scores"""
        malformed_score = stream.Stream()
        
        tool = HarmonizationTool(clean_score_storage)
        quality = tool._check_voice_leading(malformed_score, "classical")
        
        # Should return default quality structure
        assert "smoothness" in quality
        assert "errors" in quality

    @pytest.mark.asyncio 
    async def test_harmonization_with_no_compatible_chords(self, clean_score_storage):
        """Test harmonization when no chords are compatible with melody"""
        # Create melody that might be hard to harmonize
        melody = stream.Stream()
        melody.append(note.Note("C#4", quarterLength=1))  # Chromatic note
        clean_score_storage["chromatic"] = melody
        
        tool = HarmonizationTool(clean_score_storage)
        result = await tool.execute(score_id="chromatic", style="classical")
        
        # Should complete successfully with fallback harmonies
        assert result["status"] == "success"


# Fixtures
@pytest.fixture
def clean_score_storage():
    """Provide clean score storage for each test"""
    return {}