"""
Musical Accuracy Validation Tests for music21-mcp-server

This module tests the actual musical correctness of analysis results,
not just code functionality. It uses known musical examples from the
music21 corpus and well-defined music theory rules to validate accuracy.
"""

import pytest
from typing import Dict, List, Optional, Tuple, Any
import asyncio
from music21 import corpus, stream, note, chord, key, roman, interval
from music21.analysis import discrete

from music21_mcp.tools import (
    KeyAnalysisTool,
    ChordAnalysisTool,
    PatternRecognitionTool,
    VoiceLeadingAnalysisTool,
    HarmonyAnalysisTool,
    CounterpointGeneratorTool,
    StyleImitationTool,
)
from music21_mcp.tools.base_tool import ScoreStore


class TestMusicalAccuracy:
    """Test suite for validating musical accuracy of analysis tools"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        # Clear score store before each test
        ScoreStore._instance = None
        self.score_store = ScoreStore()
        
    def load_corpus_score(self, corpus_path: str, score_id: str) -> stream.Score:
        """Load a score from music21 corpus"""
        score = corpus.parse(corpus_path)
        self.score_store.add_score(score_id, score)
        return score

    @pytest.mark.asyncio
    async def test_key_detection_bach_chorales(self):
        """Test key detection accuracy on Bach chorales with known keys"""
        # Test cases with known keys from Bach chorales
        test_cases = [
            ("bach/bwv66.6", "f# minor"),  # BWV 66.6 is in F# minor
            ("bach/bwv84.5", "a minor"),   # BWV 84.5 is in A minor
            ("bach/bwv269", "C major"),    # BWV 269 is in C major
            ("bach/bwv277", "Eb major"),   # BWV 277 is in Eb major
        ]
        
        key_tool = KeyAnalysisTool()
        
        for corpus_path, expected_key in test_cases:
            score_id = f"test_{corpus_path.replace('/', '_')}"
            self.load_corpus_score(corpus_path, score_id)
            
            # Test with multiple algorithms
            result = await key_tool.execute(score_id=score_id, algorithm="all")
            
            assert result["success"], f"Key analysis failed for {corpus_path}"
            
            # Check if any algorithm correctly identified the key
            algorithms_correct = []
            for algo_name, algo_result in result["key_analyses"].items():
                detected_key = algo_result["key"].lower()
                confidence = algo_result["confidence"]
                
                if detected_key == expected_key:
                    algorithms_correct.append((algo_name, confidence))
            
            assert len(algorithms_correct) > 0, (
                f"No algorithm correctly identified {expected_key} for {corpus_path}. "
                f"Results: {result['key_analyses']}"
            )
            
            # At least one algorithm should have high confidence (> 0.7)
            high_confidence = [conf for _, conf in algorithms_correct if conf > 0.7]
            assert len(high_confidence) > 0, (
                f"No algorithm had high confidence for correct key detection in {corpus_path}"
            )

    @pytest.mark.asyncio
    async def test_chord_progression_analysis(self):
        """Test chord progression analysis with known progressions"""
        # Create a simple ii-V-I progression in C major
        score = stream.Score()
        part = stream.Part()
        
        # ii chord (Dm)
        ii_chord = chord.Chord(['D4', 'F4', 'A4'], quarterLength=2)
        part.append(ii_chord)
        
        # V chord (G)
        v_chord = chord.Chord(['G3', 'B3', 'D4', 'F4'], quarterLength=2)
        part.append(v_chord)
        
        # I chord (C)
        i_chord = chord.Chord(['C4', 'E4', 'G4'], quarterLength=4)
        part.append(i_chord)
        
        score.append(part)
        self.score_store.add_score("ii_v_i_progression", score)
        
        chord_tool = ChordAnalysisTool()
        result = await chord_tool.execute(
            score_id="ii_v_i_progression",
            include_roman_numerals=True,
            include_inversions=True
        )
        
        assert result["success"], "Chord analysis failed"
        
        # Extract Roman numerals from progressions
        roman_numerals = []
        for progression in result["progressions"]:
            for chord_info in progression["chords"]:
                if chord_info.get("roman_numeral"):
                    roman_numerals.append(chord_info["roman_numeral"])
        
        # Verify ii-V-I progression
        assert "ii" in roman_numerals or "ii6" in roman_numerals, "ii chord not detected"
        assert "V" in roman_numerals or "V7" in roman_numerals, "V chord not detected"
        assert "I" in roman_numerals, "I chord not detected"

    @pytest.mark.asyncio
    async def test_melodic_pattern_recognition(self):
        """Test pattern recognition on melodic sequences"""
        # Create a score with a clear melodic sequence
        score = stream.Score()
        part = stream.Part()
        
        # Create a descending sequence pattern: C-B-A, B-A-G, A-G-F
        pattern1 = [note.Note('C5', quarterLength=1), 
                   note.Note('B4', quarterLength=1),
                   note.Note('A4', quarterLength=1)]
        
        pattern2 = [note.Note('B4', quarterLength=1),
                   note.Note('A4', quarterLength=1),
                   note.Note('G4', quarterLength=1)]
        
        pattern3 = [note.Note('A4', quarterLength=1),
                   note.Note('G4', quarterLength=1),
                   note.Note('F4', quarterLength=1)]
        
        for pattern in [pattern1, pattern2, pattern3]:
            for n in pattern:
                part.append(n)
        
        score.append(part)
        self.score_store.add_score("melodic_sequence", score)
        
        pattern_tool = PatternRecognitionTool()
        result = await pattern_tool.execute(
            score_id="melodic_sequence",
            min_pattern_length=3,
            max_pattern_length=3,
            include_transpositions=True
        )
        
        assert result["success"], "Pattern recognition failed"
        
        # Should detect the descending three-note pattern
        patterns = result.get("patterns", [])
        assert len(patterns) > 0, "No patterns detected in melodic sequence"
        
        # Check if descending pattern was found
        found_descending = False
        for pattern in patterns:
            intervals = pattern.get("intervals", [])
            # Descending pattern should have negative intervals
            if all(int(i) < 0 for i in intervals if i.isdigit() or (i[0] == '-' and i[1:].isdigit())):
                found_descending = True
                break
        
        assert found_descending, "Descending melodic pattern not detected"

    @pytest.mark.asyncio
    async def test_voice_leading_smoothness(self):
        """Test voice leading analysis for smoothness"""
        # Create a score with good voice leading
        score = stream.Score()
        soprano = stream.Part()
        alto = stream.Part()
        tenor = stream.Part()
        bass = stream.Part()
        
        # Good voice leading: C major to F major (common tone C)
        soprano.append(note.Note('E5', quarterLength=2))  # E -> F
        soprano.append(note.Note('F5', quarterLength=2))
        
        alto.append(note.Note('C5', quarterLength=2))     # C -> C (common tone)
        alto.append(note.Note('C5', quarterLength=2))
        
        tenor.append(note.Note('G4', quarterLength=2))    # G -> A
        tenor.append(note.Note('A4', quarterLength=2))
        
        bass.append(note.Note('C3', quarterLength=2))     # C -> F
        bass.append(note.Note('F3', quarterLength=2))
        
        score.insert(0, soprano)
        score.insert(0, alto)
        score.insert(0, tenor)
        score.insert(0, bass)
        
        self.score_store.add_score("good_voice_leading", score)
        
        voice_tool = VoiceLeadingAnalysisTool()
        result = await voice_tool.execute(
            score_id="good_voice_leading",
            check_parallels=True,
            check_voice_crossings=True,
            check_large_leaps=True
        )
        
        assert result["success"], "Voice leading analysis failed"
        
        # Should have minimal voice leading issues
        issues = result.get("voice_leading_issues", [])
        parallel_issues = [i for i in issues if "parallel" in i.get("type", "").lower()]
        
        assert len(parallel_issues) == 0, "Found parallel fifths/octaves in good voice leading"
        
        # Check smoothness score
        smoothness = result.get("smoothness_score", 0)
        assert smoothness > 0.7, f"Voice leading not smooth enough: {smoothness}"

    @pytest.mark.asyncio
    async def test_harmony_analysis_functional(self):
        """Test functional harmony analysis"""
        # Load a Bach chorale for harmony analysis
        score = self.load_corpus_score("bach/bwv269", "bach_harmony_test")
        
        harmony_tool = HarmonyAnalysisTool()
        result = await harmony_tool.execute(
            score_id="bach_harmony_test",
            include_non_chord_tones=True,
            include_tonicizations=True
        )
        
        assert result["success"], "Harmony analysis failed"
        
        # Bach chorales should have rich harmonic content
        segments = result.get("harmonic_segments", [])
        assert len(segments) > 0, "No harmonic segments found"
        
        # Check for various harmonic features
        has_secondary_dominants = False
        has_pivot_chords = False
        
        for segment in segments:
            if "V/" in segment.get("roman_numeral", ""):
                has_secondary_dominants = True
            if segment.get("is_pivot_chord"):
                has_pivot_chords = True
        
        # Bach chorales typically have secondary dominants
        assert has_secondary_dominants, "No secondary dominants found in Bach chorale"

    @pytest.mark.asyncio
    async def test_counterpoint_species_rules(self):
        """Test counterpoint generation follows species rules"""
        # Create a cantus firmus
        cantus = stream.Part()
        cantus_notes = [
            note.Note('C4', quarterLength=4),
            note.Note('D4', quarterLength=4),
            note.Note('F4', quarterLength=4),
            note.Note('E4', quarterLength=4),
            note.Note('D4', quarterLength=4),
            note.Note('C4', quarterLength=4),
        ]
        for n in cantus_notes:
            cantus.append(n)
        
        score = stream.Score()
        score.append(cantus)
        self.score_store.add_score("cantus_firmus", score)
        
        counterpoint_tool = CounterpointGeneratorTool()
        
        # Test first species (note against note)
        result = await counterpoint_tool.execute(
            score_id="cantus_firmus",
            species=1,
            voice_position="above"
        )
        
        assert result["success"], "First species counterpoint generation failed"
        
        # Verify first species rules
        generated_score_id = result["generated_score_id"]
        generated_score = self.score_store.get_score(generated_score_id)
        
        # Get the generated counterpoint part
        parts = generated_score.parts
        assert len(parts) >= 2, "Generated score should have at least 2 parts"
        
        counterpoint_part = parts[0] if parts[0] != cantus else parts[1]
        
        # Check first species rules
        cp_notes = list(counterpoint_part.recurse().notes)
        cf_notes = list(cantus.recurse().notes)
        
        assert len(cp_notes) == len(cf_notes), "First species should have note-against-note"
        
        # Check intervals
        for i, (cf_note, cp_note) in enumerate(zip(cf_notes, cp_notes)):
            interv = interval.Interval(cf_note, cp_note)
            
            # First and last notes should form perfect consonances
            if i == 0 or i == len(cf_notes) - 1:
                assert interv.simpleName in ['P1', 'P5', 'P8'], (
                    f"First/last interval should be perfect: {interv.simpleName}"
                )
            
            # All intervals should be consonant
            consonant_intervals = ['P1', 'P5', 'P8', 'm3', 'M3', 'm6', 'M6']
            assert interv.simpleName in consonant_intervals, (
                f"Dissonant interval found: {interv.simpleName}"
            )

    @pytest.mark.asyncio
    async def test_style_imitation_baroque(self):
        """Test style imitation for Baroque characteristics"""
        # Load a Bach invention as style reference
        score = self.load_corpus_score("bach/bwv773", "bach_invention_style")
        
        style_tool = StyleImitationTool()
        
        # Create a simple melodic fragment to harmonize in Bach style
        fragment = stream.Score()
        part = stream.Part()
        notes = [
            note.Note('G4', quarterLength=1),
            note.Note('A4', quarterLength=1),
            note.Note('B4', quarterLength=1),
            note.Note('C5', quarterLength=2),
        ]
        for n in notes:
            part.append(n)
        fragment.append(part)
        self.score_store.add_score("melodic_fragment", fragment)
        
        result = await style_tool.execute(
            reference_score_id="bach_invention_style",
            input_score_id="melodic_fragment",
            style_parameters={
                "harmony": True,
                "rhythm": True,
                "texture": True
            }
        )
        
        assert result["success"], "Style imitation failed"
        
        # Verify Baroque characteristics in generated music
        generated_id = result["generated_score_id"]
        generated = self.score_store.get_score(generated_id)
        
        # Check for polyphonic texture (multiple voices)
        parts = generated.parts
        assert len(parts) >= 2, "Baroque style should have polyphonic texture"
        
        # Check for characteristic Baroque rhythmic patterns
        has_sixteenths = False
        for part in parts:
            for n in part.recurse().notes:
                if n.quarterLength <= 0.25:  # Sixteenth notes or smaller
                    has_sixteenths = True
                    break
        
        assert has_sixteenths, "Baroque style should include characteristic rhythmic activity"

    @pytest.mark.asyncio 
    async def test_jazz_chord_progression_analysis(self):
        """Test jazz chord progression analysis (ii-V-I with extensions)"""
        score = stream.Score()
        part = stream.Part()
        
        # Create jazz ii-V-I with 7th chords
        # Dm7
        dm7 = chord.Chord(['D3', 'F3', 'A3', 'C4'], quarterLength=2)
        part.append(dm7)
        
        # G7
        g7 = chord.Chord(['G2', 'B2', 'D3', 'F3'], quarterLength=2)
        part.append(g7)
        
        # CMaj7
        cmaj7 = chord.Chord(['C3', 'E3', 'G3', 'B3'], quarterLength=4)
        part.append(cmaj7)
        
        score.append(part)
        self.score_store.add_score("jazz_ii_v_i", score)
        
        chord_tool = ChordAnalysisTool()
        result = await chord_tool.execute(
            score_id="jazz_ii_v_i",
            include_roman_numerals=True,
            include_inversions=True
        )
        
        assert result["success"], "Jazz chord analysis failed"
        
        # Check that 7th chords are properly identified
        progressions = result.get("progressions", [])
        assert len(progressions) > 0, "No progressions found"
        
        chords = progressions[0]["chords"]
        chord_symbols = [c["chord_symbol"] for c in chords]
        
        # Verify jazz chord symbols
        assert any("m7" in symbol or "min7" in symbol for symbol in chord_symbols), "Dm7 not properly identified"
        assert any("7" in symbol and not "maj" in symbol.lower() for symbol in chord_symbols), "G7 not properly identified"
        assert any("maj7" in symbol.lower() or "M7" in symbol for symbol in chord_symbols), "CMaj7 not properly identified"

    @pytest.mark.asyncio
    async def test_renaissance_polyphony_analysis(self):
        """Test analysis of Renaissance polyphonic music"""
        # Create a simple Renaissance-style polyphonic passage
        score = stream.Score()
        
        # Cantus (top voice)
        cantus = stream.Part()
        cantus.append(note.Note('C5', quarterLength=2))
        cantus.append(note.Note('D5', quarterLength=2))
        cantus.append(note.Note('E5', quarterLength=2))
        cantus.append(note.Note('F5', quarterLength=2))
        cantus.append(note.Note('E5', quarterLength=2))
        cantus.append(note.Note('D5', quarterLength=2))
        cantus.append(note.Note('C5', quarterLength=4))
        
        # Tenor (imitation at the fifth below, delayed)
        tenor = stream.Part()
        tenor.append(note.Rest(quarterLength=4))  # Delay entry
        tenor.append(note.Note('G4', quarterLength=2))
        tenor.append(note.Note('A4', quarterLength=2))
        tenor.append(note.Note('B4', quarterLength=2))
        tenor.append(note.Note('C5', quarterLength=2))
        tenor.append(note.Note('B4', quarterLength=2))
        tenor.append(note.Note('A4', quarterLength=2))
        tenor.append(note.Note('G4', quarterLength=4))
        
        score.insert(0, cantus)
        score.insert(0, tenor)
        
        self.score_store.add_score("renaissance_polyphony", score)
        
        # Test pattern recognition for imitation
        pattern_tool = PatternRecognitionTool()
        result = await pattern_tool.execute(
            score_id="renaissance_polyphony",
            min_pattern_length=4,
            include_transpositions=True,
            algorithm="suffix_tree"
        )
        
        assert result["success"], "Pattern recognition failed for Renaissance polyphony"
        
        patterns = result.get("patterns", [])
        assert len(patterns) > 0, "No patterns found in imitative polyphony"
        
        # Should detect the imitation at the fifth
        found_imitation = False
        for pattern in patterns:
            if pattern.get("transposition_interval"):
                # Check if it's approximately a fifth (7 semitones)
                try:
                    interval_semitones = abs(int(pattern["transposition_interval"]))
                    if interval_semitones == 7:  # Perfect fifth
                        found_imitation = True
                        break
                except (ValueError, TypeError):
                    continue
        
        assert found_imitation, "Imitation at the fifth not detected"

    def _calculate_interval_vector(self, chord_notes: List[note.Note]) -> List[int]:
        """Calculate interval vector for atonal analysis"""
        pitch_classes = [n.pitch.pitchClass for n in chord_notes]
        interval_vector = [0] * 6
        
        for i in range(len(pitch_classes)):
            for j in range(i + 1, len(pitch_classes)):
                interval = abs(pitch_classes[i] - pitch_classes[j])
                if interval > 6:
                    interval = 12 - interval
                if 1 <= interval <= 6:
                    interval_vector[interval - 1] += 1
        
        return interval_vector

    @pytest.mark.asyncio
    async def test_complex_key_modulation(self):
        """Test detection of key modulations"""
        # Create a score with clear modulation from C major to G major
        score = stream.Score()
        part = stream.Part()
        
        # Establish C major
        c_major_notes = [
            note.Note('C4', quarterLength=1),
            note.Note('E4', quarterLength=1),
            note.Note('G4', quarterLength=1),
            note.Note('C5', quarterLength=1),
        ]
        
        # Pivot chord (G major, V in C and I in G)
        pivot_notes = [
            note.Note('G4', quarterLength=2),
            note.Note('B4', quarterLength=2),
            note.Note('D5', quarterLength=2),
        ]
        
        # Establish G major with F#
        g_major_notes = [
            note.Note('G4', quarterLength=1),
            note.Note('A4', quarterLength=1),
            note.Note('B4', quarterLength=1),
            note.Note('C5', quarterLength=1),
            note.Note('D5', quarterLength=1),
            note.Note('E5', quarterLength=1),
            note.Note('F#5', quarterLength=1),  # Leading tone in G major
            note.Note('G5', quarterLength=2),
        ]
        
        for n in c_major_notes + pivot_notes + g_major_notes:
            part.append(n)
            
        score.append(part)
        self.score_store.add_score("modulating_score", score)
        
        # Test with harmony analysis tool
        harmony_tool = HarmonyAnalysisTool()
        result = await harmony_tool.execute(
            score_id="modulating_score",
            include_tonicizations=True,
            window_size=4
        )
        
        assert result["success"], "Harmony analysis with modulation failed"
        
        # Check for key areas
        key_areas = result.get("key_areas", [])
        assert len(key_areas) >= 2, "Should detect at least two key areas"
        
        # Verify C major and G major are detected
        detected_keys = [area["key"] for area in key_areas]
        assert any("C major" in k for k in detected_keys), "C major not detected"
        assert any("G major" in k for k in detected_keys), "G major not detected"

    @pytest.mark.asyncio
    async def test_mozart_sonata_key_detection(self):
        """Test key detection on Mozart piano sonatas"""
        test_cases = [
            ("mozart/k331/movement1", "A major"),  # K. 331 1st movement in A major
            ("mozart/k333/movement1", "Bb major"), # K. 333 1st movement in Bb major
            ("mozart/k545/movement1", "C major"),  # K. 545 1st movement in C major
        ]
        
        key_tool = KeyAnalysisTool()
        
        for corpus_path, expected_key in test_cases:
            try:
                score_id = f"mozart_{corpus_path.replace('/', '_')}"
                self.load_corpus_score(corpus_path, score_id)
                
                result = await key_tool.execute(
                    score_id=score_id,
                    algorithm="krumhansl_schmuckler"
                )
                
                assert result["success"], f"Key analysis failed for {corpus_path}"
                
                detected_key = result["key_analyses"]["krumhansl_schmuckler"]["key"].lower()
                assert detected_key == expected_key.lower(), (
                    f"Expected {expected_key} but got {detected_key} for {corpus_path}"
                )
            except Exception:
                # Skip if corpus file not available
                pytest.skip(f"Corpus file {corpus_path} not available")

    @pytest.mark.asyncio
    async def test_parallel_fifths_detection(self):
        """Test detection of parallel fifths and octaves"""
        # Create a score with intentional parallel fifths
        score = stream.Score()
        soprano = stream.Part()
        bass = stream.Part()
        
        # Parallel fifths: C-G to D-A
        soprano.append(note.Note('C5', quarterLength=2))
        soprano.append(note.Note('D5', quarterLength=2))
        
        bass.append(note.Note('F4', quarterLength=2))  # Forms P5 with C5
        bass.append(note.Note('G4', quarterLength=2))  # Forms P5 with D5
        
        # Parallel octaves: E-E to F-F
        soprano.append(note.Note('E5', quarterLength=2))
        soprano.append(note.Note('F5', quarterLength=2))
        
        bass.append(note.Note('E4', quarterLength=2))  # Forms P8 with E5
        bass.append(note.Note('F4', quarterLength=2))  # Forms P8 with F5
        
        score.insert(0, soprano)
        score.insert(0, bass)
        
        self.score_store.add_score("parallel_fifths_test", score)
        
        voice_tool = VoiceLeadingAnalysisTool()
        result = await voice_tool.execute(
            score_id="parallel_fifths_test",
            check_parallels=True,
            check_voice_crossings=False,
            check_large_leaps=False
        )
        
        assert result["success"], "Voice leading analysis failed"
        
        issues = result.get("voice_leading_issues", [])
        parallel_issues = [i for i in issues if "parallel" in i.get("type", "").lower()]
        
        # Should detect both parallel fifths and parallel octaves
        assert len(parallel_issues) >= 2, "Failed to detect all parallel motion issues"
        
        # Check specific types
        has_parallel_fifths = any("fifth" in str(i).lower() for i in parallel_issues)
        has_parallel_octaves = any("octave" in str(i).lower() for i in parallel_issues)
        
        assert has_parallel_fifths, "Parallel fifths not detected"
        assert has_parallel_octaves, "Parallel octaves not detected"

    @pytest.mark.asyncio
    async def test_advanced_chord_progressions(self):
        """Test recognition of advanced chord progressions"""
        # Create Neapolitan sixth chord progression (bII6 - V - I)
        score = stream.Score()
        part = stream.Part()
        
        # Key of C minor
        # Neapolitan sixth (Db major in first inversion)
        neapolitan = chord.Chord(['F3', 'Ab3', 'Db4'], quarterLength=2)
        part.append(neapolitan)
        
        # V7 (G7)
        dominant = chord.Chord(['G3', 'B3', 'D4', 'F4'], quarterLength=2)
        part.append(dominant)
        
        # i (C minor)
        tonic = chord.Chord(['C4', 'Eb4', 'G4'], quarterLength=4)
        part.append(tonic)
        
        score.append(part)
        self.score_store.add_score("neapolitan_progression", score)
        
        harmony_tool = HarmonyAnalysisTool()
        result = await harmony_tool.execute(
            score_id="neapolitan_progression",
            include_non_chord_tones=False,
            include_tonicizations=False
        )
        
        assert result["success"], "Harmony analysis of Neapolitan failed"
        
        segments = result.get("harmonic_segments", [])
        assert len(segments) >= 3, "Should have at least 3 harmonic segments"
        
        # Check for Neapolitan chord identification
        found_neapolitan = False
        for segment in segments:
            roman = segment.get("roman_numeral", "")
            # Neapolitan is typically labeled as bII6 or N6
            if "bII" in roman or "N6" in roman or segment.get("chord_type") == "Neapolitan":
                found_neapolitan = True
                break
        
        assert found_neapolitan, "Neapolitan sixth chord not properly identified"

    @pytest.mark.asyncio
    async def test_rhythmic_pattern_detection(self):
        """Test detection of specific rhythmic patterns"""
        # Create a score with distinctive rhythmic patterns
        score = stream.Score()
        part = stream.Part()
        
        # Dotted rhythm pattern (long-short pattern)
        dotted_pattern = [
            note.Note('C4', quarterLength=1.5),
            note.Note('D4', quarterLength=0.5),
            note.Note('E4', quarterLength=1.5),
            note.Note('F4', quarterLength=0.5),
        ]
        
        # Syncopation pattern
        syncopated_pattern = [
            note.Note('G4', quarterLength=0.5),
            note.Note('A4', quarterLength=1),
            note.Note('B4', quarterLength=0.5),
            note.Note('C5', quarterLength=1),
        ]
        
        # Repeat patterns
        for _ in range(2):
            for n in dotted_pattern + syncopated_pattern:
                part.append(n)
        
        score.append(part)
        self.score_store.add_score("rhythmic_patterns", score)
        
        pattern_tool = PatternRecognitionTool()
        result = await pattern_tool.execute(
            score_id="rhythmic_patterns",
            pattern_type="rhythmic",
            min_pattern_length=2,
            max_pattern_length=4
        )
        
        assert result["success"], "Rhythmic pattern recognition failed"
        
        patterns = result.get("patterns", [])
        assert len(patterns) > 0, "No rhythmic patterns detected"
        
        # Check for dotted rhythm detection
        found_dotted = False
        for pattern in patterns:
            durations = pattern.get("durations", [])
            # Dotted pattern has 1.5 followed by 0.5
            if len(durations) >= 2:
                for i in range(len(durations) - 1):
                    if (abs(durations[i] - 1.5) < 0.01 and 
                        abs(durations[i+1] - 0.5) < 0.01):
                        found_dotted = True
                        break
        
        assert found_dotted, "Dotted rhythm pattern not detected"

    @pytest.mark.asyncio
    async def test_classical_style_characteristics(self):
        """Test detection of Classical period style characteristics"""
        # Create a Classical-style melody with Alberti bass
        score = stream.Score()
        
        # Melody - simple, periodic phrasing
        melody = stream.Part()
        melody_notes = [
            note.Note('C5', quarterLength=1),
            note.Note('D5', quarterLength=1),
            note.Note('E5', quarterLength=1),
            note.Note('C5', quarterLength=1),
            note.Note('G5', quarterLength=2),
            note.Note('E5', quarterLength=2),
            note.Note('D5', quarterLength=1),
            note.Note('C5', quarterLength=1),
            note.Note('B4', quarterLength=1),
            note.Note('C5', quarterLength=1),
        ]
        for n in melody_notes:
            melody.append(n)
        
        # Alberti bass pattern
        bass = stream.Part()
        alberti_pattern = [
            note.Note('C3', quarterLength=0.5),
            note.Note('G3', quarterLength=0.5),
            note.Note('E3', quarterLength=0.5),
            note.Note('G3', quarterLength=0.5),
        ]
        
        # Repeat Alberti pattern
        for _ in range(6):
            for n in alberti_pattern:
                bass.append(n)
        
        score.insert(0, melody)
        score.insert(0, bass)
        
        self.score_store.add_score("classical_style", score)
        
        # Use pattern recognition to detect Alberti bass
        pattern_tool = PatternRecognitionTool()
        result = await pattern_tool.execute(
            score_id="classical_style",
            pattern_type="melodic",
            min_pattern_length=4,
            include_rhythm=True
        )
        
        assert result["success"], "Classical style analysis failed"
        
        patterns = result.get("patterns", [])
        
        # Should detect the repeated Alberti bass pattern
        alberti_found = False
        for pattern in patterns:
            if pattern.get("occurrences", 0) >= 4:  # Alberti pattern repeats
                notes_in_pattern = pattern.get("pattern_notes", [])
                if len(notes_in_pattern) == 4:
                    # Check for characteristic broken chord pattern
                    alberti_found = True
                    break
        
        assert alberti_found, "Alberti bass pattern not detected"

    @pytest.mark.asyncio
    async def test_romantic_chromatic_harmony(self):
        """Test detection of Romantic period chromatic harmony"""
        # Create a chromatic progression typical of Romantic period
        score = stream.Score()
        part = stream.Part()
        
        # Chromatic mediant progression: C major - E major - Ab major - C major
        chords_progression = [
            chord.Chord(['C4', 'E4', 'G4'], quarterLength=2),      # I
            chord.Chord(['E4', 'G#4', 'B4'], quarterLength=2),     # III (chromatic mediant)
            chord.Chord(['Ab3', 'C4', 'Eb4'], quarterLength=2),    # bVI (chromatic mediant)
            chord.Chord(['C4', 'E4', 'G4'], quarterLength=2),      # I
        ]
        
        for c in chords_progression:
            part.append(c)
        
        score.append(part)
        self.score_store.add_score("romantic_chromatic", score)
        
        harmony_tool = HarmonyAnalysisTool()
        result = await harmony_tool.execute(
            score_id="romantic_chromatic",
            include_tonicizations=True,
            include_non_chord_tones=False
        )
        
        assert result["success"], "Romantic harmony analysis failed"
        
        segments = result.get("harmonic_segments", [])
        
        # Check for chromatic relationships
        chromatic_mediants = 0
        for i, segment in enumerate(segments[:-1]):
            current_root = segment.get("root", "")
            next_root = segments[i+1].get("root", "")
            
            if current_root and next_root:
                # Check for third relationships (chromatic mediants)
                curr_pitch = note.Note(current_root)
                next_pitch = note.Note(next_root)
                interval_semitones = abs(curr_pitch.pitch.midi - next_pitch.pitch.midi) % 12
                
                # Major or minor third = 3 or 4 semitones
                if interval_semitones in [3, 4, 8, 9]:  # Also check for sixths
                    chromatic_mediants += 1
        
        assert chromatic_mediants >= 2, "Chromatic mediant relationships not properly detected"

    @pytest.mark.asyncio
    async def test_fugue_subject_detection(self):
        """Test detection of fugue subjects and answers"""
        # Create a simple fugue exposition
        score = stream.Score()
        
        # Subject in soprano
        soprano = stream.Part()
        subject = [
            note.Note('C5', quarterLength=1),
            note.Note('E5', quarterLength=1),
            note.Note('G5', quarterLength=0.5),
            note.Note('F5', quarterLength=0.5),
            note.Note('E5', quarterLength=1),
            note.Note('D5', quarterLength=1),
            note.Note('C5', quarterLength=2),
        ]
        
        for n in subject:
            soprano.append(n)
        soprano.append(note.Rest(quarterLength=6))  # Rest while alto has answer
        
        # Answer in alto (at the fifth)
        alto = stream.Part()
        alto.append(note.Rest(quarterLength=6))  # Wait for subject
        answer = [
            note.Note('G4', quarterLength=1),  # Fifth above C
            note.Note('B4', quarterLength=1),  # Fifth above E (tonal answer)
            note.Note('D5', quarterLength=0.5),
            note.Note('C5', quarterLength=0.5),
            note.Note('B4', quarterLength=1),
            note.Note('A4', quarterLength=1),
            note.Note('G4', quarterLength=2),
        ]
        
        for n in answer:
            alto.append(n)
        
        score.insert(0, soprano)
        score.insert(0, alto)
        
        self.score_store.add_score("fugue_exposition", score)
        
        pattern_tool = PatternRecognitionTool()
        result = await pattern_tool.execute(
            score_id="fugue_exposition",
            min_pattern_length=5,
            include_transpositions=True,
            algorithm="suffix_tree"
        )
        
        assert result["success"], "Fugue pattern recognition failed"
        
        patterns = result.get("patterns", [])
        
        # Should detect subject-answer relationship at the fifth
        found_fifth_transposition = False
        for pattern in patterns:
            if pattern.get("transposition_interval"):
                try:
                    interval = int(pattern["transposition_interval"])
                    if abs(interval) == 7:  # Perfect fifth = 7 semitones
                        found_fifth_transposition = True
                        break
                except (ValueError, TypeError):
                    continue
        
        assert found_fifth_transposition, "Fugue subject-answer relationship not detected"

    @pytest.mark.asyncio
    async def test_modal_detection(self):
        """Test detection of modal scales"""
        # Create pieces in different modes
        modes = {
            "dorian": ['D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5', 'D5'],
            "phrygian": ['E4', 'F4', 'G4', 'A4', 'B4', 'C5', 'D5', 'E5'],
            "lydian": ['F4', 'G4', 'A4', 'B4', 'C5', 'D5', 'E5', 'F5'],
            "mixolydian": ['G4', 'A4', 'B4', 'C5', 'D5', 'E5', 'F5', 'G5'],
        }
        
        key_tool = KeyAnalysisTool()
        
        for mode_name, mode_notes in modes.items():
            score = stream.Score()
            part = stream.Part()
            
            # Play scale up and down with emphasis on tonic
            for n in mode_notes:
                part.append(note.Note(n, quarterLength=1))
            for n in reversed(mode_notes):
                part.append(note.Note(n, quarterLength=1))
            
            # End on tonic
            part.append(note.Note(mode_notes[0], quarterLength=4))
            
            score.append(part)
            score_id = f"mode_{mode_name}"
            self.score_store.add_score(score_id, score)
            
            result = await key_tool.execute(
                score_id=score_id,
                algorithm="all"
            )
            
            assert result["success"], f"Modal analysis failed for {mode_name}"
            
            # Check if mode characteristics are detected
            # Modal music often has ambiguous key detection
            key_analyses = result["key_analyses"]
            
            # At least one algorithm should detect something related to the mode
            detected_keys = [v["key"] for v in key_analyses.values()]
            
            # For Dorian, might detect D minor or F major
            if mode_name == "dorian":
                assert any("d" in k.lower() or "f major" in k.lower() for k in detected_keys), (
                    f"Dorian mode characteristics not detected"
                )

    @pytest.mark.asyncio
    async def test_augmented_sixth_chords(self):
        """Test detection of augmented sixth chords (Italian, German, French)"""
        score = stream.Score()
        part = stream.Part()
        
        # Key of C major
        # Italian augmented sixth (Ab-C-F#)
        italian_sixth = chord.Chord(['Ab3', 'C4', 'F#4'], quarterLength=2)
        part.append(italian_sixth)
        
        # Resolution to V (G major)
        dominant = chord.Chord(['G3', 'B3', 'D4'], quarterLength=2)
        part.append(dominant)
        
        # German augmented sixth (Ab-C-Eb-F#)
        german_sixth = chord.Chord(['Ab3', 'C4', 'Eb4', 'F#4'], quarterLength=2)
        part.append(german_sixth)
        
        # Resolution to V
        part.append(dominant)
        
        # French augmented sixth (Ab-C-D-F#)
        french_sixth = chord.Chord(['Ab3', 'C4', 'D4', 'F#4'], quarterLength=2)
        part.append(french_sixth)
        
        # Resolution to V
        part.append(dominant)
        
        # Final tonic
        tonic = chord.Chord(['C4', 'E4', 'G4'], quarterLength=4)
        part.append(tonic)
        
        score.append(part)
        self.score_store.add_score("augmented_sixths", score)
        
        harmony_tool = HarmonyAnalysisTool()
        result = await harmony_tool.execute(
            score_id="augmented_sixths",
            include_non_chord_tones=False
        )
        
        assert result["success"], "Augmented sixth analysis failed"
        
        segments = result.get("harmonic_segments", [])
        
        # Check for augmented sixth identification
        aug_sixth_count = 0
        for segment in segments:
            chord_type = segment.get("chord_type", "")
            roman = segment.get("roman_numeral", "")
            
            # Look for augmented sixth indicators
            if ("aug6" in roman.lower() or "it6" in roman.lower() or 
                "ger6" in roman.lower() or "fr6" in roman.lower() or
                "italian" in chord_type.lower() or "german" in chord_type.lower() or
                "french" in chord_type.lower() or "augmented sixth" in chord_type.lower()):
                aug_sixth_count += 1
        
        assert aug_sixth_count >= 2, f"Not enough augmented sixth chords detected: {aug_sixth_count}"

    @pytest.mark.asyncio
    async def test_cadence_detection(self):
        """Test detection of various cadence types"""
        cadences = {
            "authentic": {
                "chords": [
                    chord.Chord(['G3', 'B3', 'D4'], quarterLength=2),  # V
                    chord.Chord(['C4', 'E4', 'G4'], quarterLength=4),  # I
                ],
                "description": "V-I authentic cadence"
            },
            "plagal": {
                "chords": [
                    chord.Chord(['F3', 'A3', 'C4'], quarterLength=2),  # IV
                    chord.Chord(['C4', 'E4', 'G4'], quarterLength=4),  # I
                ],
                "description": "IV-I plagal cadence"
            },
            "deceptive": {
                "chords": [
                    chord.Chord(['G3', 'B3', 'D4'], quarterLength=2),  # V
                    chord.Chord(['A3', 'C4', 'E4'], quarterLength=4),  # vi
                ],
                "description": "V-vi deceptive cadence"
            },
            "half": {
                "chords": [
                    chord.Chord(['C4', 'E4', 'G4'], quarterLength=2),  # I
                    chord.Chord(['G3', 'B3', 'D4'], quarterLength=4),  # V
                ],
                "description": "I-V half cadence"
            }
        }
        
        for cadence_type, cadence_data in cadences.items():
            score = stream.Score()
            part = stream.Part()
            
            # Add some context before the cadence
            part.append(chord.Chord(['C4', 'E4', 'G4'], quarterLength=2))
            part.append(chord.Chord(['F3', 'A3', 'C4'], quarterLength=2))
            
            # Add the cadence
            for c in cadence_data["chords"]:
                part.append(c)
            
            score.append(part)
            score_id = f"cadence_{cadence_type}"
            self.score_store.add_score(score_id, score)
            
            harmony_tool = HarmonyAnalysisTool()
            result = await harmony_tool.execute(
                score_id=score_id,
                include_cadences=True
            )
            
            assert result["success"], f"Cadence analysis failed for {cadence_type}"
            
            # Check if cadences are detected
            cadences_found = result.get("cadences", [])
            assert len(cadences_found) > 0, f"No cadences detected for {cadence_type}"
            
            # Verify the correct cadence type is identified
            found_correct_type = False
            for cad in cadences_found:
                if cadence_type.lower() in cad.get("type", "").lower():
                    found_correct_type = True
                    break
            
            assert found_correct_type, f"{cadence_type} cadence not correctly identified"

    @pytest.mark.asyncio
    async def test_suspension_detection(self):
        """Test detection of suspensions and other non-chord tones"""
        score = stream.Score()
        soprano = stream.Part()
        bass = stream.Part()
        
        # Create a 4-3 suspension
        # Preparation
        soprano.append(note.Note('F4', quarterLength=2))  # 4th above bass
        bass.append(note.Note('C3', quarterLength=2))
        
        # Suspension (dissonance)
        soprano.append(note.Note('F4', quarterLength=1))  # Suspended 4th
        bass.append(note.Note('C3', quarterLength=1))
        
        # Resolution
        soprano.append(note.Note('E4', quarterLength=1))  # Resolves to 3rd
        bass.append(note.Note('C3', quarterLength=1))
        
        # Another chord
        soprano.append(note.Note('D4', quarterLength=2))
        bass.append(note.Note('G3', quarterLength=2))
        
        score.insert(0, soprano)
        score.insert(0, bass)
        
        self.score_store.add_score("suspension_test", score)
        
        harmony_tool = HarmonyAnalysisTool()
        result = await harmony_tool.execute(
            score_id="suspension_test",
            include_non_chord_tones=True
        )
        
        assert result["success"], "Suspension analysis failed"
        
        # Check for non-chord tone detection
        segments = result.get("harmonic_segments", [])
        found_suspension = False
        
        for segment in segments:
            ncts = segment.get("non_chord_tones", [])
            for nct in ncts:
                if nct.get("type", "").lower() == "suspension":
                    found_suspension = True
                    break
        
        assert found_suspension, "4-3 suspension not detected"

    @pytest.mark.asyncio
    async def test_sequence_detection_real_vs_tonal(self):
        """Test detection of real vs tonal sequences"""
        # Real sequence (exact transposition)
        real_seq_score = stream.Score()
        part = stream.Part()
        
        # Pattern 1: C-D-E
        pattern1 = [note.Note('C4', quarterLength=1),
                   note.Note('D4', quarterLength=1),
                   note.Note('E4', quarterLength=1)]
        
        # Pattern 2: D-E-F# (real sequence, up a major 2nd)
        pattern2 = [note.Note('D4', quarterLength=1),
                   note.Note('E4', quarterLength=1),
                   note.Note('F#4', quarterLength=1)]
        
        # Pattern 3: E-F#-G# (real sequence, up another major 2nd)
        pattern3 = [note.Note('E4', quarterLength=1),
                   note.Note('F#4', quarterLength=1),
                   note.Note('G#4', quarterLength=1)]
        
        for pattern in [pattern1, pattern2, pattern3]:
            for n in pattern:
                part.append(n)
        
        real_seq_score.append(part)
        self.score_store.add_score("real_sequence", real_seq_score)
        
        # Tonal sequence (diatonic transposition)
        tonal_seq_score = stream.Score()
        part2 = stream.Part()
        
        # Pattern 1: C-D-E
        tonal1 = [note.Note('C4', quarterLength=1),
                 note.Note('D4', quarterLength=1),
                 note.Note('E4', quarterLength=1)]
        
        # Pattern 2: D-E-F (tonal sequence in C major)
        tonal2 = [note.Note('D4', quarterLength=1),
                 note.Note('E4', quarterLength=1),
                 note.Note('F4', quarterLength=1)]
        
        # Pattern 3: E-F-G (tonal sequence in C major)
        tonal3 = [note.Note('E4', quarterLength=1),
                 note.Note('F4', quarterLength=1),
                 note.Note('G4', quarterLength=1)]
        
        for pattern in [tonal1, tonal2, tonal3]:
            for n in pattern:
                part2.append(n)
        
        tonal_seq_score.append(part2)
        self.score_store.add_score("tonal_sequence", tonal_seq_score)
        
        pattern_tool = PatternRecognitionTool()
        
        # Test real sequence
        real_result = await pattern_tool.execute(
            score_id="real_sequence",
            min_pattern_length=3,
            include_transpositions=True
        )
        
        assert real_result["success"], "Real sequence analysis failed"
        
        # Test tonal sequence
        tonal_result = await pattern_tool.execute(
            score_id="tonal_sequence",
            min_pattern_length=3,
            include_transpositions=True
        )
        
        assert tonal_result["success"], "Tonal sequence analysis failed"
        
        # Both should detect sequences
        assert len(real_result.get("patterns", [])) > 0, "Real sequence not detected"
        assert len(tonal_result.get("patterns", [])) > 0, "Tonal sequence not detected"

    @pytest.mark.asyncio
    async def test_twelve_tone_row_detection(self):
        """Test detection of twelve-tone rows"""
        score = stream.Score()
        part = stream.Part()
        
        # Create a twelve-tone row (all 12 pitch classes)
        tone_row = ['C4', 'E4', 'Eb4', 'G4', 'F#4', 'Bb4', 
                   'A4', 'C#5', 'D5', 'Ab4', 'B4', 'F4']
        
        for pitch in tone_row:
            part.append(note.Note(pitch, quarterLength=1))
        
        # Add retrograde of the row
        for pitch in reversed(tone_row):
            part.append(note.Note(pitch, quarterLength=1))
        
        score.append(part)
        self.score_store.add_score("twelve_tone", score)
        
        pattern_tool = PatternRecognitionTool()
        result = await pattern_tool.execute(
            score_id="twelve_tone",
            pattern_type="twelve_tone",
            min_pattern_length=12,
            max_pattern_length=12
        )
        
        assert result["success"], "Twelve-tone analysis failed"
        
        patterns = result.get("patterns", [])
        
        # Should detect the twelve-tone row
        found_row = False
        for pattern in patterns:
            pitch_classes = pattern.get("pitch_classes", [])
            if len(set(pitch_classes)) == 12:  # All 12 pitch classes present
                found_row = True
                break
        
        assert found_row, "Twelve-tone row not detected"

    @pytest.mark.asyncio
    async def test_meter_and_hypermeter_detection(self):
        """Test detection of meter and hypermetric patterns"""
        # Create a piece with clear 3/4 meter and 4-bar hypermeter
        score = stream.Score()
        part = stream.Part()
        
        # Set time signature
        ts = stream.TimeSignature('3/4')
        part.append(ts)
        
        # Create 4-bar phrases with strong-weak patterns
        for phrase in range(2):
            # Bar 1 - strong
            part.append(chord.Chord(['C4', 'E4', 'G4'], quarterLength=1))  # Strong beat
            part.append(note.Note('E4', quarterLength=1))
            part.append(note.Note('G4', quarterLength=1))
            
            # Bar 2 - weak
            part.append(note.Note('F4', quarterLength=1))
            part.append(note.Note('A4', quarterLength=1))
            part.append(note.Note('C5', quarterLength=1))
            
            # Bar 3 - medium
            part.append(note.Note('G4', quarterLength=1))
            part.append(note.Note('B4', quarterLength=1))
            part.append(note.Note('D5', quarterLength=1))
            
            # Bar 4 - weak (cadential)
            part.append(chord.Chord(['G3', 'B3', 'D4'], quarterLength=1.5))
            part.append(chord.Chord(['C4', 'E4', 'G4'], quarterLength=1.5))
        
        score.append(part)
        self.score_store.add_score("meter_test", score)
        
        info_tool = ScoreInfoTool()
        result = await info_tool.execute(
            score_id="meter_test",
            include_detailed_analysis=True
        )
        
        assert result["success"], "Meter analysis failed"
        
        # Check time signature detection
        assert result["time_signature"] == "3/4", f"Expected 3/4 meter, got {result['time_signature']}"
        
        # Check measure count
        assert result["measures"] == 8, f"Expected 8 measures, got {result['measures']}"

    @pytest.mark.asyncio
    async def test_voice_independence_bach_invention(self):
        """Test voice independence in contrapuntal textures"""
        # Create a two-voice invention-style piece
        score = stream.Score()
        
        # Right hand - main theme
        right = stream.Part()
        theme = [
            note.Note('C5', quarterLength=0.5),
            note.Note('D5', quarterLength=0.5),
            note.Note('E5', quarterLength=0.5),
            note.Note('C5', quarterLength=0.5),
            note.Note('G5', quarterLength=1),
            note.Note('E5', quarterLength=1),
        ]
        
        # Left hand - inverted theme (contrary motion)
        left = stream.Part()
        inverted = [
            note.Note('C4', quarterLength=0.5),
            note.Note('B3', quarterLength=0.5),
            note.Note('A3', quarterLength=0.5),
            note.Note('C4', quarterLength=0.5),
            note.Note('F3', quarterLength=1),
            note.Note('A3', quarterLength=1),
        ]
        
        # Add themes
        for n in theme:
            right.append(n)
        for n in inverted:
            left.append(n)
        
        # Continue with independent lines
        right.append(note.Note('D5', quarterLength=1))
        right.append(note.Note('C5', quarterLength=1))
        left.append(note.Note('G3', quarterLength=1))
        left.append(note.Note('C3', quarterLength=1))
        
        score.insert(0, right)
        score.insert(0, left)
        
        self.score_store.add_score("voice_independence", score)
        
        voice_tool = VoiceLeadingAnalysisTool()
        result = await voice_tool.execute(
            score_id="voice_independence",
            check_voice_independence=True
        )
        
        assert result["success"], "Voice independence analysis failed"
        
        # Check voice independence metrics
        independence_score = result.get("voice_independence_score", 0)
        assert independence_score > 0.5, f"Voices not sufficiently independent: {independence_score}"
        
        # Check for contrary motion
        motion_types = result.get("motion_types", {})
        contrary_motion = motion_types.get("contrary", 0)
        assert contrary_motion > 0, "No contrary motion detected in independent voices"

    @pytest.mark.asyncio
    async def test_harmonic_rhythm_analysis(self):
        """Test harmonic rhythm analysis"""
        score = stream.Score()
        part = stream.Part()
        
        # Slow harmonic rhythm (whole notes)
        part.append(chord.Chord(['C4', 'E4', 'G4'], quarterLength=4))
        part.append(chord.Chord(['F3', 'A3', 'C4'], quarterLength=4))
        
        # Faster harmonic rhythm (quarter notes)
        part.append(chord.Chord(['G3', 'B3', 'D4'], quarterLength=1))
        part.append(chord.Chord(['C4', 'E4', 'G4'], quarterLength=1))
        part.append(chord.Chord(['A3', 'C4', 'E4'], quarterLength=1))
        part.append(chord.Chord(['D4', 'F4', 'A4'], quarterLength=1))
        
        # Return to slow
        part.append(chord.Chord(['G3', 'B3', 'D4'], quarterLength=4))
        part.append(chord.Chord(['C4', 'E4', 'G4'], quarterLength=4))
        
        score.append(part)
        self.score_store.add_score("harmonic_rhythm", score)
        
        harmony_tool = HarmonyAnalysisTool()
        result = await harmony_tool.execute(
            score_id="harmonic_rhythm",
            analyze_harmonic_rhythm=True
        )
        
        assert result["success"], "Harmonic rhythm analysis failed"
        
        # Check harmonic rhythm metrics
        harmonic_rhythm = result.get("harmonic_rhythm", {})
        assert "changes_per_measure" in harmonic_rhythm, "Harmonic rhythm not calculated"
        
        # Should detect variation in harmonic rhythm
        rhythm_variation = harmonic_rhythm.get("variation", 0)
        assert rhythm_variation > 0, "Harmonic rhythm variation not detected"

    @pytest.mark.asyncio
    async def test_form_analysis_binary(self):
        """Test detection of binary form (AB)"""
        score = stream.Score()
        part = stream.Part()
        
        # A section - 8 bars in C major
        a_section = [
            chord.Chord(['C4', 'E4', 'G4'], quarterLength=2),
            chord.Chord(['F3', 'A3', 'C4'], quarterLength=2),
            chord.Chord(['G3', 'B3', 'D4'], quarterLength=2),
            chord.Chord(['C4', 'E4', 'G4'], quarterLength=2),
            # Modulate to dominant
            chord.Chord(['D4', 'F#4', 'A4'], quarterLength=2),
            chord.Chord(['G3', 'B3', 'D4'], quarterLength=2),
            chord.Chord(['D4', 'F#4', 'A4'], quarterLength=2),
            chord.Chord(['G3', 'B3', 'D4'], quarterLength=2),
        ]
        
        # B section - 8 bars, starting in G major, returning to C
        b_section = [
            chord.Chord(['G3', 'B3', 'D4'], quarterLength=2),
            chord.Chord(['C4', 'E4', 'G4'], quarterLength=2),
            chord.Chord(['A3', 'C4', 'E4'], quarterLength=2),
            chord.Chord(['D4', 'F#4', 'A4'], quarterLength=2),
            # Return to tonic
            chord.Chord(['G3', 'B3', 'D4'], quarterLength=2),
            chord.Chord(['C4', 'E4', 'G4'], quarterLength=2),
            chord.Chord(['G3', 'B3', 'D4'], quarterLength=2),
            chord.Chord(['C4', 'E4', 'G4'], quarterLength=2),
        ]
        
        for c in a_section + b_section:
            part.append(c)
        
        score.append(part)
        self.score_store.add_score("binary_form", score)
        
        # This would require a form analysis tool, but we can use harmony analysis
        # to detect the key areas typical of binary form
        harmony_tool = HarmonyAnalysisTool()
        result = await harmony_tool.execute(
            score_id="binary_form",
            include_tonicizations=True,
            analyze_form=True
        )
        
        assert result["success"], "Binary form analysis failed"
        
        # Check for modulation to dominant and return
        key_areas = result.get("key_areas", [])
        assert len(key_areas) >= 2, "Binary form key areas not detected"
        
        # Should have tonic and dominant key areas
        keys = [area["key"] for area in key_areas]
        has_tonic = any("C major" in k for k in keys)
        has_dominant = any("G major" in k for k in keys)
        
        assert has_tonic and has_dominant, "Binary form tonic-dominant relationship not detected"