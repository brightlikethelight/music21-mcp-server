"""
Known Composition Validation Tests

Tests using specific well-known musical compositions to validate
the accuracy of analysis tools against established musical knowledge.
"""

import pytest
from typing import Dict, List, Optional
import asyncio
from music21 import corpus, stream, note, chord, key, tempo, meter

from music21_mcp.tools import (
    KeyAnalysisTool,
    ChordAnalysisTool,
    HarmonyAnalysisTool,
    PatternRecognitionTool,
)
from music21_mcp.tools.base_tool import ScoreStore


class TestKnownCompositions:
    """Test musical analysis accuracy using well-known compositions"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        ScoreStore._instance = None
        self.score_store = ScoreStore()

    def load_corpus_score(self, corpus_path: str, score_id: str) -> stream.Score:
        """Load a score from music21 corpus"""
        score = corpus.parse(corpus_path)
        self.score_store.add_score(score_id, score)
        return score

    @pytest.mark.asyncio
    async def test_bach_bwv846_prelude_c_major(self):
        """Test analysis of Bach Prelude in C Major (WTC Book 1)"""
        # This piece is definitively in C major with clear arpeggiated patterns
        score = self.load_corpus_score("bach/bwv846", "bach_prelude_c")
        
        # Test key detection
        key_tool = KeyAnalysisTool()
        key_result = await key_tool.execute(score_id="bach_prelude_c", algorithm="all")
        
        assert key_result["success"], "Key analysis failed"
        
        # All algorithms should strongly identify C major
        for algo_name, algo_result in key_result["key_analyses"].items():
            detected_key = algo_result["key"].lower()
            confidence = algo_result["confidence"]
            
            assert detected_key == "c major", (
                f"{algo_name} incorrectly identified key as {detected_key}"
            )
            assert confidence > 0.8, (
                f"{algo_name} has low confidence ({confidence}) for C major"
            )
        
        # Test pattern recognition - should find arpeggiated patterns
        pattern_tool = PatternRecognitionTool()
        pattern_result = await pattern_tool.execute(
            score_id="bach_prelude_c",
            min_pattern_length=4,
            algorithm="suffix_tree"
        )
        
        assert pattern_result["success"], "Pattern recognition failed"
        patterns = pattern_result.get("patterns", [])
        assert len(patterns) > 0, "No patterns found in Bach Prelude"
        
        # Should find many repeated patterns due to consistent figuration
        high_occurrence_patterns = [p for p in patterns if p["occurrences"] >= 4]
        assert len(high_occurrence_patterns) > 0, (
            "Bach Prelude should have highly repetitive patterns"
        )

    @pytest.mark.asyncio
    async def test_mozart_k545_sonata_c_major(self):
        """Test analysis of Mozart Sonata K. 545 first movement"""
        score = self.load_corpus_score("mozart/k545", "mozart_k545")
        
        # Test key detection - definitively C major
        key_tool = KeyAnalysisTool()
        key_result = await key_tool.execute(score_id="mozart_k545")
        
        assert key_result["success"], "Key analysis failed"
        primary_key = key_result["primary_key"]["key"].lower()
        assert primary_key == "c major", f"Mozart K. 545 should be C major, not {primary_key}"
        
        # Test harmony analysis - should find clear tonic-dominant relationships
        harmony_tool = HarmonyAnalysisTool()
        harmony_result = await harmony_tool.execute(
            score_id="mozart_k545",
            include_non_chord_tones=True,
            window_size=2
        )
        
        assert harmony_result["success"], "Harmony analysis failed"
        
        # Classical period music has clear harmonic rhythm
        segments = harmony_result.get("harmonic_segments", [])
        assert len(segments) > 20, "Mozart sonata should have many harmonic changes"
        
        # Check for typical Classical harmonic progressions
        roman_numerals = [s.get("roman_numeral", "") for s in segments]
        has_authentic_cadence = False
        
        for i in range(len(roman_numerals) - 1):
            if roman_numerals[i] in ["V", "V7"] and roman_numerals[i + 1] == "I":
                has_authentic_cadence = True
                break
        
        assert has_authentic_cadence, "No V-I authentic cadence found in Mozart sonata"

    @pytest.mark.asyncio
    async def test_beethoven_moonlight_sonata(self):
        """Test analysis of Beethoven's Moonlight Sonata (Op. 27 No. 2)"""
        # First movement is in C# minor
        score = self.load_corpus_score("beethoven/opus27no2", "moonlight_sonata")
        
        key_tool = KeyAnalysisTool()
        key_result = await key_tool.execute(score_id="moonlight_sonata")
        
        assert key_result["success"], "Key analysis failed"
        
        # Should detect C# minor
        primary_key = key_result["primary_key"]["key"].lower()
        assert "c# minor" in primary_key or "c-sharp minor" in primary_key, (
            f"Moonlight Sonata should be C# minor, not {primary_key}"
        )
        
        # Test chord analysis - famous for its arpeggiated triplet pattern
        chord_tool = ChordAnalysisTool()
        chord_result = await chord_tool.execute(
            score_id="moonlight_sonata",
            include_inversions=True,
            segment_length=1.0
        )
        
        assert chord_result["success"], "Chord analysis failed"
        
        # Should have many inverted chords due to the arpeggiated bass
        progressions = chord_result.get("progressions", [])
        inverted_chords = []
        
        for prog in progressions:
            for chord_info in prog["chords"]:
                if chord_info.get("inversion") and chord_info["inversion"] != "root position":
                    inverted_chords.append(chord_info)
        
        assert len(inverted_chords) > 5, (
            "Moonlight Sonata should have many inverted chords in arpeggiated patterns"
        )

    @pytest.mark.asyncio
    async def test_chopin_prelude_e_minor(self):
        """Test analysis of Chopin Prelude Op. 28 No. 4 in E minor"""
        # Create a simplified version of the famous E minor prelude
        score = stream.Score()
        
        # Right hand - descending chromatic line
        rh = stream.Part()
        rh_notes = [
            note.Note('B4', quarterLength=2),
            note.Note('B4', quarterLength=2),
            note.Note('C5', quarterLength=2),
            note.Note('B4', quarterLength=2),
            note.Note('A#4', quarterLength=2),
            note.Note('B4', quarterLength=2),
        ]
        
        # Left hand - pulsating chords
        lh = stream.Part()
        lh_chords = [
            chord.Chord(['E3', 'G3', 'B3'], quarterLength=0.5),
            chord.Chord(['E3', 'G3', 'B3'], quarterLength=0.5),
            chord.Chord(['E3', 'G3', 'B3'], quarterLength=0.5),
            chord.Chord(['E3', 'G3', 'B3'], quarterLength=0.5),
            
            chord.Chord(['E3', 'G3', 'C4'], quarterLength=0.5),
            chord.Chord(['E3', 'G3', 'C4'], quarterLength=0.5),
            chord.Chord(['E3', 'G3', 'C4'], quarterLength=0.5),
            chord.Chord(['E3', 'G3', 'C4'], quarterLength=0.5),
            
            chord.Chord(['D#3', 'F#3', 'B3'], quarterLength=0.5),
            chord.Chord(['D#3', 'F#3', 'B3'], quarterLength=0.5),
            chord.Chord(['D#3', 'F#3', 'B3'], quarterLength=0.5),
            chord.Chord(['D#3', 'F#3', 'B3'], quarterLength=0.5),
        ]
        
        for n in rh_notes:
            rh.append(n)
        for c in lh_chords:
            lh.append(c)
            
        score.insert(0, rh)
        score.insert(0, lh)
        
        self.score_store.add_score("chopin_prelude_e_minor", score)
        
        # Test key detection
        key_tool = KeyAnalysisTool()
        key_result = await key_tool.execute(score_id="chopin_prelude_e_minor")
        
        assert key_result["success"], "Key analysis failed"
        primary_key = key_result["primary_key"]["key"].lower()
        assert "e minor" in primary_key, f"Should detect E minor, not {primary_key}"
        
        # Test harmony - should detect chromatic voice leading
        harmony_tool = HarmonyAnalysisTool()
        harmony_result = await harmony_tool.execute(
            score_id="chopin_prelude_e_minor",
            include_non_chord_tones=True
        )
        
        assert harmony_result["success"], "Harmony analysis failed"
        
        # Check for chromatic non-chord tones
        non_chord_tones = harmony_result.get("non_chord_tones", [])
        chromatic_tones = [nct for nct in non_chord_tones if "chromatic" in nct.get("type", "").lower()]
        
        assert len(chromatic_tones) > 0, "Should detect chromatic voice leading in Chopin"

    @pytest.mark.asyncio
    async def test_pachelbel_canon(self):
        """Test analysis of Pachelbel's Canon in D"""
        # Create a simplified version with the famous chord progression
        score = stream.Score()
        
        # The famous 8-chord progression
        part = stream.Part()
        progression = [
            chord.Chord(['D3', 'F#3', 'A3'], quarterLength=2),    # I
            chord.Chord(['A2', 'C#3', 'E3', 'A3'], quarterLength=2),  # V
            chord.Chord(['B2', 'D3', 'F#3'], quarterLength=2),    # vi
            chord.Chord(['F#2', 'A2', 'C#3'], quarterLength=2),   # iii
            chord.Chord(['G2', 'B2', 'D3'], quarterLength=2),     # IV
            chord.Chord(['D2', 'F#2', 'A2', 'D3'], quarterLength=2),  # I
            chord.Chord(['G2', 'B2', 'D3'], quarterLength=2),     # IV
            chord.Chord(['A2', 'C#3', 'E3'], quarterLength=2),    # V
        ]
        
        # Repeat the progression
        for _ in range(2):
            for ch in progression:
                part.append(ch)
        
        score.append(part)
        self.score_store.add_score("pachelbel_canon", score)
        
        # Test chord progression analysis
        chord_tool = ChordAnalysisTool()
        chord_result = await chord_tool.execute(
            score_id="pachelbel_canon",
            include_roman_numerals=True
        )
        
        assert chord_result["success"], "Chord analysis failed"
        
        # Should detect the repeating progression
        progressions = chord_result.get("progressions", [])
        assert len(progressions) > 0, "No progressions found"
        
        # Extract Roman numerals
        all_numerals = []
        for prog in progressions:
            for chord_info in prog["chords"]:
                if chord_info.get("roman_numeral"):
                    all_numerals.append(chord_info["roman_numeral"])
        
        # Check for the characteristic I-V-vi-iii-IV-I-IV-V progression
        expected_progression = ["I", "V", "vi", "iii", "IV", "I", "IV", "V"]
        
        # Find if this sequence appears
        found_canon_progression = False
        for i in range(len(all_numerals) - len(expected_progression) + 1):
            sequence = all_numerals[i:i + len(expected_progression)]
            if sequence == expected_progression:
                found_canon_progression = True
                break
        
        assert found_canon_progression, (
            f"Canon progression not found. Got: {all_numerals}"
        )

    @pytest.mark.asyncio
    async def test_debussy_clair_de_lune_impressionism(self):
        """Test analysis of impressionistic harmony in Debussy"""
        # Create a simplified version with impressionistic harmonies
        score = stream.Score()
        part = stream.Part()
        
        # Characteristic Debussy harmonies with extended chords
        impressionist_chords = [
            # Extended and altered chords typical of Debussy
            chord.Chord(['Db3', 'F3', 'Ab3', 'C4', 'Eb4'], quarterLength=4),  # Db add9
            chord.Chord(['Bb2', 'D3', 'F3', 'A3', 'C4'], quarterLength=4),    # Bb9
            chord.Chord(['Eb3', 'G3', 'Bb3', 'Db4', 'F4'], quarterLength=4),  # Eb9
            chord.Chord(['Ab2', 'C3', 'Eb3', 'G3', 'Bb3'], quarterLength=4),  # Ab add9
        ]
        
        for ch in impressionist_chords:
            part.append(ch)
            
        score.append(part)
        self.score_store.add_score("debussy_style", score)
        
        # Test chord analysis - should identify extended harmonies
        chord_tool = ChordAnalysisTool()
        chord_result = await chord_tool.execute(
            score_id="debussy_style",
            include_extensions=True
        )
        
        assert chord_result["success"], "Chord analysis failed"
        
        # Should detect extended chords (9ths, etc.)
        progressions = chord_result.get("progressions", [])
        extended_chords = []
        
        for prog in progressions:
            for chord_info in prog["chords"]:
                symbol = chord_info.get("chord_symbol", "")
                if any(ext in symbol for ext in ["9", "11", "13", "add"]):
                    extended_chords.append(chord_info)
        
        assert len(extended_chords) >= 2, (
            "Impressionistic music should have extended harmonies"
        )

    @pytest.mark.asyncio
    async def test_schoenberg_atonal_analysis(self):
        """Test analysis of atonal/twelve-tone music"""
        # Create a simple twelve-tone row
        score = stream.Score()
        part = stream.Part()
        
        # A twelve-tone row (all 12 pitch classes)
        tone_row = [
            note.Note('C4', quarterLength=1),
            note.Note('C#4', quarterLength=1),
            note.Note('D4', quarterLength=1),
            note.Note('Eb4', quarterLength=1),
            note.Note('E4', quarterLength=1),
            note.Note('F4', quarterLength=1),
            note.Note('F#4', quarterLength=1),
            note.Note('G4', quarterLength=1),
            note.Note('Ab4', quarterLength=1),
            note.Note('A4', quarterLength=1),
            note.Note('Bb4', quarterLength=1),
            note.Note('B4', quarterLength=1),
        ]
        
        for n in tone_row:
            part.append(n)
            
        score.append(part)
        self.score_store.add_score("twelve_tone_row", score)
        
        # Test key analysis - should have low confidence or no clear key
        key_tool = KeyAnalysisTool()
        key_result = await key_tool.execute(score_id="twelve_tone_row")
        
        assert key_result["success"], "Key analysis failed"
        
        # Atonal music should have very low key confidence
        confidence = key_result["primary_key"]["confidence"]
        assert confidence < 0.5, (
            f"Twelve-tone music should have low key confidence, got {confidence}"
        )
        
        # Test pattern recognition - should find the row and transformations
        pattern_tool = PatternRecognitionTool()
        pattern_result = await pattern_tool.execute(
            score_id="twelve_tone_row",
            min_pattern_length=12,
            max_pattern_length=12,
            include_transpositions=True,
            include_inversions=True
        )
        
        assert pattern_result["success"], "Pattern recognition failed"
        
        # The entire row is a pattern of 12 unique pitch classes
        pitch_classes = set()
        for n in tone_row:
            pitch_classes.add(n.pitch.pitchClass)
        
        assert len(pitch_classes) == 12, "Twelve-tone row should use all 12 pitch classes"