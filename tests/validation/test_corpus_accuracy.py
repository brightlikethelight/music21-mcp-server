"""
Corpus-based Musical Accuracy Tests for music21-mcp-server

This module tests the accuracy of analysis tools against well-known
musical works from the music21 corpus with established analytical interpretations.
"""

import pytest
from typing import Dict, List, Optional, Tuple, Any
import asyncio
from music21 import corpus, stream, note, chord, key, roman, interval

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


class TestCorpusAccuracy:
    """Test suite for validating accuracy against known corpus examples"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        # Clear score store before each test
        ScoreStore._instance = None
        self.score_store = ScoreStore()
        
    def load_corpus_score(self, corpus_path: str, score_id: str) -> stream.Score:
        """Load a score from music21 corpus"""
        try:
            score = corpus.parse(corpus_path)
            self.score_store.add_score(score_id, score)
            return score
        except Exception as e:
            pytest.skip(f"Corpus file {corpus_path} not available: {e}")

    @pytest.mark.asyncio
    async def test_bach_chorale_analysis_comprehensive(self):
        """Comprehensive analysis of Bach chorales with known properties"""
        # Test multiple Bach chorales with well-documented analyses
        chorale_analyses = [
            {
                "path": "bach/bwv66.6",
                "key": "f# minor",
                "final_cadence": "authentic",
                "contains_modulations": True,
                "common_progressions": ["ii-V-i", "iv-V-i"],
            },
            {
                "path": "bach/bwv269",
                "key": "C major",
                "final_cadence": "authentic",
                "contains_modulations": True,
                "common_progressions": ["IV-V-I", "ii-V-I"],
            },
            {
                "path": "bach/bwv277",
                "key": "Eb major",
                "final_cadence": "authentic",
                "contains_modulations": True,
                "common_progressions": ["IV-V-I", "vi-ii-V-I"],
            },
        ]
        
        for chorale_data in chorale_analyses:
            score_id = f"chorale_{chorale_data['path'].replace('/', '_')}"
            score = self.load_corpus_score(chorale_data["path"], score_id)
            
            # Test key detection
            key_tool = KeyAnalysisTool()
            key_result = await key_tool.execute(
                score_id=score_id,
                algorithm="krumhansl_schmuckler"
            )
            
            assert key_result["success"], f"Key analysis failed for {chorale_data['path']}"
            detected_key = key_result["key_analyses"]["krumhansl_schmuckler"]["key"].lower()
            assert detected_key == chorale_data["key"], (
                f"Expected {chorale_data['key']} but got {detected_key}"
            )
            
            # Test harmony analysis
            harmony_tool = HarmonyAnalysisTool()
            harmony_result = await harmony_tool.execute(
                score_id=score_id,
                include_cadences=True,
                include_tonicizations=True
            )
            
            assert harmony_result["success"], f"Harmony analysis failed for {chorale_data['path']}"
            
            # Check for modulations
            if chorale_data["contains_modulations"]:
                key_areas = harmony_result.get("key_areas", [])
                assert len(key_areas) > 1, f"No modulations detected in {chorale_data['path']}"
            
            # Check final cadence
            cadences = harmony_result.get("cadences", [])
            if cadences and chorale_data["final_cadence"]:
                final_cadence = cadences[-1]
                assert chorale_data["final_cadence"] in final_cadence.get("type", "").lower(), (
                    f"Expected {chorale_data['final_cadence']} cadence at end"
                )

    @pytest.mark.asyncio
    async def test_mozart_k545_analysis(self):
        """Test analysis of Mozart K. 545 (Sonata Facile) first movement"""
        score_id = "mozart_k545_1"
        score = self.load_corpus_score("mozart/k545/movement1", score_id)
        
        # Known properties of K. 545, movement 1
        expected_key = "C major"
        expected_form = "sonata"
        expected_sections = ["exposition", "development", "recapitulation"]
        
        # Test key analysis
        key_tool = KeyAnalysisTool()
        key_result = await key_tool.execute(score_id=score_id, algorithm="all")
        
        assert key_result["success"], "Key analysis failed for Mozart K. 545"
        
        # At least one algorithm should correctly identify C major
        detected_c_major = False
        for algo_name, algo_result in key_result["key_analyses"].items():
            if algo_result["key"].lower() == expected_key.lower():
                detected_c_major = True
                break
        
        assert detected_c_major, f"C major not detected in Mozart K. 545"
        
        # Test pattern recognition for Alberti bass
        pattern_tool = PatternRecognitionTool()
        pattern_result = await pattern_tool.execute(
            score_id=score_id,
            pattern_type="melodic",
            min_pattern_length=4,
            include_rhythm=True
        )
        
        assert pattern_result["success"], "Pattern recognition failed"
        
        # Mozart K. 545 is famous for its Alberti bass patterns
        patterns = pattern_result.get("patterns", [])
        assert len(patterns) > 0, "No patterns detected in Mozart K. 545"
        
        # Check for repeated patterns (characteristic of Classical style)
        high_occurrence_patterns = [p for p in patterns if p.get("occurrences", 0) > 5]
        assert len(high_occurrence_patterns) > 0, "No frequently repeated patterns found"

    @pytest.mark.asyncio
    async def test_beethoven_opus27_no2_moonlight(self):
        """Test analysis of Beethoven's Moonlight Sonata first movement"""
        score_id = "beethoven_moonlight_1"
        
        try:
            score = self.load_corpus_score("beethoven/opus27no2/movement1", score_id)
        except:
            # Try alternative path
            score = self.load_corpus_score("beethoven/opus27no2-1", score_id)
        
        # Known properties of Moonlight Sonata, movement 1
        expected_key = "c# minor"
        expected_texture = "homophonic"
        expected_character = "sustained triplets"
        
        # Test key analysis
        key_tool = KeyAnalysisTool()
        key_result = await key_tool.execute(
            score_id=score_id,
            algorithm="krumhansl_schmuckler"
        )
        
        assert key_result["success"], "Key analysis failed for Moonlight Sonata"
        
        detected_key = key_result["key_analyses"]["krumhansl_schmuckler"]["key"].lower()
        assert detected_key == expected_key, (
            f"Expected {expected_key} but got {detected_key} for Moonlight Sonata"
        )
        
        # Test pattern recognition for the famous triplet pattern
        pattern_tool = PatternRecognitionTool()
        pattern_result = await pattern_tool.execute(
            score_id=score_id,
            pattern_type="rhythmic",
            min_pattern_length=3
        )
        
        assert pattern_result["success"], "Pattern recognition failed"
        
        # Should detect the persistent triplet pattern
        patterns = pattern_result.get("patterns", [])
        triplet_patterns = []
        for pattern in patterns:
            durations = pattern.get("durations", [])
            # Check for triplet durations (approximately 0.333)
            if any(0.3 < d < 0.4 for d in durations):
                triplet_patterns.append(pattern)
        
        assert len(triplet_patterns) > 0, "Triplet patterns not detected in Moonlight Sonata"

    @pytest.mark.asyncio
    async def test_schubert_lieder_analysis(self):
        """Test analysis of Schubert Lieder (art songs)"""
        # Test "Gretchen am Spinnrade" if available
        score_id = "schubert_gretchen"
        
        try:
            score = self.load_corpus_score("schubert/d118", score_id)
        except:
            pytest.skip("Schubert D. 118 not available in corpus")
        
        # Known properties of Gretchen am Spinnrade
        expected_key = "d minor"
        expected_features = ["spinning wheel figuration", "text painting"]
        
        # Test key analysis
        key_tool = KeyAnalysisTool()
        key_result = await key_tool.execute(score_id=score_id)
        
        assert key_result["success"], "Key analysis failed for Schubert"
        
        # Test pattern recognition for spinning wheel figuration
        pattern_tool = PatternRecognitionTool()
        pattern_result = await pattern_tool.execute(
            score_id=score_id,
            pattern_type="melodic",
            min_pattern_length=2,
            max_pattern_length=4,
            include_rhythm=True
        )
        
        assert pattern_result["success"], "Pattern recognition failed"
        
        # Should detect highly repetitive patterns (spinning wheel)
        patterns = pattern_result.get("patterns", [])
        repetitive_patterns = [p for p in patterns if p.get("occurrences", 0) > 10]
        assert len(repetitive_patterns) > 0, "Spinning wheel figuration not detected"

    @pytest.mark.asyncio
    async def test_palestrina_counterpoint(self):
        """Test analysis of Palestrina's Renaissance counterpoint"""
        score_id = "palestrina_kyrie"
        
        try:
            # Try to load a Palestrina mass movement
            score = self.load_corpus_score("palestrina/kyrie", score_id)
        except:
            pytest.skip("Palestrina not available in corpus")
        
        # Test voice leading in Renaissance style
        voice_tool = VoiceLeadingAnalysisTool()
        result = await voice_tool.execute(
            score_id=score_id,
            check_parallels=True,
            check_voice_crossings=True,
            check_large_leaps=True,
            style_period="renaissance"
        )
        
        assert result["success"], "Voice leading analysis failed"
        
        # Palestrina's style should have minimal voice leading issues
        issues = result.get("voice_leading_issues", [])
        parallel_issues = [i for i in issues if "parallel" in i.get("type", "").lower()]
        
        # Renaissance style allows some parallel thirds and sixths
        severe_parallels = [i for i in parallel_issues 
                           if "fifth" in str(i).lower() or "octave" in str(i).lower()]
        
        assert len(severe_parallels) == 0, "Parallel fifths/octaves found in Palestrina"
        
        # Check smoothness
        smoothness = result.get("smoothness_score", 0)
        assert smoothness > 0.8, f"Palestrina voice leading not smooth enough: {smoothness}"

    @pytest.mark.asyncio
    async def test_debussy_impressionism(self):
        """Test analysis of Debussy's impressionistic harmony"""
        score_id = "debussy_clair"
        
        try:
            score = self.load_corpus_score("debussy/claire_de_lune", score_id)
        except:
            pytest.skip("Debussy Clair de Lune not available in corpus")
        
        # Test harmony analysis for impressionistic features
        harmony_tool = HarmonyAnalysisTool()
        result = await harmony_tool.execute(
            score_id=score_id,
            include_extended_harmony=True
        )
        
        assert result["success"], "Harmony analysis failed for Debussy"
        
        segments = result.get("harmonic_segments", [])
        
        # Check for extended harmonies (7ths, 9ths, etc.)
        extended_chords = 0
        for segment in segments:
            chord_type = segment.get("chord_type", "")
            if any(ext in chord_type for ext in ["7", "9", "11", "13"]):
                extended_chords += 1
        
        # Debussy uses many extended harmonies
        assert extended_chords > len(segments) * 0.3, (
            "Not enough extended harmonies detected for impressionistic style"
        )

    @pytest.mark.asyncio
    async def test_bach_invention_analysis(self):
        """Test analysis of Bach Two-Part Inventions"""
        # Test Invention No. 1 in C major
        score_id = "bach_invention_1"
        score = self.load_corpus_score("bach/bwv772", score_id)
        
        # Known properties of Invention No. 1
        expected_key = "C major"
        expected_texture = "two-voice counterpoint"
        expected_form = "binary"
        
        # Test key analysis
        key_tool = KeyAnalysisTool()
        key_result = await key_tool.execute(score_id=score_id)
        
        assert key_result["success"], "Key analysis failed for Bach Invention"
        
        # Test pattern recognition for the main motif
        pattern_tool = PatternRecognitionTool()
        pattern_result = await pattern_tool.execute(
            score_id=score_id,
            min_pattern_length=4,
            include_transpositions=True,
            algorithm="suffix_tree"
        )
        
        assert pattern_result["success"], "Pattern recognition failed"
        
        patterns = pattern_result.get("patterns", [])
        
        # Bach inventions feature extensive motivic development
        transposed_patterns = [p for p in patterns if p.get("transposition_interval")]
        assert len(transposed_patterns) > 5, (
            "Not enough transposed patterns found in Bach Invention"
        )
        
        # Test voice independence
        voice_tool = VoiceLeadingAnalysisTool()
        voice_result = await voice_tool.execute(
            score_id=score_id,
            check_voice_independence=True
        )
        
        assert voice_result["success"], "Voice independence analysis failed"
        
        independence_score = voice_result.get("voice_independence_score", 0)
        assert independence_score > 0.7, (
            f"Bach Invention voices not independent enough: {independence_score}"
        )

    @pytest.mark.asyncio
    async def test_chopin_mazurka_analysis(self):
        """Test analysis of Chopin Mazurkas for dance characteristics"""
        score_id = "chopin_mazurka"
        
        try:
            # Try to load a Chopin Mazurka
            score = self.load_corpus_score("chopin/mazurka06-1", score_id)
        except:
            pytest.skip("Chopin Mazurka not available in corpus")
        
        # Test for mazurka characteristics
        # Mazurkas are in 3/4 with accent on beat 2 or 3
        info_tool = ScoreInfoTool()
        info_result = await info_tool.execute(
            score_id=score_id,
            include_detailed_analysis=True
        )
        
        assert info_result["success"], "Score info failed for Chopin Mazurka"
        
        # Check time signature
        time_sig = info_result.get("time_signature", "")
        assert "3/4" in time_sig or "3" in time_sig, (
            f"Expected 3/4 time for Mazurka, got {time_sig}"
        )
        
        # Test harmony for Romantic characteristics
        harmony_tool = HarmonyAnalysisTool()
        harmony_result = await harmony_tool.execute(
            score_id=score_id,
            include_tonicizations=True
        )
        
        assert harmony_result["success"], "Harmony analysis failed"
        
        # Chopin often uses chromatic harmony
        segments = harmony_result.get("harmonic_segments", [])
        chromatic_segments = [s for s in segments 
                             if s.get("is_chromatic") or "#" in s.get("root", "") or "b" in s.get("root", "")]
        
        assert len(chromatic_segments) > 0, "No chromatic harmony detected in Chopin"

    @pytest.mark.asyncio
    async def test_monteverdi_madrigal_analysis(self):
        """Test analysis of Monteverdi madrigals for text-music relationships"""
        score_id = "monteverdi_madrigal"
        
        try:
            score = self.load_corpus_score("monteverdi/madrigal.3.1", score_id)
        except:
            pytest.skip("Monteverdi madrigal not available in corpus")
        
        # Test voice leading for Renaissance/early Baroque style
        voice_tool = VoiceLeadingAnalysisTool()
        result = await voice_tool.execute(
            score_id=score_id,
            check_parallels=True,
            check_voice_crossings=True,
            style_period="renaissance"
        )
        
        assert result["success"], "Voice leading analysis failed"
        
        # Madrigals have more chromatic voice leading than sacred music
        issues = result.get("voice_leading_issues", [])
        
        # Should still follow basic voice leading rules
        parallel_fifths = [i for i in issues if "parallel" in i.get("type", "").lower() and "fifth" in str(i).lower()]
        assert len(parallel_fifths) < 3, "Too many parallel fifths in madrigal"

    @pytest.mark.asyncio
    async def test_messiaen_modes_analysis(self):
        """Test analysis of Messiaen's modes of limited transposition"""
        # Create a piece using Messiaen's Mode 1 (whole tone scale)
        score = stream.Score()
        part = stream.Part()
        
        # Whole tone scale: C-D-E-F#-G#-A#-C
        whole_tone_notes = ['C4', 'D4', 'E4', 'F#4', 'G#4', 'A#4', 'C5']
        
        for note_name in whole_tone_notes * 2:  # Repeat pattern
            part.append(note.Note(note_name, quarterLength=1))
        
        score.append(part)
        self.score_store.add_score("messiaen_mode1", score)
        
        # Test pattern recognition
        pattern_tool = PatternRecognitionTool()
        result = await pattern_tool.execute(
            score_id="messiaen_mode1",
            pattern_type="interval",
            min_pattern_length=2
        )
        
        assert result["success"], "Pattern recognition failed for Messiaen mode"
        
        patterns = result.get("patterns", [])
        
        # Should detect consistent whole tone intervals
        whole_tone_patterns = []
        for pattern in patterns:
            intervals = pattern.get("intervals", [])
            # Whole tone = major 2nd = 2 semitones
            if all(abs(int(i)) == 2 for i in intervals if i.isdigit() or (i[0] == '-' and i[1:].isdigit())):
                whole_tone_patterns.append(pattern)
        
        assert len(whole_tone_patterns) > 0, "Whole tone patterns not detected"

    @pytest.mark.asyncio
    async def test_reich_minimalism_analysis(self):
        """Test analysis of minimalist techniques (phase shifting, etc.)"""
        # Create a simple minimalist pattern with gradual changes
        score = stream.Score()
        
        # Pattern 1 - basic pattern
        part1 = stream.Part()
        pattern = ['C4', 'E4', 'B3', 'E4']
        
        # Repeat pattern many times
        for _ in range(20):
            for note_name in pattern:
                part1.append(note.Note(note_name, quarterLength=0.5))
        
        # Pattern 2 - slightly phase-shifted
        part2 = stream.Part()
        part2.append(note.Rest(quarterLength=0.25))  # Phase shift
        
        for _ in range(20):
            for note_name in pattern:
                part2.append(note.Note(note_name, quarterLength=0.5))
        
        score.insert(0, part1)
        score.insert(0, part2)
        
        self.score_store.add_score("reich_minimalism", score)
        
        # Test pattern recognition
        pattern_tool = PatternRecognitionTool()
        result = await pattern_tool.execute(
            score_id="reich_minimalism",
            min_pattern_length=4,
            max_pattern_length=4
        )
        
        assert result["success"], "Pattern recognition failed for minimalism"
        
        patterns = result.get("patterns", [])
        
        # Should detect highly repetitive patterns
        repetitive_patterns = [p for p in patterns if p.get("occurrences", 0) > 15]
        assert len(repetitive_patterns) > 0, "Repetitive minimalist patterns not detected"
        
        # Test for phase relationships between parts
        # This would require specialized analysis for minimalist techniques