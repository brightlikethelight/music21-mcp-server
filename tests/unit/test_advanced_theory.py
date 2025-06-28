"""
Unit tests for advanced theory analysis functionality
"""
import pytest
from music21 import stream, note, chord, key, interval, roman

from src.music21_mcp.core.advanced_theory import (
    AdvancedTheoryAnalyzer, ChromaticFunction, PhraseType,
    ScaleDegreeAnalysis, IntervalVectorAnalysis, ChromaticAnalysis,
    AdvancedHarmonicAnalysis, PhraseStructure
)


class TestAdvancedTheoryAnalyzer:
    """Test advanced theory analysis"""
    
    @pytest.fixture
    def analyzer(self):
        return AdvancedTheoryAnalyzer()
    
    @pytest.mark.asyncio
    async def test_scale_degree_analysis(self, analyzer):
        """Test scale degree analysis"""
        # Create melody with clear scale degrees
        s = stream.Score()
        p = stream.Part()
        p.append(key.KeySignature(0))  # C major
        
        # Scale degrees: 1-2-3-4-5-6-7-1
        notes = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5']
        for pitch in notes:
            p.append(note.Note(pitch, quarterLength=1))
        
        # Add leading tone resolution (7->1)
        p.append(note.Note('B4', quarterLength=1))
        p.append(note.Note('C5', quarterLength=1))
        
        s.append(p)
        
        result = await analyzer.analyze_scale_degrees(s)
        
        assert isinstance(result, ScaleDegreeAnalysis)
        assert 1 in result.scale_degree_histogram  # Tonic
        assert result.scale_degree_histogram[1] >= 2  # C appears at least twice
        assert '7->1' in result.tendency_tone_resolutions  # Leading tone resolution
        assert result.step_leap_ratio > 0.8  # Mostly stepwise motion
        assert result.chromatic_percentage == 0  # No chromatic notes
    
    @pytest.mark.asyncio
    async def test_interval_vector_calculation(self, analyzer):
        """Test interval vector calculation"""
        # Create C major triad
        pitches = [note.Note('C4').pitch, note.Note('E4').pitch, note.Note('G4').pitch]
        
        result = await analyzer.calculate_interval_vector(pitches)
        
        assert isinstance(result, IntervalVectorAnalysis)
        assert len(result.interval_vector) == 6
        # C major triad has: M3 (IC4), m3 (IC3), P5 (IC5)
        assert result.interval_vector[2] == 1  # One IC3 (E-G)
        assert result.interval_vector[3] == 1  # One IC4 (C-E)
        assert result.consonance_ratio > 0.5  # Mostly consonant
    
    @pytest.mark.asyncio
    async def test_chromatic_analysis(self, analyzer):
        """Test chromatic element detection"""
        s = stream.Score()
        p = stream.Part()
        p.append(key.KeySignature(0))  # C major
        
        # Add chromatic passing tone
        p.append(note.Note('C4', quarterLength=1))
        p.append(note.Note('C#4', quarterLength=0.5))  # Chromatic
        p.append(note.Note('D4', quarterLength=1))
        
        # Add chromatic neighbor
        p.append(note.Note('E4', quarterLength=1))
        p.append(note.Note('F#4', quarterLength=0.5))  # Chromatic
        p.append(note.Note('E4', quarterLength=1))
        
        s.append(p)
        
        result = await analyzer.analyze_chromatic_elements(s)
        
        assert isinstance(result, ChromaticAnalysis)
        assert len(result.chromatic_notes) == 2
        assert result.chromatic_density > 0
        assert ChromaticFunction.PASSING in result.chromatic_functions
        assert ChromaticFunction.NEIGHBOR in result.chromatic_functions
    
    @pytest.mark.asyncio
    async def test_secondary_dominant_detection(self, analyzer):
        """Test secondary dominant detection"""
        s = stream.Score()
        p = stream.Part()
        p.append(key.KeySignature(0))  # C major
        
        # C - D7 - G - C (I - V7/V - V - I)
        p.append(chord.Chord(['C4', 'E4', 'G4']))  # I
        p.append(chord.Chord(['D4', 'F#4', 'A4', 'C5']))  # V7/V (D7)
        p.append(chord.Chord(['G4', 'B4', 'D5']))  # V
        p.append(chord.Chord(['C4', 'E4', 'G4']))  # I
        
        s.append(p)
        
        harmony_result = await analyzer.detect_advanced_harmony(s)
        
        assert isinstance(harmony_result, AdvancedHarmonicAnalysis)
        assert len(harmony_result.secondary_dominants) > 0
        
        # Should find V/V
        v_of_v = harmony_result.secondary_dominants[0]
        assert v_of_v['target_degree'] == 5  # Targets the dominant
        assert 'V' in v_of_v['symbol']
    
    @pytest.mark.asyncio
    async def test_augmented_sixth_detection(self, analyzer):
        """Test augmented sixth chord detection"""
        s = stream.Score()
        p = stream.Part()
        p.append(key.KeySignature(0))  # C major
        
        # Italian augmented sixth: Ab-C-F# resolving to G
        p.append(chord.Chord(['A-4', 'C5', 'F#5']))  # It+6
        p.append(chord.Chord(['G4', 'B4', 'D5']))    # V
        
        s.append(p)
        
        harmony_result = await analyzer.detect_advanced_harmony(s)
        
        assert len(harmony_result.augmented_sixths) > 0
        aug6 = harmony_result.augmented_sixths[0]
        assert aug6['type'] == 'Italian'
    
    @pytest.mark.asyncio
    async def test_neapolitan_detection(self, analyzer):
        """Test Neapolitan chord detection"""
        s = stream.Score()
        p = stream.Part()
        p.append(key.KeySignature(0))  # C major
        
        # Neapolitan: Db major in first inversion (F-Ab-Db)
        p.append(chord.Chord(['F4', 'A-4', 'D-5']))  # N6
        p.append(chord.Chord(['G4', 'B4', 'D5']))    # V
        p.append(chord.Chord(['C4', 'E4', 'G4']))    # I
        
        s.append(p)
        
        harmony_result = await analyzer.detect_advanced_harmony(s)
        
        assert len(harmony_result.neapolitan_chords) > 0
        neap = harmony_result.neapolitan_chords[0]
        assert neap['function'] == 'bII6'  # Flat II in first inversion
    
    @pytest.mark.asyncio
    async def test_phrase_structure_period(self, analyzer):
        """Test period phrase structure detection"""
        s = stream.Score()
        p = stream.Part()
        p.append(key.KeySignature(0))
        
        # Antecedent phrase (4 bars)
        for pitch in ['C4', 'D4', 'E4', 'F4']:
            p.append(note.Note(pitch, quarterLength=1))
        
        # Half cadence
        p.append(chord.Chord(['G4', 'B4', 'D5']))
        
        # Consequent phrase (4 bars)
        for pitch in ['C4', 'D4', 'E4', 'F4']:
            p.append(note.Note(pitch, quarterLength=1))
        
        # Authentic cadence
        p.append(chord.Chord(['G4', 'B4', 'D5']))
        p.append(chord.Chord(['C4', 'E4', 'G4']))
        
        s.append(p)
        
        result = await analyzer.analyze_phrase_structure(s)
        
        assert isinstance(result, PhraseStructure)
        assert result.phrase_type == PhraseType.PERIOD
        assert len(result.cadences) >= 2
    
    @pytest.mark.asyncio
    async def test_phrase_structure_sentence(self, analyzer):
        """Test sentence phrase structure detection"""
        s = stream.Score()
        p = stream.Part()
        
        # Basic idea (2 bars)
        p.append(note.Note('C4', quarterLength=2))
        p.append(note.Note('D4', quarterLength=2))
        
        # Repetition (2 bars)
        p.append(note.Note('C4', quarterLength=2))
        p.append(note.Note('D4', quarterLength=2))
        
        # Continuation (4 bars)
        for pitch in ['E4', 'F4', 'G4', 'C5']:
            p.append(note.Note(pitch, quarterLength=2))
        
        s.append(p)
        
        result = await analyzer.analyze_phrase_structure(s)
        
        # Should detect 2+2+4 structure
        assert result.phrase_lengths == [2, 2, 4] or result.phrase_type == PhraseType.SENTENCE
    
    @pytest.mark.asyncio
    async def test_dissonance_treatment(self, analyzer):
        """Test dissonance treatment analysis"""
        s = stream.Score()
        p1 = stream.Part()
        p2 = stream.Part()
        
        # Create suspension: C-B-C against E
        p1.append(note.Note('C4', quarterLength=1))  # Consonant
        p1.append(note.Note('B3', quarterLength=1))  # Dissonant
        p1.append(note.Note('C4', quarterLength=1))  # Resolution
        
        p2.append(note.Note('E4', quarterLength=3))  # Sustained
        
        s.append(p1)
        s.append(p2)
        
        result = await analyzer.analyze_dissonance_treatment(s)
        
        assert result['total_dissonances'] > 0
        assert 'suspension' in result['dissonance_types']


class TestChromaticFunctions:
    """Test chromatic function classification"""
    
    @pytest.fixture
    def analyzer(self):
        return AdvancedTheoryAnalyzer()
    
    @pytest.mark.asyncio
    async def test_passing_tone_classification(self, analyzer):
        """Test passing tone identification"""
        s = stream.Score()
        p = stream.Part()
        p.append(key.KeySignature(0))  # C major
        
        # C - C# - D (chromatic passing tone)
        notes = [
            note.Note('C4', quarterLength=1),
            note.Note('C#4', quarterLength=0.5),
            note.Note('D4', quarterLength=1)
        ]
        
        for n in notes:
            p.append(n)
        
        s.append(p)
        
        key_context = key.Key('C')
        function = await analyzer._classify_chromatic_function(
            notes[1], 1, notes, key_context
        )
        
        assert function == ChromaticFunction.PASSING
    
    @pytest.mark.asyncio
    async def test_neighbor_tone_classification(self, analyzer):
        """Test chromatic neighbor tone identification"""
        s = stream.Score()
        p = stream.Part()
        p.append(key.KeySignature(0))  # C major
        
        # E - F# - E (chromatic upper neighbor)
        notes = [
            note.Note('E4', quarterLength=1),
            note.Note('F#4', quarterLength=0.5),
            note.Note('E4', quarterLength=1)
        ]
        
        key_context = key.Key('C')
        function = await analyzer._classify_chromatic_function(
            notes[1], 1, notes, key_context
        )
        
        assert function == ChromaticFunction.NEIGHBOR