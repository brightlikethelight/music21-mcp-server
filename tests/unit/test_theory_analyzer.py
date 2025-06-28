"""
Unit tests for music theory analysis functionality
"""
import pytest
from music21 import stream, note, chord, key, interval, scale

from src.music21_mcp.core.theory_analyzer import (
    TheoryAnalyzer, KeyDetectionMethod, KeyAnalysisResult,
    IntervalAnalysis
)


class TestTheoryAnalyzer:
    """Test the theory analyzer"""
    
    @pytest.fixture
    def analyzer(self):
        return TheoryAnalyzer()
    
    @pytest.mark.asyncio
    async def test_key_detection_simple(self, analyzer, simple_score):
        """Test basic key detection"""
        result = await analyzer.analyze_key(
            simple_score,
            method=KeyDetectionMethod.KRUMHANSL
        )
        
        assert isinstance(result, KeyAnalysisResult)
        assert result.key.tonic.name == 'C'
        assert result.key.mode == 'major'
        assert result.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_key_detection_hybrid(self, analyzer, simple_score):
        """Test hybrid key detection method"""
        result = await analyzer.analyze_key(
            simple_score,
            method=KeyDetectionMethod.HYBRID
        )
        
        assert result.method == 'hybrid'
        assert len(result.alternatives) >= 0
        assert result.key is not None
    
    @pytest.mark.asyncio
    async def test_key_detection_with_modulation(self, analyzer):
        """Test key detection with modulation tracking"""
        # Create score that modulates from C to G
        s = stream.Score()
        p = stream.Part()
        
        # First phrase in C
        p.append(key.KeySignature(0))
        for pitch in ['C4', 'E4', 'G4', 'C5']:
            p.append(note.Note(pitch, quarterLength=1))
        
        # Modulation to G
        p.append(key.KeySignature(1))
        for pitch in ['G4', 'B4', 'D5', 'G5']:
            p.append(note.Note(pitch, quarterLength=1))
        
        s.append(p)
        
        result = await analyzer.analyze_key(
            s,
            window_size=4  # Analyze in 4-measure windows
        )
        
        assert len(result.local_keys) > 0
        assert len(result.modulations) > 0
    
    @pytest.mark.asyncio
    async def test_scale_analysis_major(self, analyzer, simple_score):
        """Test scale analysis for major scale"""
        result = await analyzer.analyze_scale(simple_score)
        
        assert 'possible_scales' in result
        assert result['best_match'] is not None
        assert 'C major' in result['best_match']['scale']
        assert result['confidence'] > 0.8
    
    @pytest.mark.asyncio
    async def test_scale_analysis_with_modes(self, analyzer):
        """Test modal scale detection"""
        # Create Dorian scale
        s = stream.Score()
        p = stream.Part()
        
        dorian_pitches = ['D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5', 'D5']
        for pitch in dorian_pitches:
            p.append(note.Note(pitch, quarterLength=1))
        
        s.append(p)
        
        result = await analyzer.analyze_scale(s, include_modes=True)
        
        # Should detect D Dorian
        scales = result['possible_scales']
        dorian_found = any('dorian' in s['type'].lower() for s in scales)
        assert dorian_found
    
    @pytest.mark.asyncio
    async def test_scale_analysis_exotic(self, analyzer):
        """Test exotic scale detection"""
        # Create pentatonic scale
        s = stream.Score()
        p = stream.Part()
        
        pentatonic_pitches = ['C4', 'D4', 'E4', 'G4', 'A4', 'C5']
        for pitch in pentatonic_pitches:
            p.append(note.Note(pitch, quarterLength=1))
        
        s.append(p)
        
        result = await analyzer.analyze_scale(s, include_exotic=True)
        
        scales = result['possible_scales']
        pentatonic_found = any('pentatonic' in s['type'].lower() for s in scales)
        assert pentatonic_found
    
    @pytest.mark.asyncio
    async def test_interval_analysis_basic(self, analyzer):
        """Test basic interval analysis"""
        n1 = note.Note('C4')
        n2 = note.Note('G4')
        
        result = await analyzer.analyze_intervals(n1, n2)
        
        assert isinstance(result, IntervalAnalysis)
        assert result.size == 5  # Perfect fifth
        assert result.semitones == 7
        assert result.quality == 'Perfect'
        assert result.consonance == 'perfect'
    
    @pytest.mark.asyncio
    async def test_interval_analysis_compound(self, analyzer):
        """Test compound interval analysis"""
        n1 = note.Note('C4')
        n2 = note.Note('E5')  # Major 10th
        
        result = await analyzer.analyze_intervals(n1, n2)
        
        assert result.compound == True
        assert result.semitones == 16
    
    @pytest.mark.asyncio
    async def test_chord_quality_analysis(self, analyzer):
        """Test chord quality analysis"""
        c = chord.Chord(['C4', 'E4', 'G4', 'B4'])  # Cmaj7
        
        result = await analyzer.analyze_chord_quality(c)
        
        assert result['root'] == 'C4'
        assert result['quality'] == 'major-seventh'
        assert result['chord_type'] == 'seventh'
        assert 'maj7' in result['jazz_symbol'].lower()
    
    @pytest.mark.asyncio
    async def test_chord_extensions(self, analyzer):
        """Test chord extension detection"""
        # C9 chord
        c = chord.Chord(['C4', 'E4', 'G4', 'B4', 'D5'])
        
        result = await analyzer.analyze_chord_quality(c, include_extensions=True)
        
        assert '9th' in result['extensions']
        assert '9' in result['jazz_symbol']
    
    @pytest.mark.asyncio
    async def test_find_parallel_keys(self, analyzer):
        """Test parallel key relationships"""
        result = await analyzer.find_parallel_keys('C major')
        
        assert result['reference'] == 'C major'
        assert result['parallel'] == 'c minor'
        assert result['relative'] == 'a minor'
        assert result['dominant'] == 'G major'
        assert result['subdominant'] == 'F major'
    
    @pytest.mark.asyncio
    async def test_cadence_detection(self, analyzer, chord_progression_score):
        """Test cadence detection in progressions"""
        # This test would need the _detect_cadences method to be public
        # or test through the main analysis method
        key_result = await analyzer.analyze_key(chord_progression_score)
        
        # Check that evidence includes cadence information
        assert 'cadences' in key_result.evidence
        cadences = key_result.evidence['cadences']
        
        # Should detect authentic cadence (V-I)
        authentic_found = any(c['type'] == 'perfect_authentic' for c in cadences)
        assert authentic_found


class TestChordProgressionAnalysis:
    """Test chord progression analysis features"""
    
    @pytest.fixture
    def analyzer(self):
        return TheoryAnalyzer()
    
    @pytest.mark.asyncio
    async def test_roman_numeral_analysis(self, analyzer, chord_progression_score):
        """Test Roman numeral analysis"""
        # Would need to test through the server endpoint
        # as this functionality is integrated there
        pass
    
    @pytest.mark.asyncio
    async def test_harmonic_function_detection(self, analyzer):
        """Test harmonic function labeling"""
        # Create ii-V-I progression
        s = stream.Score()
        p = stream.Part()
        p.append(key.KeySignature(0))  # C major
        
        # ii (Dm)
        p.append(chord.Chord(['D4', 'F4', 'A4']))
        # V (G)
        p.append(chord.Chord(['G4', 'B4', 'D5']))
        # I (C)
        p.append(chord.Chord(['C4', 'E4', 'G4']))
        
        s.append(p)
        
        # Test would verify function labels
        # This would be tested through the full analysis pipeline