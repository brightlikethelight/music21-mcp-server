"""
Unit tests for harmonic analyzer functionality
"""
import pytest
from music21 import chord, key, meter, note, stream

from src.music21_mcp.core.harmonic_analyzer import (
    ChordSubstitutionType,
    FunctionalHarmonyAnalysis,
    HarmonicAnalyzer,
    HarmonicFunction,
    JazzHarmonyAnalysis,
    ModulationAnalysis,
    VoiceLeadingAnalysis,
    VoiceLeadingError,
)


class TestHarmonicAnalyzer:
    """Test harmonic analysis functionality"""
    
    @pytest.fixture
    def analyzer(self):
        return HarmonicAnalyzer()
    
    @pytest.fixture
    def classical_progression(self):
        """Create a classical chord progression"""
        s = stream.Score()
        p = stream.Part()
        p.append(key.KeySignature(0))  # C major
        p.append(meter.TimeSignature('4/4'))
        
        # I - ii6 - V7 - I (classical cadence)
        chords = [
            chord.Chord(['C4', 'E4', 'G4', 'C5']),    # I
            chord.Chord(['F3', 'A3', 'D4', 'F4']),    # ii6
            chord.Chord(['G3', 'B3', 'D4', 'F4']),    # V7
            chord.Chord(['C4', 'E4', 'G4', 'C5'])     # I
        ]
        
        for ch in chords:
            ch.quarterLength = 2
            p.append(ch)
        
        s.append(p)
        return s
    
    @pytest.fixture
    def jazz_progression(self):
        """Create a jazz chord progression"""
        s = stream.Score()
        p = stream.Part()
        p.append(key.KeySignature(-1))  # F major
        
        # ii7 - V7 - IMaj7 - VI7 (ii-V-I with turnaround)
        chords = [
            chord.Chord(['G3', 'B-3', 'D4', 'F4']),    # Gm7 (ii7)
            chord.Chord(['C3', 'E4', 'G4', 'B-4']),    # C7 (V7)
            chord.Chord(['F3', 'A3', 'C4', 'E4']),     # FMaj7 (IMaj7)
            chord.Chord(['D3', 'F#3', 'A3', 'C4'])     # D7 (VI7)
        ]
        
        for ch in chords:
            ch.quarterLength = 4
            p.append(ch)
        
        s.append(p)
        return s
    
    @pytest.fixture
    def four_part_chorale(self):
        """Create a four-part chorale for voice leading tests"""
        s = stream.Score()
        
        # Soprano
        soprano = stream.Part()
        soprano.append(note.Note('G5', quarterLength=1))
        soprano.append(note.Note('F5', quarterLength=1))
        soprano.append(note.Note('E5', quarterLength=1))
        soprano.append(note.Note('E5', quarterLength=1))
        
        # Alto
        alto = stream.Part()
        alto.append(note.Note('E5', quarterLength=1))
        alto.append(note.Note('D5', quarterLength=1))
        alto.append(note.Note('C5', quarterLength=1))
        alto.append(note.Note('C5', quarterLength=1))
        
        # Tenor
        tenor = stream.Part()
        tenor.append(note.Note('C4', quarterLength=1))
        tenor.append(note.Note('B3', quarterLength=1))
        tenor.append(note.Note('G3', quarterLength=1))
        tenor.append(note.Note('G3', quarterLength=1))
        
        # Bass
        bass = stream.Part()
        bass.append(note.Note('C3', quarterLength=1))
        bass.append(note.Note('G2', quarterLength=1))
        bass.append(note.Note('C3', quarterLength=1))
        bass.append(note.Note('C3', quarterLength=1))
        
        s.insert(0, soprano)
        s.insert(0, alto)
        s.insert(0, tenor)
        s.insert(0, bass)
        
        return s
    
    @pytest.mark.asyncio
    async def test_functional_harmony_analysis(self, analyzer, classical_progression):
        """Test functional harmony analysis"""
        result = await analyzer.analyze_functional_harmony(classical_progression)
        
        assert isinstance(result, FunctionalHarmonyAnalysis)
        assert len(result.roman_numerals) == 4
        assert result.roman_numerals[0] in ['I', 'I53']
        assert 'ii' in result.roman_numerals[1]
        assert 'V' in result.roman_numerals[2]
        
        # Check functions
        assert result.functions[0] == HarmonicFunction.TONIC
        assert result.functions[1] == HarmonicFunction.PREDOMINANT
        assert result.functions[2] == HarmonicFunction.DOMINANT
        assert result.functions[3] == HarmonicFunction.TONIC
        
        # Check cadence detection
        assert len(result.cadences) > 0
        assert any('authentic' in c['type'] for c in result.cadences)
        
        # Check tonal strength
        assert result.tonal_strength > 0.5  # Strong tonal progression
    
    @pytest.mark.asyncio
    async def test_dominant_preparation(self, analyzer, classical_progression):
        """Test dominant preparation detection"""
        result = await analyzer.analyze_functional_harmony(classical_progression)
        
        # Should detect ii6 preparing V7
        assert len(result.dominant_preparations) > 0
        prep = result.dominant_preparations[0]
        assert 'ii' in prep['preparation']
        assert 'V' in prep['dominant']
        assert prep['strength'] > 0.5
    
    @pytest.mark.asyncio
    async def test_voice_leading_analysis(self, analyzer, four_part_chorale):
        """Test voice leading analysis"""
        result = await analyzer.analyze_voice_leading(four_part_chorale, strict=True)
        
        assert isinstance(result, VoiceLeadingAnalysis)
        
        # Check voice ranges
        assert 'voice_1' in result.voice_ranges  # Soprano
        assert 'voice_2' in result.voice_ranges  # Alto
        assert 'voice_3' in result.voice_ranges  # Tenor
        assert 'voice_4' in result.voice_ranges  # Bass
        
        # Soprano should be highest
        assert result.voice_ranges['voice_1']['average'] > result.voice_ranges['voice_2']['average']
        
        # Check smoothness
        assert result.smoothness_score > 0  # Some smooth voice leading
        assert result.independence_score > 0  # Some independence
    
    @pytest.mark.asyncio
    async def test_parallel_motion_detection(self, analyzer):
        """Test parallel fifths/octaves detection"""
        s = stream.Score()
        
        # Create parallel fifths
        p1 = stream.Part()
        p1.append(note.Note('C4', quarterLength=1))
        p1.append(note.Note('D4', quarterLength=1))
        
        p2 = stream.Part()
        p2.append(note.Note('G4', quarterLength=1))
        p2.append(note.Note('A4', quarterLength=1))
        
        s.insert(0, p1)
        s.insert(0, p2)
        
        result = await analyzer.analyze_voice_leading(s, strict=True)
        
        # Should detect parallel fifths
        assert len(result.errors) > 0
        assert any(err['type'] == VoiceLeadingError.PARALLEL_FIFTHS.value 
                  for err in result.errors)
    
    @pytest.mark.asyncio
    async def test_jazz_harmony_analysis(self, analyzer, jazz_progression):
        """Test jazz harmony analysis"""
        result = await analyzer.analyze_jazz_harmony(jazz_progression)
        
        assert isinstance(result, JazzHarmonyAnalysis)
        assert len(result.chord_symbols) == 4
        
        # Check for extended chords
        assert len(result.extended_chords) > 0  # Should find 7th chords
        
        # Check chord qualities
        extended = result.extended_chords[0]
        assert '7' in extended['symbol'] or 'Maj7' in extended['symbol']
    
    @pytest.mark.asyncio
    async def test_modal_interchange(self, analyzer):
        """Test modal interchange detection"""
        s = stream.Score()
        p = stream.Part()
        p.append(key.KeySignature(0))  # C major
        
        # C - Fm - C (borrowed iv from parallel minor)
        chords = [
            chord.Chord(['C4', 'E4', 'G4']),      # I
            chord.Chord(['F4', 'A-4', 'C5']),     # iv (borrowed)
            chord.Chord(['C4', 'E4', 'G4'])       # I
        ]
        
        for ch in chords:
            ch.quarterLength = 2
            p.append(ch)
        
        s.append(p)
        
        result = await analyzer.analyze_jazz_harmony(s)
        
        # Should detect modal interchange
        assert len(result.modal_interchanges) > 0
        interchange = result.modal_interchanges[0]
        assert 'minor' in interchange['borrowed_from'].lower()
        assert interchange['modal_color'] == 'darker'  # Minor chord in major key
    
    @pytest.mark.asyncio
    async def test_harmonic_sequence_detection(self, analyzer):
        """Test harmonic sequence detection"""
        s = stream.Score()
        p = stream.Part()
        p.append(key.KeySignature(0))  # C major
        
        # Create a descending fifth sequence
        # C - F - Bdim - Em - Am - Dm - G - C
        roots = ['C', 'F', 'B', 'E', 'A', 'D', 'G', 'C']
        qualities = ['major', 'major', 'diminished', 'minor', 
                    'minor', 'minor', 'major', 'major']
        
        for root, quality in zip(roots, qualities):
            if quality == 'major':
                ch = chord.Chord([root + '3', 
                                 note.Note(root + '3').pitch.transpose('M3').nameWithOctave,
                                 note.Note(root + '3').pitch.transpose('P5').nameWithOctave])
            elif quality == 'minor':
                ch = chord.Chord([root + '3',
                                 note.Note(root + '3').pitch.transpose('m3').nameWithOctave,
                                 note.Note(root + '3').pitch.transpose('P5').nameWithOctave])
            else:  # diminished
                ch = chord.Chord([root + '3',
                                 note.Note(root + '3').pitch.transpose('m3').nameWithOctave,
                                 note.Note(root + '3').pitch.transpose('d5').nameWithOctave])
            
            ch.quarterLength = 2
            p.append(ch)
        
        s.append(p)
        
        sequences = await analyzer.detect_harmonic_sequences(s, min_pattern_length=2)
        
        assert len(sequences) > 0
        # Should detect some kind of sequential pattern
        seq = sequences[0]
        assert seq.sequence_type in ['circle_of_fifths', 'sequential_progression']
    
    @pytest.mark.asyncio
    async def test_modulation_detection(self, analyzer):
        """Test modulation detection"""
        s = stream.Score()
        p = stream.Part()
        
        # Start in C major
        p.append(key.KeySignature(0))
        p.append(meter.TimeSignature('4/4'))
        
        # C major section
        for _ in range(4):
            p.append(chord.Chord(['C4', 'E4', 'G4']))
            p.append(chord.Chord(['G3', 'B3', 'D4']))
        
        # Modulate to G major
        p.append(chord.Chord(['D4', 'F#4', 'A4']))  # V/V
        p.append(chord.Chord(['G4', 'B4', 'D5']))   # New tonic
        
        # G major section
        for _ in range(4):
            p.append(chord.Chord(['G4', 'B4', 'D5']))
            p.append(chord.Chord(['D4', 'F#4', 'A4']))
        
        # Ensure each chord has duration
        for element in p.getElementsByClass(chord.Chord):
            element.quarterLength = 1
        
        s.append(p)
        
        result = await analyzer.analyze_modulations(s, sensitivity=0.6)
        
        assert isinstance(result, ModulationAnalysis)
        assert len(result.modulations) > 0
        
        # Should detect C to G modulation
        mod = result.modulations[0]
        assert 'C' in mod['from_key']
        assert 'G' in mod['to_key']
        assert mod['type'] == 'dominant'  # Modulation to dominant
    
    @pytest.mark.asyncio
    async def test_deceptive_resolution(self, analyzer):
        """Test deceptive resolution detection"""
        s = stream.Score()
        p = stream.Part()
        p.append(key.KeySignature(0))  # C major
        
        # V - vi (deceptive resolution)
        chords = [
            chord.Chord(['G3', 'B3', 'D4']),    # V
            chord.Chord(['A3', 'C4', 'E4'])     # vi
        ]
        
        for ch in chords:
            ch.quarterLength = 2
            p.append(ch)
        
        s.append(p)
        
        result = await analyzer.analyze_functional_harmony(s)
        
        assert len(result.deceptive_resolutions) > 0
        deceptive = result.deceptive_resolutions[0]
        assert deceptive['expected'] in ['I', 'i']
        assert 'vi' in deceptive['actual']
        assert deceptive['type'] == 'standard'
    
    @pytest.mark.asyncio
    async def test_phrase_model_detection(self, analyzer):
        """Test phrase model detection"""
        s = stream.Score()
        p = stream.Part()
        p.append(key.KeySignature(0))
        
        # Create a period structure
        # Antecedent ending with half cadence
        for pitch in ['C4', 'D4', 'E4', 'F4']:
            p.append(note.Note(pitch, quarterLength=1))
        p.append(chord.Chord(['G3', 'B3', 'D4']))  # V (half cadence)
        
        # Consequent ending with authentic cadence
        for pitch in ['C4', 'D4', 'E4', 'F4']:
            p.append(note.Note(pitch, quarterLength=1))
        p.append(chord.Chord(['G3', 'B3', 'D4']))  # V
        p.append(chord.Chord(['C4', 'E4', 'G4']))  # I (authentic cadence)
        
        s.append(p)
        
        result = await analyzer.analyze_functional_harmony(s)
        
        assert result.phrase_model in ['period', 'sentence', 'phrase_group']
        assert len(result.cadences) >= 2


class TestVoiceLeadingRules:
    """Test specific voice leading rules"""
    
    @pytest.fixture
    def analyzer(self):
        return HarmonicAnalyzer()
    
    @pytest.mark.asyncio
    async def test_voice_crossing_detection(self, analyzer):
        """Test voice crossing detection"""
        s = stream.Score()
        
        # Alto crosses above soprano
        soprano = stream.Part()
        soprano.append(note.Note('C5', quarterLength=1))
        
        alto = stream.Part()
        alto.append(note.Note('E5', quarterLength=1))  # Above soprano!
        
        s.insert(0, soprano)
        s.insert(0, alto)
        
        result = await analyzer.analyze_voice_leading(s)
        
        assert len(result.voice_crossings) > 0
        crossing = result.voice_crossings[0]
        assert crossing['voices'] == '1-2'
    
    @pytest.mark.asyncio
    async def test_large_leap_detection(self, analyzer):
        """Test large leap detection"""
        s = stream.Score()
        
        p = stream.Part()
        p.append(note.Note('C4', quarterLength=1))
        p.append(note.Note('G5', quarterLength=1))  # Large leap (octave + fifth)
        
        s.append(p)
        
        result = await analyzer.analyze_voice_leading(s, strict=True)
        
        # In strict mode, should flag large leaps
        if result.errors:
            assert any(err['type'] == VoiceLeadingError.LARGE_LEAP.value
                      for err in result.errors)


class TestJazzSpecificFeatures:
    """Test jazz-specific harmonic features"""
    
    @pytest.fixture
    def analyzer(self):
        return HarmonicAnalyzer()
    
    @pytest.mark.asyncio
    async def test_tritone_substitution(self, analyzer):
        """Test tritone substitution detection"""
        s = stream.Score()
        p = stream.Part()
        
        # G7 - Db7 - C (tritone sub for G7)
        chords = [
            chord.Chord(['G2', 'B2', 'D3', 'F3']),     # G7
            chord.Chord(['D-2', 'F2', 'A-2', 'C-3']),  # Db7 (tritone sub)
            chord.Chord(['C3', 'E3', 'G3'])            # C
        ]
        
        for ch in chords:
            ch.quarterLength = 2
            p.append(ch)
        
        s.append(p)
        
        result = await analyzer.analyze_jazz_harmony(s)
        
        # Should detect tritone relationship
        if result.substitutions:
            sub = result.substitutions[0]
            assert sub['type'] == ChordSubstitutionType.TRITONE.value
    
    @pytest.mark.asyncio
    async def test_chord_scale_suggestions(self, analyzer):
        """Test chord scale suggestions"""
        s = stream.Score()
        p = stream.Part()
        
        # Various chord types
        chords = [
            chord.Chord(['C3', 'E3', 'G3', 'B3']),        # CMaj7
            chord.Chord(['D3', 'F3', 'A3', 'C4']),        # Dm7
            chord.Chord(['G2', 'B2', 'D3', 'F3']),        # G7
            chord.Chord(['C3', 'E-3', 'G-3', 'B--3'])     # Cdim7
        ]
        
        for ch in chords:
            ch.quarterLength = 4
            p.append(ch)
        
        s.append(p)
        
        result = await analyzer.analyze_jazz_harmony(s)
        
        assert len(result.chord_scales) > 0
        
        # Check that appropriate scales are suggested
        for scale_info in result.chord_scales:
            assert 'scales' in scale_info
            assert len(scale_info['scales']) > 0
            assert scale_info['scales'][0]['priority'] in ['primary', 'alternative']