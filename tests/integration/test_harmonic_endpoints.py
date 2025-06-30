"""
Integration tests for harmonic analysis MCP endpoints
"""
import sys

import pytest
from music21 import chord, key, meter, note, stream

sys.path.insert(0, 'src')
from music21_mcp.server import (
    detect_harmonic_sequences,
    functional_harmony_analysis,
    jazz_harmony_analysis,
    modulation_analysis,
    score_manager,
    voice_leading_analysis,
)


class TestHarmonicEndpoints:
    """Test harmonic analysis endpoints"""
    
    @pytest.fixture
    def setup_classical_score(self):
        """Create and import a classical score"""
        s = stream.Score()
        p = stream.Part()
        p.append(key.KeySignature(0))  # C major
        p.append(meter.TimeSignature('4/4'))
        
        # Classical progression with cadence
        # I - vi - ii6 - V7 - I
        chords = [
            chord.Chord(['C4', 'E4', 'G4', 'C5']),     # I
            chord.Chord(['A3', 'C4', 'E4', 'A4']),     # vi
            chord.Chord(['F3', 'A3', 'D4', 'F4']),     # ii6
            chord.Chord(['G3', 'B3', 'D4', 'F4']),     # V7
            chord.Chord(['C4', 'E4', 'G4', 'C5'])      # I
        ]
        
        for i, ch in enumerate(chords):
            ch.quarterLength = 2
            ch.offset = i * 2
            p.append(ch)
        
        s.append(p)
        score_manager.add_score("classical_test", s, {})
        return "classical_test"
    
    @pytest.fixture
    def setup_jazz_score(self):
        """Create and import a jazz score"""
        s = stream.Score()
        p = stream.Part()
        p.append(key.KeySignature(-2))  # Bb major
        
        # Jazz progression: ii7 - V7alt - IMaj7 - VIm7
        chords = [
            chord.Chord(['C3', 'E-3', 'G3', 'B-3']),      # Cm7
            chord.Chord(['F3', 'A3', 'C4', 'E-4', 'G#4']), # F7alt
            chord.Chord(['B-2', 'D3', 'F3', 'A3']),       # BbMaj7
            chord.Chord(['G3', 'B-3', 'D4', 'F4'])        # Gm7
        ]
        
        for i, ch in enumerate(chords):
            ch.quarterLength = 4
            ch.offset = i * 4
            p.append(ch)
        
        s.append(p)
        score_manager.add_score("jazz_test", s, {})
        return "jazz_test"
    
    @pytest.fixture
    def setup_chorale_score(self):
        """Create a four-part chorale for voice leading tests"""
        s = stream.Score()
        
        # SATB parts
        soprano = stream.Part()
        alto = stream.Part()
        tenor = stream.Part()
        bass = stream.Part()
        
        # Simple progression with voice leading
        # Measure 1
        soprano.append(note.Note('E5', quarterLength=1))
        alto.append(note.Note('C5', quarterLength=1))
        tenor.append(note.Note('G4', quarterLength=1))
        bass.append(note.Note('C3', quarterLength=1))
        
        # Measure 2 - with parallel fifths between soprano and alto
        soprano.append(note.Note('F5', quarterLength=1))
        alto.append(note.Note('D5', quarterLength=1))
        tenor.append(note.Note('A4', quarterLength=1))
        bass.append(note.Note('D3', quarterLength=1))
        
        # Measure 3
        soprano.append(note.Note('G5', quarterLength=1))
        alto.append(note.Note('E5', quarterLength=1))
        tenor.append(note.Note('C5', quarterLength=1))
        bass.append(note.Note('C3', quarterLength=1))
        
        s.insert(0, soprano)
        s.insert(0, alto)
        s.insert(0, tenor)
        s.insert(0, bass)
        
        score_manager.add_score("chorale_test", s, {})
        return "chorale_test"
    
    @pytest.mark.asyncio
    async def test_functional_harmony_endpoint(self, setup_classical_score):
        """Test functional harmony analysis endpoint"""
        result = await functional_harmony_analysis(
            score_id=setup_classical_score,
            window_size=4
        )
        
        assert result['status'] == 'success'
        assert 'roman_numerals' in result
        assert len(result['roman_numerals']) == 5
        
        # Check functions
        assert 'functions' in result
        assert result['functions'][0] == 'T'  # Tonic
        assert any(f == 'D' for f in result['functions'])  # Has dominant
        
        # Check cadences
        assert 'cadences' in result
        assert len(result['cadences']) > 0
        
        # Check tonal strength
        assert 'tonal_strength' in result
        assert result['tonal_strength'] > 0.5  # Strong tonal music
    
    @pytest.mark.asyncio
    async def test_predominant_detection(self, setup_classical_score):
        """Test predominant chord detection"""
        result = await functional_harmony_analysis(score_id=setup_classical_score)
        
        assert 'predominant_chords' in result
        assert len(result['predominant_chords']) > 0
        
        # ii6 should be detected as predominant
        pred = result['predominant_chords'][0]
        assert 'ii' in pred['chord']
        assert pred['function'] == 'predominant'
    
    @pytest.mark.asyncio
    async def test_voice_leading_endpoint(self, setup_chorale_score):
        """Test voice leading analysis endpoint"""
        result = await voice_leading_analysis(
            score_id=setup_chorale_score,
            strict=True
        )
        
        assert result['status'] == 'success'
        
        # Check voice ranges
        assert 'voice_ranges' in result
        assert 'voice_1' in result['voice_ranges']  # Soprano
        assert 'voice_4' in result['voice_ranges']  # Bass
        
        # Check for errors (should find parallel fifths)
        assert 'errors' in result
        if result['errors']:
            assert any(err['type'] == 'parallel_fifths' for err in result['errors'])
        
        # Check scores
        assert 'smoothness_score' in result
        assert 'independence_score' in result
        assert 0 <= result['smoothness_score'] <= 1
        assert 0 <= result['independence_score'] <= 1
    
    @pytest.mark.asyncio
    async def test_jazz_harmony_endpoint(self, setup_jazz_score):
        """Test jazz harmony analysis endpoint"""
        result = await jazz_harmony_analysis(
            score_id=setup_jazz_score,
            include_tensions=True
        )
        
        assert result['status'] == 'success'
        
        # Check chord symbols
        assert 'chord_symbols' in result
        assert len(result['chord_symbols']) == 4
        
        # Check extended chords
        assert 'extended_chords' in result
        assert len(result['extended_chords']) > 0  # Should find 7th chords
        
        # Check for altered chord
        for ext in result['extended_chords']:
            if 'F' in ext['symbol']:
                # F7alt should have alterations
                assert 'alterations' in ext or 'extensions' in ext
        
        # Check chord scales
        assert 'chord_scales' in result
        assert len(result['chord_scales']) > 0
    
    @pytest.mark.asyncio
    async def test_harmonic_sequences_endpoint(self):
        """Test harmonic sequence detection endpoint"""
        # Create a score with sequences
        s = stream.Score()
        p = stream.Part()
        p.append(key.KeySignature(0))
        
        # Create a simple sequence: I-V pattern repeated
        pattern = [
            chord.Chord(['C4', 'E4', 'G4']),    # I
            chord.Chord(['G3', 'B3', 'D4']),    # V
        ]
        
        # Repeat pattern 3 times at different pitch levels
        for transp in [0, -2, -4]:  # C, Bb, Ab
            for ch in pattern:
                new_ch = ch.transpose(transp)
                new_ch.quarterLength = 1
                p.append(new_ch)
        
        s.append(p)
        score_manager.add_score("sequence_test", s, {})
        
        result = await detect_harmonic_sequences(
            score_id="sequence_test",
            min_pattern_length=2,
            min_occurrences=2
        )
        
        assert result['status'] == 'success'
        assert 'sequences' in result
        assert result['total_sequences'] > 0
        
        # Should detect the I-V pattern
        seq = result['sequences'][0]
        assert len(seq['pattern']) >= 2
        assert len(seq['occurrences']) >= 2
        assert 'sequence_type' in seq
    
    @pytest.mark.asyncio
    async def test_modulation_analysis_endpoint(self):
        """Test modulation analysis endpoint"""
        # Create score with modulation
        s = stream.Score()
        p = stream.Part()
        
        # Start in C major
        p.append(key.KeySignature(0))
        
        # C major section (4 measures)
        for _ in range(8):
            p.append(chord.Chord(['C4', 'E4', 'G4'], quarterLength=0.5))
            p.append(chord.Chord(['G3', 'B3', 'D4'], quarterLength=0.5))
        
        # Pivot chord and modulation to G major
        p.append(chord.Chord(['D4', 'F#4', 'A4'], quarterLength=1))  # V/V
        
        # G major section (4 measures)
        for _ in range(8):
            p.append(chord.Chord(['G4', 'B4', 'D5'], quarterLength=0.5))
            p.append(chord.Chord(['D4', 'F#4', 'A4'], quarterLength=0.5))
        
        s.append(p)
        score_manager.add_score("modulation_test", s, {})
        
        result = await modulation_analysis(
            score_id="modulation_test",
            sensitivity=0.6
        )
        
        assert result['status'] == 'success'
        assert 'modulations' in result
        assert result['total_modulations'] > 0
        
        # Should detect C to G modulation
        mod = result['modulations'][0]
        assert 'C' in mod['from_key']
        assert 'G' in mod['to_key']
        assert mod['type'] in ['dominant', 'step', 'mediant']
        
        # Check key areas
        assert 'key_areas' in result
        assert len(result['key_areas']) >= 2
    
    @pytest.mark.asyncio
    async def test_nonexistent_score_handling(self):
        """Test error handling for nonexistent scores"""
        result = await functional_harmony_analysis(score_id="nonexistent")
        
        assert result['status'] == 'error'
        assert 'not found' in result['message']
    
    @pytest.mark.asyncio
    async def test_empty_score_handling(self):
        """Test handling of empty scores"""
        # Create empty score
        s = stream.Score()
        score_manager.add_score("empty_test", s, {})
        
        result = await functional_harmony_analysis(score_id="empty_test")
        
        # Should handle gracefully
        assert result['status'] in ['success', 'error']
        if result['status'] == 'success':
            assert result['roman_numerals'] == []


class TestComplexHarmonicScenarios:
    """Test complex harmonic scenarios"""
    
    @pytest.mark.asyncio
    async def test_chromatic_harmony(self):
        """Test chromatic harmony detection"""
        s = stream.Score()
        p = stream.Part()
        p.append(key.KeySignature(0))  # C major
        
        # Chromatic progression with augmented sixth
        chords = [
            chord.Chord(['C4', 'E4', 'G4']),           # I
            chord.Chord(['A-3', 'C4', 'E-4', 'F#4']),  # Ger+6
            chord.Chord(['G3', 'B3', 'D4']),           # V
            chord.Chord(['C4', 'E4', 'G4'])            # I
        ]
        
        for ch in chords:
            ch.quarterLength = 2
            p.append(ch)
        
        s.append(p)
        score_manager.add_score("chromatic_test", s, {})
        
        result = await functional_harmony_analysis(score_id="chromatic_test")
        
        assert result['status'] == 'success'
        # Should handle chromatic chords
        assert len(result['roman_numerals']) == 4
    
    @pytest.mark.asyncio
    async def test_jazz_reharmonization(self):
        """Test jazz reharmonization detection"""
        s = stream.Score()
        p = stream.Part()
        
        # Original: C - Am - Dm - G7
        # Reharmonized: CMaj7 - A7alt - Dm7 - Db7 (tritone sub)
        chords = [
            chord.Chord(['C3', 'E3', 'G3', 'B3']),           # CMaj7
            chord.Chord(['A2', 'C#3', 'E3', 'G3', 'C4']),   # A7alt
            chord.Chord(['D3', 'F3', 'A3', 'C4']),          # Dm7
            chord.Chord(['D-2', 'F2', 'A-2', 'C-3'])        # Db7
        ]
        
        for ch in chords:
            ch.quarterLength = 2
            p.append(ch)
        
        s.append(p)
        score_manager.add_score("reharmonization_test", s, {})
        
        result = await jazz_harmony_analysis(score_id="reharmonization_test")
        
        assert result['status'] == 'success'
        # Should detect the tritone substitution
        if result['substitutions']:
            assert any('tritone' in str(sub.get('type', '')).lower() 
                      for sub in result['substitutions'])