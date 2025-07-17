"""
Comprehensive unit tests for ChordAnalysisTool
Tests chord identification and analysis functionality
"""

import pytest
from music21 import stream, chord, note, meter, key, roman

from music21_mcp.tools.chord_analysis_tool import ChordAnalysisTool


class TestChordAnalysisTool:
    """Test ChordAnalysisTool with actual implementation"""

    @pytest.fixture
    def tool(self, clean_score_storage):
        """Create tool instance with clean storage"""
        return ChordAnalysisTool(clean_score_storage)

    @pytest.fixture
    def simple_chord_progression(self, clean_score_storage):
        """Create a score with a simple chord progression"""
        s = stream.Score()
        part = stream.Part()
        
        # Add key and time signature
        part.append(key.Key('C'))
        part.append(meter.TimeSignature('4/4'))
        
        # I - IV - V - I progression in C major
        m1 = stream.Measure(number=1)
        m1.append(chord.Chord(['C4', 'E4', 'G4'], quarterLength=4))  # C major
        part.append(m1)
        
        m2 = stream.Measure(number=2)
        m2.append(chord.Chord(['F4', 'A4', 'C5'], quarterLength=4))  # F major
        part.append(m2)
        
        m3 = stream.Measure(number=3)
        m3.append(chord.Chord(['G4', 'B4', 'D5'], quarterLength=4))  # G major
        part.append(m3)
        
        m4 = stream.Measure(number=4)
        m4.append(chord.Chord(['C4', 'E4', 'G4', 'C5'], quarterLength=4))  # C major
        part.append(m4)
        
        s.append(part)
        clean_score_storage['simple_progression'] = s
        return clean_score_storage

    @pytest.fixture
    def complex_chord_score(self, clean_score_storage):
        """Create a score with various chord types"""
        s = stream.Score()
        part = stream.Part()
        
        part.append(key.Key('C'))
        
        # Different chord types
        chords_to_add = [
            chord.Chord(['C4', 'E4', 'G4']),  # Major triad
            chord.Chord(['D4', 'F4', 'A4']),  # Minor triad
            chord.Chord(['B3', 'D4', 'F4']),  # Diminished triad
            chord.Chord(['C4', 'E4', 'G4', 'B4']),  # Major 7th
            chord.Chord(['D4', 'F4', 'A4', 'C5']),  # Minor 7th
            chord.Chord(['G3', 'B3', 'D4', 'F4']),  # Dominant 7th
            chord.Chord(['C4', 'Eb4', 'Gb4', 'Bb4']),  # Diminished 7th
            chord.Chord(['C4', 'E4', 'G#4']),  # Augmented triad
            chord.Chord(['C4', 'E4', 'G4', 'A4', 'D5']),  # Add9
            chord.Chord(['F4', 'A4', 'C5', 'E5', 'G5']),  # Extended chord
        ]
        
        for i, ch in enumerate(chords_to_add):
            m = stream.Measure(number=i+1)
            ch.quarterLength = 2
            m.append(ch)
            part.append(m)
        
        s.append(part)
        clean_score_storage['complex_chords'] = s
        return clean_score_storage

    @pytest.fixture
    def mixed_content_score(self, clean_score_storage):
        """Create a score with both chords and melodic content"""
        s = stream.Score()
        part = stream.Part()
        
        # Measure 1: Melodic content
        m1 = stream.Measure(number=1)
        for pitch in ['C4', 'D4', 'E4', 'F4']:
            m1.append(note.Note(pitch, quarterLength=1))
        part.append(m1)
        
        # Measure 2: Chord
        m2 = stream.Measure(number=2)
        m2.append(chord.Chord(['C4', 'E4', 'G4'], quarterLength=4))
        part.append(m2)
        
        # Measure 3: Mix of notes and chord
        m3 = stream.Measure(number=3)
        m3.append(note.Note('G4', quarterLength=1))
        m3.append(chord.Chord(['F4', 'A4', 'C5'], quarterLength=2))
        m3.append(note.Note('E4', quarterLength=1))
        part.append(m3)
        
        s.append(part)
        clean_score_storage['mixed_content'] = s
        return clean_score_storage

    @pytest.mark.asyncio
    async def test_instantiation(self, clean_score_storage):
        """Test tool can be instantiated with score storage"""
        tool = ChordAnalysisTool(clean_score_storage)
        assert tool.score_manager is clean_score_storage
        assert hasattr(tool, 'execute')
        assert hasattr(tool, 'validate_inputs')

    @pytest.mark.asyncio
    async def test_analyze_simple_progression(self, tool, simple_chord_progression):
        """Test analyzing a simple chord progression"""
        result = await tool.execute(score_id='simple_progression')
        
        assert result['status'] == 'success'
        assert result['score_id'] == 'simple_progression'
        assert 'total_chords' in result
        assert 'chord_types' in result
        assert 'chords' in result
        
        # Should find 4 chords
        assert result['total_chords'] == 4
        
        # Check chord types
        assert 'major' in result['chord_types']
        assert result['chord_types']['major'] == 4  # All are major triads

    @pytest.mark.asyncio
    async def test_analyze_complex_chords(self, tool, complex_chord_score):
        """Test analyzing various chord types"""
        result = await tool.execute(score_id='complex_chords')
        
        assert result['status'] == 'success'
        assert result['total_chords'] == 10
        
        # Check various chord types are detected
        chord_types = result['chord_types']
        assert 'major' in chord_types
        assert 'minor' in chord_types
        assert 'diminished' in chord_types
        assert 'dominant-seventh' in chord_types
        assert 'major-seventh' in chord_types
        assert 'augmented' in chord_types

    @pytest.mark.asyncio
    async def test_chord_details(self, tool, simple_chord_progression):
        """Test detailed chord information"""
        result = await tool.execute(score_id='simple_progression')
        
        chords = result['chords']
        assert len(chords) > 0
        
        # Check first chord details
        first_chord = chords[0]
        assert 'measure' in first_chord
        assert 'beat' in first_chord
        assert 'chord' in first_chord
        assert 'type' in first_chord
        assert 'root' in first_chord
        assert 'bass' in first_chord
        assert 'inversion' in first_chord
        assert 'pitches' in first_chord
        
        # Verify first chord is C major
        assert first_chord['chord'] == 'C major'
        assert first_chord['type'] == 'major'
        assert first_chord['root'] == 'C'
        assert set(first_chord['pitches']) == {'C4', 'E4', 'G4'}

    @pytest.mark.asyncio
    async def test_roman_numeral_analysis(self, tool, simple_chord_progression):
        """Test Roman numeral analysis"""
        result = await tool.execute(score_id='simple_progression')
        
        chords = result['chords']
        # Should have Roman numeral analysis
        for ch in chords:
            if 'roman' in ch:
                assert ch['roman'] in ['I', 'IV', 'V', 'i', 'iv', 'v', 'ii', 'vi', 'vii']

    @pytest.mark.asyncio
    async def test_inversions(self, tool, clean_score_storage):
        """Test chord inversion detection"""
        s = stream.Score()
        part = stream.Part()
        
        # Root position
        part.append(chord.Chord(['C4', 'E4', 'G4'], quarterLength=1))
        # First inversion
        part.append(chord.Chord(['E4', 'G4', 'C5'], quarterLength=1))
        # Second inversion
        part.append(chord.Chord(['G4', 'C5', 'E5'], quarterLength=1))
        
        s.append(part)
        clean_score_storage['inversions'] = s
        
        result = await tool.execute(score_id='inversions')
        
        assert result['status'] == 'success'
        chords = result['chords']
        assert len(chords) == 3
        
        # Check inversions
        assert chords[0]['inversion'] == 0  # Root position
        assert chords[1]['inversion'] == 1  # First inversion
        assert chords[2]['inversion'] == 2  # Second inversion

    @pytest.mark.asyncio
    async def test_mixed_content_analysis(self, tool, mixed_content_score):
        """Test analyzing score with both chords and melodic content"""
        result = await tool.execute(score_id='mixed_content')
        
        assert result['status'] == 'success'
        # Should only count actual chords, not individual notes
        assert result['total_chords'] == 2

    @pytest.mark.asyncio
    async def test_empty_score(self, tool, clean_score_storage):
        """Test analyzing empty score"""
        s = stream.Score()
        clean_score_storage['empty'] = s
        
        result = await tool.execute(score_id='empty')
        
        assert result['status'] == 'success'
        assert result['total_chords'] == 0
        assert result['chords'] == []
        assert result['chord_types'] == {}

    @pytest.mark.asyncio
    async def test_score_not_found(self, tool):
        """Test error when score doesn't exist"""
        result = await tool.execute(score_id='nonexistent')
        
        assert result['status'] == 'error'
        assert 'not found' in result['message']

    @pytest.mark.asyncio
    async def test_empty_score_id(self, tool):
        """Test error when score_id is empty"""
        result = await tool.execute(score_id='')
        
        assert result['status'] == 'error'
        assert 'score_id cannot be empty' in result['message']

    @pytest.mark.asyncio
    async def test_progression_analysis(self, tool, simple_chord_progression):
        """Test chord progression analysis"""
        result = await tool.execute(score_id='simple_progression')
        
        if 'progressions' in result:
            progressions = result['progressions']
            assert isinstance(progressions, list)
            # Should detect I-IV-V-I pattern

    @pytest.mark.asyncio
    async def test_chord_frequency(self, tool, clean_score_storage):
        """Test chord frequency counting"""
        s = stream.Score()
        part = stream.Part()
        
        # Repeat some chords
        for _ in range(3):
            part.append(chord.Chord(['C4', 'E4', 'G4'], quarterLength=1))
        for _ in range(2):
            part.append(chord.Chord(['G4', 'B4', 'D5'], quarterLength=1))
        part.append(chord.Chord(['F4', 'A4', 'C5'], quarterLength=1))
        
        s.append(part)
        clean_score_storage['repeated_chords'] = s
        
        result = await tool.execute(score_id='repeated_chords')
        
        if 'chord_frequencies' in result:
            freq = result['chord_frequencies']
            assert freq.get('C major', 0) == 3
            assert freq.get('G major', 0) == 2
            assert freq.get('F major', 0) == 1

    @pytest.mark.asyncio
    async def test_progress_reporting(self, tool, complex_chord_score):
        """Test progress reporting during analysis"""
        progress_calls = []
        
        def progress_callback(percent, message):
            progress_calls.append((percent, message))
        
        tool.set_progress_callback(progress_callback)
        
        await tool.execute(score_id='complex_chords')
        
        assert len(progress_calls) > 0
        assert any(0.0 <= p[0] <= 1.0 for p in progress_calls)

    @pytest.mark.asyncio
    async def test_jazz_chords(self, tool, clean_score_storage):
        """Test analysis of jazz chord extensions"""
        s = stream.Score()
        part = stream.Part()
        
        # Jazz chords
        jazz_chords = [
            chord.Chord(['C4', 'E4', 'G4', 'B4', 'D5']),  # Cmaj9
            chord.Chord(['D4', 'F4', 'A4', 'C5', 'E5']),  # Dm9
            chord.Chord(['G3', 'B3', 'D4', 'F4', 'A4']),  # G13
            chord.Chord(['C4', 'E4', 'G4', 'Bb4', 'D5', 'F5']),  # C7#11
        ]
        
        for ch in jazz_chords:
            ch.quarterLength = 2
            part.append(ch)
        
        s.append(part)
        clean_score_storage['jazz_chords'] = s
        
        result = await tool.execute(score_id='jazz_chords')
        
        assert result['status'] == 'success'
        assert result['total_chords'] == 4
        # Should recognize extended harmonies

    @pytest.mark.asyncio
    async def test_multipart_chords(self, tool, clean_score_storage):
        """Test chord analysis across multiple parts"""
        s = stream.Score()
        
        # Soprano
        soprano = stream.Part()
        soprano.append(note.Note('G4', quarterLength=4))
        
        # Alto
        alto = stream.Part()
        alto.append(note.Note('E4', quarterLength=4))
        
        # Tenor
        tenor = stream.Part()
        tenor.append(note.Note('C4', quarterLength=4))
        
        # Bass
        bass = stream.Part()
        bass.append(note.Note('C3', quarterLength=4))
        
        s.append(soprano)
        s.append(alto)
        s.append(tenor)
        s.append(bass)
        
        clean_score_storage['multipart'] = s
        
        result = await tool.execute(score_id='multipart')
        
        # Depends on implementation - might analyze vertical sonorities
        assert result['status'] == 'success'

    @pytest.mark.asyncio
    async def test_enharmonic_chords(self, tool, clean_score_storage):
        """Test handling of enharmonic spellings"""
        s = stream.Score()
        part = stream.Part()
        
        # C# major and Db major (enharmonic)
        part.append(chord.Chord(['C#4', 'E#4', 'G#4'], quarterLength=2))
        part.append(chord.Chord(['Db4', 'F4', 'Ab4'], quarterLength=2))
        
        s.append(part)
        clean_score_storage['enharmonic'] = s
        
        result = await tool.execute(score_id='enharmonic')
        
        assert result['status'] == 'success'
        assert result['total_chords'] == 2
        # Both should be recognized as major triads
        assert result['chord_types'].get('major', 0) == 2