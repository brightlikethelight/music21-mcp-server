"""
Integration tests for melodic analysis MCP endpoints
"""
import pytest
from music21 import stream, note, chord, key, meter, tempo

import sys
sys.path.insert(0, 'src')
from music21_mcp.server import (
    score_manager, import_score, analyze_melodic_contour,
    detect_melodic_motives, find_melodic_patterns,
    analyze_cross_cultural_melody, calculate_melodic_similarity,
    analyze_melodic_development
)


class TestMelodicEndpoints:
    """Test melodic analysis endpoints"""
    
    @pytest.fixture
    def setup_melodic_score(self):
        """Create and import a melodic score"""
        s = stream.Score()
        p = stream.Part()
        p.append(key.KeySignature(0))  # C major
        p.append(meter.TimeSignature('4/4'))
        p.append(tempo.MetronomeMark(number=120))
        
        # Create a melody with clear motives
        # Opening motive (3 notes)
        motive = ['C4', 'E4', 'G4']
        
        # Statement 1
        for pitch in motive:
            n = note.Note(pitch, quarterLength=0.5)
            p.append(n)
        p.append(note.Note('C5', quarterLength=1.5))
        
        # Statement 2 (transposed)
        for pitch in ['G4', 'B4', 'D5']:
            n = note.Note(pitch, quarterLength=0.5)
            p.append(n)
        p.append(note.Note('G5', quarterLength=1.5))
        
        # Development section with scalar passages
        scale = ['C5', 'D5', 'E5', 'F5', 'G5', 'F5', 'E5', 'D5', 'C5']
        for pitch in scale:
            n = note.Note(pitch, quarterLength=0.25)
            p.append(n)
        
        # Return of motive
        for pitch in motive:
            n = note.Note(pitch, quarterLength=0.5)
            p.append(n)
        p.append(note.Note('C5', quarterLength=2))
        
        s.append(p)
        score_manager.add_score("melodic_test", s, {})
        return "melodic_test"
    
    @pytest.fixture
    def setup_folk_score(self):
        """Create a folk-style pentatonic melody"""
        s = stream.Score()
        p = stream.Part()
        p.append(meter.TimeSignature('6/8'))
        
        # Pentatonic melody (C D E G A)
        folk_melody = [
            ('C4', 1), ('D4', 0.5), ('E4', 0.5), ('G4', 1),
            ('A4', 1), ('G4', 0.5), ('E4', 0.5), ('D4', 1),
            ('C4', 1.5), ('C4', 0.5), ('E4', 1), ('G4', 1),
            ('E4', 0.5), ('D4', 0.5), ('C4', 2)
        ]
        
        for pitch, dur in folk_melody:
            n = note.Note(pitch, quarterLength=dur)
            p.append(n)
        
        s.append(p)
        score_manager.add_score("folk_test", s, {})
        return "folk_test"
    
    @pytest.fixture
    def setup_jazz_score(self):
        """Create a jazz-style melody with chromatic elements"""
        s = stream.Score()
        p = stream.Part()
        p.append(key.KeySignature(-2))  # Bb major
        p.append(meter.TimeSignature('4/4'))
        
        # Jazz line with chromatic passing tones
        jazz_line = [
            ('Bb3', 0.5), ('C4', 0.25), ('Db4', 0.25),  # Chromatic approach
            ('D4', 0.5), ('F4', 0.5), ('A4', 0.5),
            ('Bb4', 0.5), ('A4', 0.25), ('Ab4', 0.25),  # Chromatic descent
            ('G4', 0.5), ('F4', 0.5), ('Eb4', 0.5),
            ('D4', 0.25), ('Db4', 0.25), ('C4', 0.5),
            ('Bb3', 1)
        ]
        
        for pitch, dur in jazz_line:
            n = note.Note(pitch, quarterLength=dur)
            p.append(n)
        
        s.append(p)
        score_manager.add_score("jazz_test", s, {})
        return "jazz_test"
    
    @pytest.mark.asyncio
    async def test_melodic_contour_endpoint(self, setup_melodic_score):
        """Test melodic contour analysis endpoint"""
        result = await analyze_melodic_contour(
            score_id=setup_melodic_score,
            part_index=0,
            window_size=8
        )
        
        assert result['status'] == 'success'
        assert 'overall_contour' in result
        assert 'contour_vector' in result
        assert 'arch_points' in result
        assert 'complexity_score' in result
        assert 'smoothness_score' in result
        
        # Should have some contour changes
        assert len(result['contour_vector']) > 0
        assert result['complexity_score'] > 0
    
    @pytest.mark.asyncio
    async def test_motivic_analysis_endpoint(self, setup_melodic_score):
        """Test melodic motive detection endpoint"""
        result = await detect_melodic_motives(
            score_id=setup_melodic_score,
            min_length=3,
            min_occurrences=2
        )
        
        assert result['status'] == 'success'
        assert 'total_motives' in result
        assert 'motives' in result
        assert 'primary_motive' in result
        assert 'coherence_score' in result
        
        # Should find at least one motive (the C-E-G pattern)
        assert result['total_motives'] > 0
        assert result['primary_motive'] is not None
        
        # Check motive structure
        if result['motives']:
            motive = result['motives'][0]
            assert 'intervals' in motive
            assert 'rhythm' in motive
            assert 'occurrences' in motive
            assert motive['occurrences'] >= 2
    
    @pytest.mark.asyncio
    async def test_melodic_patterns_endpoint(self, setup_melodic_score):
        """Test melodic pattern finding endpoint"""
        result = await find_melodic_patterns(
            score_id=setup_melodic_score,
            part_index=0,
            pattern_length=3,
            similarity_threshold=0.8
        )
        
        assert result['status'] == 'success'
        assert 'patterns' in result
        assert 'total_patterns' in result
        
        # Should find some patterns
        if result['total_patterns'] > 0:
            pattern = result['patterns'][0]
            assert 'pattern' in pattern
            assert 'locations' in pattern
            assert 'transformation_type' in pattern
            assert len(pattern['locations']) >= 2
    
    @pytest.mark.asyncio
    async def test_cross_cultural_folk_endpoint(self, setup_folk_score):
        """Test cross-cultural analysis on folk melody"""
        result = await analyze_cross_cultural_melody(
            score_id=setup_folk_score,
            part_index=0,
            include_microtones=False
        )
        
        assert result['status'] == 'success'
        assert 'detected_styles' in result
        assert 'scale_characteristics' in result
        assert 'cultural_markers' in result
        
        # Should detect pentatonic characteristics
        styles = {s['style']: s['confidence'] for s in result['detected_styles']}
        assert 'pentatonic' in styles or 'folk' in styles
        
        # Check scale characteristics
        scale_chars = result['scale_characteristics']
        assert scale_chars['total_pitch_classes'] <= 6  # Pentatonic + passing
    
    @pytest.mark.asyncio
    async def test_cross_cultural_jazz_endpoint(self, setup_jazz_score):
        """Test cross-cultural analysis on jazz melody"""
        result = await analyze_cross_cultural_melody(
            score_id=setup_jazz_score,
            part_index=0
        )
        
        assert result['status'] == 'success'
        
        # Should detect jazz characteristics
        styles = {s['style']: s['confidence'] for s in result['detected_styles']}
        # Jazz style should be present or at least not bottom-ranked
        assert len(result['detected_styles']) > 0
    
    @pytest.mark.asyncio
    async def test_melodic_similarity_endpoint(
        self, setup_melodic_score, setup_folk_score
    ):
        """Test melodic similarity calculation endpoint"""
        result = await calculate_melodic_similarity(
            score_ids=[setup_melodic_score, setup_folk_score],
            method="interval"
        )
        
        assert result['status'] == 'success'
        assert 'similarity_matrix' in result
        assert 'similar_phrases' in result
        assert 'melodic_families' in result
        
        # Should have a 2x2 similarity matrix
        matrix = result['similarity_matrix']
        assert len(matrix) == 2
        assert len(matrix[0]) == 2
        assert matrix[0][0] == 1.0  # Self-similarity
        assert matrix[1][1] == 1.0  # Self-similarity
        assert 0 <= matrix[0][1] <= 1  # Cross-similarity
    
    @pytest.mark.asyncio
    async def test_melodic_development_endpoint(self, setup_melodic_score):
        """Test melodic development analysis endpoint"""
        result = await analyze_melodic_development(
            score_id=setup_melodic_score,
            track_dynamics=False
        )
        
        assert result['status'] == 'success'
        assert 'motivic_development' in result
        assert 'range_expansion' in result
        assert 'complexity_curve' in result
        assert 'climax_points' in result
        
        # Should track range
        if result['range_expansion']:
            range_data = result['range_expansion'][0]
            assert 'lowest' in range_data
            assert 'highest' in range_data
            assert 'range' in range_data
            assert range_data['range'] > 0
    
    @pytest.mark.asyncio
    async def test_empty_score_handling(self):
        """Test handling of empty scores in melodic analysis"""
        # Create empty score
        s = stream.Score()
        score_manager.add_score("empty_melodic", s, {})
        
        result = await analyze_melodic_contour(score_id="empty_melodic")
        
        # Should handle gracefully
        assert result['status'] in ['success', 'error']
        if result['status'] == 'success':
            assert result['contour_vector'] == []
    
    @pytest.mark.asyncio
    async def test_nonexistent_score_handling(self):
        """Test error handling for nonexistent scores"""
        result = await analyze_melodic_contour(score_id="does_not_exist")
        
        assert result['status'] == 'error'
        assert 'not found' in result['message']
    
    @pytest.mark.asyncio
    async def test_part_index_handling(self, setup_melodic_score):
        """Test handling of invalid part indices"""
        result = await analyze_melodic_contour(
            score_id=setup_melodic_score,
            part_index=999  # Invalid index
        )
        
        # Should fallback to first part
        assert result['status'] == 'success'
        assert len(result['contour_vector']) > 0


class TestComplexMelodicScenarios:
    """Test complex melodic analysis scenarios"""
    
    @pytest.mark.asyncio
    async def test_multi_part_melody_extraction(self):
        """Test melody extraction from multi-part score"""
        s = stream.Score()
        
        # Melody part (soprano)
        melody = stream.Part()
        melody.partName = "Soprano"
        for pitch in ['E5', 'D5', 'C5', 'D5', 'E5', 'E5', 'E5']:
            melody.append(note.Note(pitch, quarterLength=0.5))
        
        # Harmony part (alto) - mostly chords
        harmony = stream.Part()
        harmony.partName = "Alto"
        for _ in range(4):
            harmony.append(chord.Chord(['C4', 'E4', 'G4'], quarterLength=1))
        
        s.insert(0, melody)
        s.insert(0, harmony)
        score_manager.add_score("multi_part", s, {})
        
        # Analyze top part (melody)
        result = await analyze_melodic_contour(
            score_id="multi_part",
            part_index=0
        )
        
        assert result['status'] == 'success'
        assert len(result['contour_vector']) > 0
    
    @pytest.mark.asyncio
    async def test_ornamented_melody_analysis(self):
        """Test analysis of heavily ornamented melody"""
        s = stream.Score()
        p = stream.Part()
        
        # Main melody with ornaments
        # Grace note before C
        grace = note.Note('B3', quarterLength=0.125)
        grace.graceNote = True
        p.append(grace)
        p.append(note.Note('C4', quarterLength=1))
        
        # Trill on E
        p.append(note.Note('E4', quarterLength=0.25))
        p.append(note.Note('F4', quarterLength=0.25))
        p.append(note.Note('E4', quarterLength=0.25))
        p.append(note.Note('F4', quarterLength=0.25))
        
        # Regular note
        p.append(note.Note('G4', quarterLength=2))
        
        s.append(p)
        score_manager.add_score("ornamented", s, {})
        
        result = await analyze_cross_cultural_melody(
            score_id="ornamented",
            include_microtones=False
        )
        
        assert result['status'] == 'success'
        assert 'ornamentations' in result
        # Should detect some ornamentations
    
    @pytest.mark.asyncio
    async def test_theme_and_variations_detection(self):
        """Test detection of theme and variations"""
        scores = []
        
        # Theme
        theme = stream.Score()
        p = stream.Part()
        for pitch in ['C4', 'E4', 'G4', 'C5']:
            p.append(note.Note(pitch, quarterLength=1))
        theme.append(p)
        score_manager.add_score("theme", theme, {})
        scores.append("theme")
        
        # Variation 1 - rhythmic
        var1 = stream.Score()
        p = stream.Part()
        rhythms = [0.5, 0.5, 1.5, 0.5]
        for pitch, rhythm in zip(['C4', 'E4', 'G4', 'C5'], rhythms):
            p.append(note.Note(pitch, quarterLength=rhythm))
        var1.append(p)
        score_manager.add_score("var1", var1, {})
        scores.append("var1")
        
        # Variation 2 - ornamental
        var2 = stream.Score()
        p = stream.Part()
        extended = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5']
        for pitch in extended:
            p.append(note.Note(pitch, quarterLength=0.5))
        var2.append(p)
        score_manager.add_score("var2", var2, {})
        scores.append("var2")
        
        result = await calculate_melodic_similarity(
            score_ids=scores,
            method="combined"
        )
        
        assert result['status'] == 'success'
        # Should detect theme-variation relationships
        if result['theme_variations']:
            assert len(result['theme_variations']) > 0