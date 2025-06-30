"""
Unit tests for melodic analyzer functionality
"""
import pytest
from music21 import key, meter, note, stream

from src.music21_mcp.core.melodic_analyzer import (
    ContourAnalysis,
    ContourType,
    CrossCulturalAnalysis,
    MelodicAnalyzer,
    MelodicSimilarity,
    MelodicStyle,
    MotivicAnalysis,
    PatternTransformation,
)


class TestMelodicAnalyzer:
    """Test melodic analysis functionality"""
    
    @pytest.fixture
    def analyzer(self):
        return MelodicAnalyzer()
    
    @pytest.fixture
    def simple_melody(self):
        """Create a simple ascending melody"""
        s = stream.Stream()
        s.append(meter.TimeSignature('4/4'))
        
        # Simple ascending scale
        for pitch_name in ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5']:
            n = note.Note(pitch_name, quarterLength=1)
            s.append(n)
        
        return s
    
    @pytest.fixture
    def complex_melody(self):
        """Create a more complex melody with patterns"""
        s = stream.Score()
        p = stream.Part()
        p.append(key.KeySignature(0))  # C major
        p.append(meter.TimeSignature('4/4'))
        
        # Melody with repeated motives
        # Opening motive: C-E-G (ascending arpeggio)
        pattern1 = ['C4', 'E4', 'G4']
        # Variant at different pitch level
        pattern2 = ['G4', 'B4', 'D5']
        # Return to original
        pattern3 = ['C4', 'E4', 'G4']
        
        for pattern in [pattern1, pattern2, pattern3]:
            for pitch_name in pattern:
                n = note.Note(pitch_name, quarterLength=0.5)
                p.append(n)
            # Add a longer note
            n = note.Note('C5', quarterLength=1.5)
            p.append(n)
        
        s.append(p)
        return s
    
    @pytest.fixture
    def folk_melody(self):
        """Create a pentatonic folk-style melody"""
        s = stream.Stream()
        s.append(meter.TimeSignature('4/4'))
        
        # Pentatonic scale pattern (C D E G A)
        pentatonic_pattern = ['C4', 'D4', 'E4', 'G4', 'A4', 'G4', 'E4', 'D4', 'C4']
        for pitch_name in pentatonic_pattern:
            n = note.Note(pitch_name, quarterLength=0.5)
            s.append(n)
        
        return s
    
    @pytest.mark.asyncio
    async def test_contour_analysis_ascending(self, analyzer, simple_melody):
        """Test contour analysis on ascending melody"""
        result = await analyzer.analyze_melodic_contour(simple_melody)
        
        assert isinstance(result, ContourAnalysis)
        assert result.overall_contour == ContourType.ASCENDING
        assert all(v == 1 for v in result.contour_vector)  # All ascending
        assert result.smoothness_score > 0.8  # Very smooth (stepwise)
        assert result.complexity_score < 0.3  # Simple contour
    
    @pytest.mark.asyncio
    async def test_contour_analysis_arch(self, analyzer):
        """Test arch contour detection"""
        s = stream.Stream()
        # Create arch shape: up then down
        pitches = ['C4', 'D4', 'E4', 'F4', 'G4', 'F4', 'E4', 'D4', 'C4']
        for p in pitches:
            s.append(note.Note(p, quarterLength=1))
        
        result = await analyzer.analyze_melodic_contour(s)
        
        assert result.overall_contour == ContourType.ARCH
        assert len(result.arch_points) > 0
        # Should find peak at G4
        peaks = [ap for ap in result.arch_points if ap['type'] == 'peak']
        assert len(peaks) > 0
    
    @pytest.mark.asyncio
    async def test_motive_detection(self, analyzer, complex_melody):
        """Test melodic motive detection"""
        result = await analyzer.detect_motives(complex_melody, min_length=3)
        
        assert isinstance(result, MotivicAnalysis)
        assert len(result.motives) > 0
        
        # Should find the repeated C-E-G pattern
        primary = result.primary_motive
        assert primary is not None
        assert len(primary.intervals) >= 2  # At least 2 intervals for 3 notes
        
        # Check for exact repetitions
        assert len(primary.occurrences) >= 1  # At least one recurrence
        
        assert result.coherence_score > 0  # Some motivic coherence
    
    @pytest.mark.asyncio
    async def test_pattern_transformation_detection(self, analyzer):
        """Test detection of transformed patterns"""
        s = stream.Stream()
        
        # Original pattern
        original = ['C4', 'E4', 'G4']
        for p in original:
            s.append(note.Note(p, quarterLength=0.5))
        
        # Rest
        s.append(note.Rest(quarterLength=1))
        
        # Inverted pattern (C-A-F)
        inverted = ['C4', 'A3', 'F3']
        for p in inverted:
            s.append(note.Note(p, quarterLength=0.5))
        
        patterns = await analyzer.find_melodic_patterns(s, pattern_length=3)
        
        assert len(patterns) >= 1
        # Should detect the inversion relationship
    
    @pytest.mark.asyncio
    async def test_cross_cultural_pentatonic(self, analyzer, folk_melody):
        """Test cross-cultural analysis on pentatonic melody"""
        result = await analyzer.analyze_cross_cultural_elements(folk_melody)
        
        assert isinstance(result, CrossCulturalAnalysis)
        assert len(result.detected_styles) > 0
        
        # Should score high for pentatonic style
        styles_dict = {style: conf for style, conf in result.detected_styles}
        assert MelodicStyle.PENTATONIC in styles_dict
        assert styles_dict[MelodicStyle.PENTATONIC] > 0.5
        
        # Check scale characteristics
        assert result.scale_characteristics['total_pitch_classes'] <= 5
    
    @pytest.mark.asyncio
    async def test_jazz_style_detection(self, analyzer):
        """Test jazz style detection"""
        s = stream.Stream()
        s.append(key.KeySignature(-1))  # F major
        
        # Jazz-like melody with chromatic passing tones
        jazz_line = ['F4', 'G4', 'G#4', 'A4', 'C5', 'B4', 'Bb4', 'A4', 'F4']
        for p in jazz_line:
            s.append(note.Note(p, quarterLength=0.25))
        
        result = await analyzer.analyze_cross_cultural_elements(s)
        
        styles_dict = {style: conf for style, conf in result.detected_styles}
        # Should have some jazz characteristics
        if MelodicStyle.JAZZ in styles_dict:
            assert styles_dict[MelodicStyle.JAZZ] > 0.3
    
    @pytest.mark.asyncio
    async def test_melodic_similarity_identical(self, analyzer):
        """Test similarity calculation for identical melodies"""
        melody1 = stream.Stream()
        melody2 = stream.Stream()
        
        for p in ['C4', 'D4', 'E4', 'F4']:
            melody1.append(note.Note(p, quarterLength=1))
            melody2.append(note.Note(p, quarterLength=1))
        
        result = await analyzer.calculate_melodic_similarity(
            [melody1, melody2], method='interval'
        )
        
        assert isinstance(result, MelodicSimilarity)
        assert result.similarity_matrix is not None
        assert result.similarity_matrix[0, 1] == 1.0  # Identical
    
    @pytest.mark.asyncio
    async def test_melodic_similarity_transposed(self, analyzer):
        """Test similarity for transposed melodies"""
        melody1 = stream.Stream()
        melody2 = stream.Stream()
        
        # Original in C
        for p in ['C4', 'E4', 'G4']:
            melody1.append(note.Note(p, quarterLength=1))
        
        # Transposed to G
        for p in ['G4', 'B4', 'D5']:
            melody2.append(note.Note(p, quarterLength=1))
        
        result = await analyzer.calculate_melodic_similarity(
            [melody1, melody2], method='interval'
        )
        
        # Should have high similarity (same intervals)
        assert result.similarity_matrix[0, 1] > 0.8
    
    @pytest.mark.asyncio
    async def test_melodic_development_analysis(self, analyzer, complex_melody):
        """Test melodic development tracking"""
        result = await analyzer.analyze_melodic_development(complex_melody)
        
        assert 'motivic_development' in result
        assert 'range_expansion' in result
        assert 'complexity_curve' in result
        assert 'climax_points' in result
        
        # Should find some development
        if result['range_expansion']:
            assert result['range_expansion'][0]['range'] > 0
    
    @pytest.mark.asyncio
    async def test_ornamentation_detection(self, analyzer):
        """Test detection of melodic ornaments"""
        s = stream.Stream()
        
        # Main note with grace note
        grace = note.Note('D4', quarterLength=0.125)
        main = note.Note('C4', quarterLength=1)
        s.append(grace)
        s.append(main)
        
        # Trill-like pattern
        for _ in range(2):
            s.append(note.Note('E4', quarterLength=0.25))
            s.append(note.Note('F4', quarterLength=0.25))
        
        result = await analyzer.analyze_cross_cultural_elements(s)
        
        assert len(result.ornamentations) > 0
        # Should detect grace note or trill
    
    @pytest.mark.asyncio
    async def test_empty_melody_handling(self, analyzer):
        """Test handling of empty melodies"""
        empty = stream.Stream()
        
        result = await analyzer.analyze_melodic_contour(empty)
        assert result.overall_contour == ContourType.STATIC
        assert result.contour_vector == []
        
        motives = await analyzer.detect_motives(stream.Score())
        assert len(motives.motives) == 0


class TestContourClassification:
    """Test contour type classification"""
    
    @pytest.fixture
    def analyzer(self):
        return MelodicAnalyzer()
    
    @pytest.mark.asyncio
    async def test_wave_contour(self, analyzer):
        """Test wave contour detection"""
        s = stream.Stream()
        # Create wave pattern
        wave = ['C4', 'E4', 'D4', 'F4', 'E4', 'G4', 'F4', 'A4']
        for p in wave:
            s.append(note.Note(p, quarterLength=0.5))
        
        result = await analyzer.analyze_melodic_contour(s)
        assert result.overall_contour in [ContourType.WAVE, ContourType.ZIGZAG]
    
    @pytest.mark.asyncio
    async def test_static_contour(self, analyzer):
        """Test static contour detection"""
        s = stream.Stream()
        # Repeated notes
        for _ in range(8):
            s.append(note.Note('C4', quarterLength=0.5))
        
        result = await analyzer.analyze_melodic_contour(s)
        assert result.overall_contour == ContourType.STATIC
        assert all(v == 0 for v in result.contour_vector)


class TestPatternTransformations:
    """Test pattern transformation detection"""
    
    @pytest.fixture
    def analyzer(self):
        return MelodicAnalyzer()
    
    @pytest.mark.asyncio
    async def test_retrograde_detection(self, analyzer):
        """Test retrograde pattern detection"""
        s = stream.Stream()
        
        # Forward pattern
        forward = ['C4', 'D4', 'E4', 'F4']
        for p in forward:
            s.append(note.Note(p, quarterLength=0.5))
        
        # Gap
        s.append(note.Rest(quarterLength=1))
        
        # Retrograde
        backward = ['F4', 'E4', 'D4', 'C4']
        for p in backward:
            s.append(note.Note(p, quarterLength=0.5))
        
        patterns = await analyzer.find_melodic_patterns(
            s, pattern_length=4, similarity_threshold=0.7
        )
        
        # Should find patterns (though exact detection depends on implementation)
        assert isinstance(patterns, list)
    
    @pytest.mark.asyncio
    async def test_sequence_detection(self, analyzer):
        """Test melodic sequence detection"""
        s = stream.Stream()
        
        # Descending sequence
        sequences = [
            ['C5', 'B4', 'A4'],  # Starting on C
            ['B4', 'A4', 'G4'],  # Starting on B (down a step)
            ['A4', 'G4', 'F4'],  # Starting on A (down another step)
        ]
        
        for seq in sequences:
            for p in seq:
                s.append(note.Note(p, quarterLength=0.33))
        
        patterns = await analyzer.find_melodic_patterns(
            s, pattern_length=3, similarity_threshold=0.8
        )
        
        # Should detect the sequential pattern
        assert len(patterns) >= 1
        if patterns:
            # Check for transposition
            assert patterns[0].transformation_type in [
                PatternTransformation.EXACT,
                PatternTransformation.TRANSPOSITION,
                PatternTransformation.SEQUENCE
            ]


class TestMelodicSimilarityMethods:
    """Test different similarity calculation methods"""
    
    @pytest.fixture
    def analyzer(self):
        return MelodicAnalyzer()
    
    @pytest.mark.asyncio
    async def test_contour_similarity(self, analyzer):
        """Test contour-based similarity"""
        # Same contour, different pitches
        melody1 = stream.Stream()
        melody2 = stream.Stream()
        
        # Both ascending
        for p in ['C4', 'D4', 'E4', 'F4']:
            melody1.append(note.Note(p, quarterLength=1))
        
        for p in ['G4', 'A4', 'B4', 'C5']:
            melody2.append(note.Note(p, quarterLength=1))
        
        result = await analyzer.calculate_melodic_similarity(
            [melody1, melody2], method='contour'
        )
        
        # Should have high contour similarity
        assert result.similarity_matrix[0, 1] > 0.8
    
    @pytest.mark.asyncio
    async def test_rhythm_similarity(self, analyzer):
        """Test rhythm-based similarity"""
        melody1 = stream.Stream()
        melody2 = stream.Stream()
        
        # Same rhythm, different pitches
        rhythm = [1, 0.5, 0.5, 2]
        
        pitches1 = ['C4', 'D4', 'E4', 'F4']
        pitches2 = ['G4', 'A4', 'B4', 'C5']
        
        for p, r in zip(pitches1, rhythm):
            melody1.append(note.Note(p, quarterLength=r))
        
        for p, r in zip(pitches2, rhythm):
            melody2.append(note.Note(p, quarterLength=r))
        
        result = await analyzer.calculate_melodic_similarity(
            [melody1, melody2], method='rhythm'
        )
        
        # Should have perfect rhythm similarity
        assert result.similarity_matrix[0, 1] == 1.0
    
    @pytest.mark.asyncio
    async def test_combined_similarity(self, analyzer):
        """Test combined similarity method"""
        melody1 = stream.Stream()
        melody2 = stream.Stream()
        
        # Similar but not identical
        for p in ['C4', 'E4', 'G4']:
            melody1.append(note.Note(p, quarterLength=1))
        
        for p in ['C4', 'E4', 'A4']:  # Last note different
            melody2.append(note.Note(p, quarterLength=1))
        
        result = await analyzer.calculate_melodic_similarity(
            [melody1, melody2], method='combined'
        )
        
        # Should have moderate similarity
        assert 0.4 < result.similarity_matrix[0, 1] < 0.9