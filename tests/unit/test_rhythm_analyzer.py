"""
Unit tests for rhythm analysis functionality
"""
import pytest
from music21 import stream, note, chord, meter, tempo, tie

from src.music21_mcp.core.rhythm_analyzer import (
    RhythmAnalyzer, RhythmicComplexity, TempoAnalysis,
    MeterAnalysis, RhythmicPattern
)


class TestRhythmAnalyzer:
    """Test the rhythm analyzer"""
    
    @pytest.fixture
    def analyzer(self):
        return RhythmAnalyzer()
    
    @pytest.mark.asyncio
    async def test_basic_rhythm_analysis(self, analyzer, simple_score):
        """Test basic rhythm analysis"""
        result = await analyzer.analyze_rhythm(simple_score)
        
        assert result.tempo is not None
        assert result.meter is not None
        assert result.complexity in [c for c in RhythmicComplexity]
        assert 0 <= result.syncopation_level <= 1
    
    @pytest.mark.asyncio
    async def test_tempo_detection(self, analyzer, simple_score):
        """Test tempo analysis"""
        # Add tempo marking
        simple_score.insert(0, tempo.MetronomeMark(number=120))
        
        tempo_result = await analyzer._analyze_tempo(simple_score)
        
        assert isinstance(tempo_result, TempoAnalysis)
        assert tempo_result.primary_tempo == 120
        assert tempo_result.tempo_character == "Allegro"
        assert tempo_result.tempo_stability > 0.9  # Should be stable
    
    @pytest.mark.asyncio
    async def test_tempo_changes(self, analyzer):
        """Test tempo change detection"""
        s = stream.Score()
        p = stream.Part()
        
        # Add multiple tempo markings
        p.insert(0, tempo.MetronomeMark(number=120))
        p.insert(4, tempo.MetronomeMark(number=80))
        p.insert(8, tempo.MetronomeMark(number=140))
        
        for i in range(12):
            p.append(note.Note('C4', quarterLength=1))
        
        s.append(p)
        
        tempo_result = await analyzer._analyze_tempo(s)
        
        assert len(tempo_result.tempo_changes) >= 2
        assert tempo_result.tempo_stability < 0.9  # Less stable due to changes
    
    @pytest.mark.asyncio
    async def test_meter_analysis(self, analyzer, simple_score):
        """Test meter analysis"""
        meter_result = await analyzer._analyze_meter(simple_score)
        
        assert isinstance(meter_result, MeterAnalysis)
        assert str(meter_result.primary_meter) == '4/4'
        assert meter_result.is_compound == False
        assert meter_result.is_asymmetric == False
        assert meter_result.meter_stability > 0.9
    
    @pytest.mark.asyncio
    async def test_compound_meter(self, analyzer):
        """Test compound meter detection"""
        s = stream.Score()
        p = stream.Part()
        p.append(meter.TimeSignature('6/8'))
        
        for i in range(6):
            p.append(note.Note('C4', quarterLength=0.5))
        
        s.append(p)
        
        meter_result = await analyzer._analyze_meter(s)
        
        assert meter_result.is_compound == True
        assert meter_result.metric_complexity == 'compound'
    
    @pytest.mark.asyncio
    async def test_asymmetric_meter(self, analyzer):
        """Test asymmetric meter detection"""
        s = stream.Score()
        p = stream.Part()
        p.append(meter.TimeSignature('7/8'))
        
        for i in range(7):
            p.append(note.Note('C4', quarterLength=0.5))
        
        s.append(p)
        
        meter_result = await analyzer._analyze_meter(s)
        
        assert meter_result.is_asymmetric == True
        assert meter_result.metric_complexity == 'asymmetric'
    
    @pytest.mark.asyncio
    async def test_pattern_extraction(self, analyzer):
        """Test rhythmic pattern extraction"""
        s = stream.Score()
        p = stream.Part()
        
        # Create repeating pattern
        pattern = [1, 0.5, 0.5, 1]
        
        for _ in range(4):  # Repeat 4 times
            for dur in pattern:
                p.append(note.Note('C4', quarterLength=dur))
        
        s.append(p)
        
        patterns = await analyzer._extract_rhythmic_patterns(s, min_length=4, min_occurrences=3)
        
        assert len(patterns) > 0
        assert patterns[0].occurrences >= 3
        assert patterns[0].pattern == pattern
    
    @pytest.mark.asyncio
    async def test_syncopation_detection(self, analyzer):
        """Test syncopation level calculation"""
        s = stream.Score()
        p = stream.Part()
        p.append(meter.TimeSignature('4/4'))
        
        # Create syncopated rhythm (off-beat emphasis)
        # Quarter rest, then notes on off-beats
        p.append(note.Rest(quarterLength=1))
        p.append(note.Note('C4', quarterLength=2))  # Across beat 2-3
        p.append(note.Note('D4', quarterLength=1))
        
        s.append(p)
        
        result = await analyzer.analyze_rhythm(s)
        
        assert result.syncopation_level > 0  # Should detect some syncopation
    
    @pytest.mark.asyncio
    async def test_groove_analysis(self, analyzer, simple_score):
        """Test groove analysis"""
        groove = await analyzer._analyze_groove(simple_score)
        
        assert 'swing_ratio' in groove
        assert 'groove_consistency' in groove
        assert 'groove_type' in groove
        assert groove['groove_type'] in ['straight', 'swing', 'shuffle']
    
    @pytest.mark.asyncio
    async def test_polyrhythm_detection(self, analyzer):
        """Test polyrhythm detection"""
        s = stream.Score()
        
        # Part 1: 3 against
        p1 = stream.Part()
        for _ in range(4):
            for _ in range(3):
                p1.append(note.Note('C4', quarterLength=1))
        
        # Part 2: 4 against
        p2 = stream.Part()
        for _ in range(3):
            for _ in range(4):
                p2.append(note.Note('E4', quarterLength=0.75))
        
        s.append(p1)
        s.append(p2)
        
        polyrhythms = await analyzer._detect_polyrhythms(s)
        
        assert len(polyrhythms) > 0
        # Should detect 3:4 or similar polyrhythm
    
    @pytest.mark.asyncio
    async def test_rhythmic_complexity(self, analyzer):
        """Test rhythmic complexity calculation"""
        # Simple rhythm
        simple = stream.Score()
        p = stream.Part()
        for _ in range(8):
            p.append(note.Note('C4', quarterLength=1))
        simple.append(p)
        
        simple_result = await analyzer.analyze_rhythm(simple)
        assert simple_result.complexity == RhythmicComplexity.SIMPLE
        
        # Complex rhythm
        complex_score = stream.Score()
        p = stream.Part()
        
        # Varied durations including tuplets
        durations = [1, 0.5, 0.25, 1.5, 0.75, 0.333, 0.667, 2]
        for dur in durations:
            n = note.Note('C4', quarterLength=dur)
            if dur in [0.333, 0.667]:  # Triplet values
                n.duration.appendTuplet(3, 2)
            p.append(n)
        
        complex_score.append(p)
        
        complex_result = await analyzer.analyze_rhythm(complex_score)
        assert complex_result.complexity in [RhythmicComplexity.COMPLEX, RhythmicComplexity.VERY_COMPLEX]
    
    @pytest.mark.asyncio
    async def test_tempo_stability(self, analyzer):
        """Test tempo stability calculation"""
        s = stream.Score()
        p = stream.Part()
        
        # Stable tempo
        p.insert(0, tempo.MetronomeMark(number=120))
        
        # Regular note pattern
        for _ in range(16):
            p.append(note.Note('C4', quarterLength=1))
        
        s.append(p)
        
        stability = await analyzer.analyze_tempo_stability(s)
        
        assert stability['stability'] > 0.8
        assert stability['trend'] == 'stable'
    
    @pytest.mark.asyncio
    async def test_beat_strength_profile(self, analyzer):
        """Test beat strength analysis"""
        s = stream.Score()
        p = stream.Part()
        
        # 4/4 time
        ts = meter.TimeSignature('4/4')
        p.append(ts)
        
        # Add notes on different beats
        for measure in range(2):
            for beat in range(4):
                n = note.Note('C4', quarterLength=1)
                p.append(n)
        
        s.append(p)
        
        meter_analysis = MeterAnalysis(
            primary_meter=ts,
            meter_changes=[],
            metric_complexity='simple',
            beat_hierarchy={},
            is_compound=False,
            is_asymmetric=False,
            is_mixed_meter=False,
            meter_stability=1.0,
            hypermeter=None
        )
        
        beat_profile = await analyzer._analyze_beat_strength(s, meter_analysis)
        
        assert len(beat_profile) > 0
        assert beat_profile[0]['beats'][0]['strength'] == 'strong'  # Downbeat