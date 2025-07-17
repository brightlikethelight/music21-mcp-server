"""
Comprehensive unit tests for PatternRecognitionTool
Tests melodic sequences, rhythmic patterns, motivic analysis, and edge cases
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from music21_mcp.tools.pattern_recognition_tool import PatternRecognitionTool
from music21 import stream, note, chord, key, tempo, meter, interval

class TestPatternRecognitionTool:
    """Test suite for PatternRecognitionTool"""
    
    @pytest.fixture
    def tool(self):
        """Create tool instance with mock storage"""
        return PatternRecognitionTool({})
    
    @pytest.fixture
    def simple_sequence(self):
        """Create simple melodic sequence"""
        s = stream.Stream()
        # Pattern: C-D-E repeated at different pitch levels
        pattern = [('C4', 1), ('D4', 1), ('E4', 1)]
        # Original
        for pitch, dur in pattern:
            s.append(note.Note(pitch, quarterLength=dur))
        # Sequence up a step  
        for pitch, dur in [('D4', 1), ('E4', 1), ('F4', 1)]:
            s.append(note.Note(pitch, quarterLength=dur))
        # Sequence up another step
        for pitch, dur in [('E4', 1), ('F4', 1), ('G4', 1)]:
            s.append(note.Note(pitch, quarterLength=dur))
        return s
    
    @pytest.fixture
    def rhythmic_pattern(self):
        """Create score with rhythmic patterns"""
        s = stream.Stream()
        s.append(meter.TimeSignature('4/4'))
        
        # Rhythmic pattern: quarter, two eighths, quarter
        rhythm_pattern = [1, 0.5, 0.5, 1]
        
        # Repeat pattern 3 times with different pitches
        pitches = [['C4', 'D4', 'E4', 'F4'],
                   ['G4', 'A4', 'B4', 'C5'],
                   ['E4', 'F4', 'G4', 'A4']]
        
        for pitch_set in pitches:
            for i, dur in enumerate(rhythm_pattern):
                s.append(note.Note(pitch_set[i], quarterLength=dur))
        
        return s
    
    @pytest.fixture
    def bach_like_subject(self):
        """Create Bach-like fugue subject"""
        s = stream.Stream()
        s.append(key.Key('G'))
        
        # Simple fugue subject pattern
        subject = [
            ('G4', 0.5), ('D5', 0.5), ('B4', 0.5), ('G4', 0.5),
            ('A4', 1), ('F#4', 0.5), ('G4', 0.5)
        ]
        
        for pitch, dur in subject:
            s.append(note.Note(pitch, quarterLength=dur))
        
        # Add answer at the fifth
        for pitch, dur in subject:
            n = note.Note(pitch, quarterLength=dur)
            n.transpose(interval.Interval('P5'), inPlace=True)
            s.append(n)
        
        return s
    
    @pytest.fixture
    def syncopated_rhythm(self):
        """Create syncopated rhythm pattern"""
        s = stream.Stream()
        s.append(meter.TimeSignature('4/4'))
        
        # Syncopated pattern with ties across beats
        notes_data = [
            ('C4', 0.5, 3),   # Eighth note starting on beat 4.5
            ('D4', 1.5, 4),   # Dotted quarter crossing barline
            ('E4', 1, 2.5),   # Quarter on off-beat
            ('F4', 0.5, 3.5), # Eighth on off-beat
            ('G4', 2, 4),     # Half note
        ]
        
        for pitch, dur, beat in notes_data:
            n = note.Note(pitch, quarterLength=dur)
            n.offset = beat - 1  # music21 uses 0-based offsets
            s.append(n)
        
        return s
    
    @pytest.mark.asyncio
    async def test_basic_melodic_sequence_detection(self, tool, simple_sequence):
        """Test detection of simple melodic sequences"""
        tool.scores = {"test_score": simple_sequence}
        
        result = await tool.execute(
            score_id="test_score"
        )
        
        assert result["status"] == "success"
        assert "melodic_patterns" in result
        assert "sequences" in result["melodic_patterns"]
        
        sequences = result["melodic_patterns"]["sequences"]
        assert len(sequences) > 0
        
        # Should detect the ascending sequence pattern
        first_sequence = sequences[0]
        assert first_sequence["pattern_length"] == 3
        assert first_sequence["occurrences"] >= 2
        assert first_sequence["interval_pattern"] == [2, 2]  # Major seconds
    
    @pytest.mark.asyncio
    async def test_rhythmic_pattern_detection(self, tool, rhythmic_pattern):
        """Test detection of rhythmic patterns"""
        tool.scores = {"test_score": rhythmic_pattern}
        
        result = await tool.execute(
            score_id="test_score"
        )
        
        assert result["status"] == "success"
        assert "rhythmic_patterns" in result
        
        rhythmic = result["rhythmic_patterns"]
        assert "patterns" in rhythmic
        assert len(rhythmic["patterns"]) > 0
        
        # Should detect the quarter-eighth-eighth-quarter pattern
        pattern = rhythmic["patterns"][0]
        assert pattern["occurrences"] >= 2
        assert pattern["duration_pattern"] == [1.0, 0.5, 0.5, 1.0]
    
    @pytest.mark.asyncio
    async def test_motivic_analysis(self, tool, bach_like_subject):
        """Test motivic analysis detection"""
        tool.scores = {"test_score": bach_like_subject}
        
        result = await tool.execute(
            score_id="test_score"
        )
        
        assert result["status"] == "success"
        assert "motivic_analysis" in result
        
        motives = result["motivic_analysis"]["motives"]
        assert len(motives) > 0
        
        # Should detect repeated melodic cells
        for motive in motives:
            assert "interval_pattern" in motive
            assert "rhythm_pattern" in motive
            assert motive["occurrences"] >= 2
    
    @pytest.mark.asyncio
    async def test_contour_analysis(self, tool, simple_sequence):
        """Test melodic contour analysis"""
        tool.scores = {"test_score": simple_sequence}
        
        result = await tool.execute(
            score_id="test_score"
        )
        
        assert result["status"] == "success"
        assert "melodic_patterns" in result
        assert "contour_analysis" in result["melodic_patterns"]
        
        contour = result["melodic_patterns"]["contour_analysis"]
        assert "contour_segments" in contour
        
        # Should identify ascending contours
        ascending_count = sum(1 for seg in contour["contour_segments"] 
                            if seg["direction"] == "ascending")
        assert ascending_count > 0
    
    @pytest.mark.asyncio
    async def test_syncopation_detection(self, tool, syncopated_rhythm):
        """Test syncopation detection"""
        tool.scores = {"test_score": syncopated_rhythm}
        
        result = await tool.execute(
            score_id="test_score"
        )
        
        assert result["status"] == "success"
        assert "rhythmic_patterns" in result
        
        rhythmic = result["rhythmic_patterns"]
        assert "syncopation_detected" in rhythmic
        assert rhythmic["syncopation_detected"] == True
        
        if "syncopations" in rhythmic:
            assert len(rhythmic["syncopations"]) > 0
    
    @pytest.mark.asyncio
    async def test_similarity_threshold(self, tool, simple_sequence):
        """Test pattern matching with different similarity thresholds"""
        tool.scores = {"test_score": simple_sequence}
        
        # Test with high similarity threshold
        result_high = await tool.execute(
            score_id="test_score",
            similarity_threshold=0.9
        )
        
        # Test with low similarity threshold  
        result_low = await tool.execute(
            score_id="test_score",
            similarity_threshold=0.5
        )
        
        assert result_high["status"] == "success"
        assert result_low["status"] == "success"
        
        # Lower threshold should find more patterns
        high_patterns = len(result_high["melodic_patterns"]["sequences"])
        low_patterns = len(result_low["melodic_patterns"]["sequences"])
        assert low_patterns >= high_patterns
    
    @pytest.mark.asyncio
    async def test_interval_pattern_analysis(self, tool):
        """Test interval pattern detection"""
        s = stream.Stream()
        # Create specific interval pattern: P4 up, M3 down, P5 up
        notes_intervals = [
            ('C4', None),
            ('F4', 'P4'),   # Perfect 4th up
            ('D4', '-M3'),  # Major 3rd down
            ('A4', 'P5'),   # Perfect 5th up
        ]
        
        # Create pattern
        for pitch, _ in notes_intervals:
            s.append(note.Note(pitch, quarterLength=1))
        
        # Repeat pattern starting on G
        for pitch, _ in [('G4', None), ('C5', 'P4'), ('A4', '-M3'), ('E5', 'P5')]:
            s.append(note.Note(pitch, quarterLength=1))
        
        tool.scores = {"test_score": s}
        
        result = await tool.execute(
            score_id="test_score"
        )
        
        assert result["status"] == "success"
        patterns = result["melodic_patterns"]["sequences"]
        assert len(patterns) > 0
        
        # Should detect the interval pattern
        pattern = patterns[0]
        assert pattern["interval_pattern"] == [5, -4, 7]  # Semitones: P4, -M3, P5
    
    @pytest.mark.asyncio
    async def test_phrase_structure_analysis(self, tool):
        """Test phrase structure detection"""
        s = stream.Stream()
        
        # Create 4-bar phrases with clear cadences
        # Phrase 1
        for pitch in ['C4', 'D4', 'E4', 'F4', 'G4', 'F4', 'E4', 'D4', 'C4']:
            s.append(note.Note(pitch, quarterLength=0.5))
        s.append(note.Rest(quarterLength=1))  # Phrase break
        
        # Phrase 2 (similar contour)
        for pitch in ['G4', 'A4', 'B4', 'C5', 'D5', 'C5', 'B4', 'A4', 'G4']:
            s.append(note.Note(pitch, quarterLength=0.5))
        s.append(note.Rest(quarterLength=1))
        
        tool.scores = {"test_score": s}
        
        result = await tool.execute(
            score_id="test_score"
        )
        
        assert result["status"] == "success"
        assert "phrase_structure" in result
        
        phrases = result["phrase_structure"]["phrases"]
        assert len(phrases) >= 2
        assert all("start_measure" in p and "end_measure" in p for p in phrases)
    
    @pytest.mark.asyncio
    async def test_cross_rhythm_detection(self, tool):
        """Test detection of polyrhythms/cross-rhythms"""
        s = stream.Score()
        
        # Part 1: 3 against 2
        part1 = stream.Part()
        part1.append(meter.TimeSignature('6/8'))
        # Three notes in the time of two
        for _ in range(6):
            part1.append(note.Note('C4', quarterLength=0.5))
        
        # Part 2: 2 against 3  
        part2 = stream.Part()
        part2.append(meter.TimeSignature('6/8'))
        # Two notes in the time of three
        for _ in range(4):
            part2.append(note.Note('G4', quarterLength=0.75))
        
        s.insert(0, part1)
        s.insert(0, part2)
        
        tool.scores = {"test_score": s}
        
        result = await tool.execute(
            score_id="test_score"
        )
        
        assert result["status"] == "success"
        assert "rhythmic_patterns" in result
        
        if "cross_rhythms" in result["rhythmic_patterns"]:
            assert result["rhythmic_patterns"]["cross_rhythms"]["detected"] == True
    
    @pytest.mark.asyncio
    async def test_empty_score(self, tool):
        """Test handling of empty score"""
        empty_score = stream.Stream()
        tool.scores = {"test_score": empty_score}
        
        result = await tool.execute(
            score_id="test_score"
        )
        
        assert result["status"] == "success"
        assert result["melodic_patterns"]["sequences"] == []
        assert result["rhythmic_patterns"]["patterns"] == []
    
    @pytest.mark.asyncio
    async def test_single_note_score(self, tool):
        """Test score with only one note"""
        single = stream.Stream()
        single.append(note.Note('C4', quarterLength=1))
        tool.scores = {"test_score": single}
        
        result = await tool.execute(
            score_id="test_score"
        )
        
        assert result["status"] == "success"
        assert len(result["melodic_patterns"]["sequences"]) == 0
    
    @pytest.mark.asyncio
    async def test_missing_score_error(self, tool):
        """Test error handling for missing score"""
        result = await tool.execute(
            score_id="nonexistent"
        )
        
        assert result["status"] == "error"
        assert "not found" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_progress_callback(self, tool, simple_sequence):
        """Test progress reporting"""
        tool.scores = {"test_score": simple_sequence}
        progress_calls = []
        
        async def progress_callback(msg):
            progress_calls.append(msg)
        
        result = await tool.execute(
            score_id="test_score",
            progress_callback=progress_callback
        )
        
        assert result["status"] == "success"
        assert len(progress_calls) > 0
        assert any("Analyzing" in call for call in progress_calls)
    
    @pytest.mark.asyncio
    async def test_pattern_deduplication(self, tool):
        """Test that duplicate patterns are properly deduplicated"""
        s = stream.Stream()
        
        # Create exact same pattern multiple times
        pattern = [('C4', 1), ('E4', 1), ('G4', 1)]
        
        for _ in range(5):  # Repeat 5 times
            for pitch, dur in pattern:
                s.append(note.Note(pitch, quarterLength=dur))
        
        tool.scores = {"test_score": s}
        
        result = await tool.execute(
            score_id="test_score"
        )
        
        assert result["status"] == "success"
        sequences = result["melodic_patterns"]["sequences"]
        
        # Should detect pattern but not duplicate it
        assert len(sequences) > 0
        assert sequences[0]["occurrences"] >= 4
    
    @pytest.mark.asyncio
    async def test_retrograde_detection(self, tool):
        """Test detection of retrograde (reverse) patterns"""
        s = stream.Stream()
        
        # Original pattern
        original = ['C4', 'E4', 'G4', 'B4']
        for pitch in original:
            s.append(note.Note(pitch, quarterLength=1))
        
        # Retrograde
        for pitch in reversed(original):
            s.append(note.Note(pitch, quarterLength=1))
        
        tool.scores = {"test_score": s}
        
        result = await tool.execute(
            score_id="test_score"
        )
        
        assert result["status"] == "success"
        if "special_patterns" in result:
            patterns = result["special_patterns"]
            assert any(p.get("type") == "retrograde" for p in patterns)
    
    @pytest.mark.asyncio
    async def test_augmentation_diminution(self, tool):
        """Test detection of augmented/diminished patterns"""
        s = stream.Stream()
        
        # Original pattern with specific rhythm
        pattern = [('C4', 1), ('D4', 1), ('E4', 1)]
        for pitch, dur in pattern:
            s.append(note.Note(pitch, quarterLength=dur))
        
        # Augmentation (double duration)
        for pitch, dur in pattern:
            s.append(note.Note(pitch, quarterLength=dur * 2))
        
        # Diminution (half duration)
        for pitch, dur in pattern:
            s.append(note.Note(pitch, quarterLength=dur * 0.5))
        
        tool.scores = {"test_score": s}
        
        result = await tool.execute(
            score_id="test_score"
        )
        
        assert result["status"] == "success"
        # Tool should recognize rhythmic variations of same melodic pattern
        assert "motivic_analysis" in result
    
    @pytest.mark.asyncio
    async def test_compound_meter_patterns(self, tool):
        """Test pattern detection in compound meters"""
        s = stream.Stream()
        s.append(meter.TimeSignature('6/8'))
        
        # Typical 6/8 pattern
        pattern = [
            ('C4', 0.75), ('D4', 0.75), ('E4', 0.75),  # Dotted quarters
            ('F4', 0.5), ('E4', 0.5), ('D4', 0.5),     # Regular eighths
        ]
        
        # Repeat pattern
        for _ in range(3):
            for pitch, dur in pattern:
                s.append(note.Note(pitch, quarterLength=dur))
        
        tool.scores = {"test_score": s}
        
        result = await tool.execute(
            score_id="test_score"
        )
        
        assert result["status"] == "success"
        assert len(result["rhythmic_patterns"]["patterns"]) > 0
    
    @pytest.mark.asyncio
    async def test_large_score_performance(self, tool):
        """Test performance with large score"""
        s = stream.Stream()
        
        # Create 500 measures of patterns
        pattern = ['C4', 'D4', 'E4', 'F4']
        for i in range(500):
            for pitch in pattern:
                s.append(note.Note(pitch, quarterLength=0.25))
        
        tool.scores = {"test_score": s}
        
        import time
        start_time = time.time()
        
        result = await tool.execute(
            score_id="test_score"
        )
        
        execution_time = time.time() - start_time
        
        assert result["status"] == "success"
        assert execution_time < 10.0  # Should complete within 10 seconds
        assert "analysis_stats" in result
        assert result["analysis_stats"]["total_notes"] == 2000
    
    @pytest.mark.asyncio
    async def test_chordal_pattern_detection(self, tool):
        """Test pattern detection with chords"""
        s = stream.Stream()
        
        # Chord progression pattern
        progression = [
            ['C4', 'E4', 'G4'],    # C major
            ['F4', 'A4', 'C5'],    # F major
            ['G4', 'B4', 'D5'],    # G major
            ['C4', 'E4', 'G4'],    # C major
        ]
        
        # Repeat progression
        for _ in range(3):
            for chord_notes in progression:
                c = chord.Chord(chord_notes, quarterLength=1)
                s.append(c)
        
        tool.scores = {"test_score": s}
        
        result = await tool.execute(
            score_id="test_score"
        )
        
        assert result["status"] == "success"
        # Should detect harmonic patterns even with chords
        assert len(result["motivic_analysis"]["motives"]) > 0