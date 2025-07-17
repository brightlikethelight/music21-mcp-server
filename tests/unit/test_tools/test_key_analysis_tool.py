"""
Comprehensive unit tests for KeyAnalysisTool
Tests all algorithms, consensus finding, confidence scoring, and edge cases
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from music21_mcp.tools.key_analysis_tool import KeyAnalysisTool
from music21 import stream, note, chord, key, tempo, meter

class TestKeyAnalysisTool:
    """Test suite for KeyAnalysisTool"""
    
    @pytest.fixture
    def tool(self):
        """Create tool instance with mock storage"""
        return KeyAnalysisTool({})
    
    @pytest.fixture
    def c_major_scale(self):
        """Create simple C major scale"""
        s = stream.Stream()
        for pitch in ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5']:
            s.append(note.Note(pitch, quarterLength=1))
        return s
    
    @pytest.fixture
    def a_minor_scale(self):
        """Create natural A minor scale"""
        s = stream.Stream()
        for pitch in ['A3', 'B3', 'C4', 'D4', 'E4', 'F4', 'G4', 'A4']:
            s.append(note.Note(pitch, quarterLength=1))
        return s
    
    @pytest.fixture
    def chromatic_passage(self):
        """Create chromatic passage for edge case testing"""
        s = stream.Stream()
        # Chromatic scale from C to C
        for i in range(13):
            n = note.Note()
            n.pitch.midi = 60 + i  # C4 to C5 chromatic
            s.append(n)
        return s
    
    @pytest.fixture
    def multi_part_score(self):
        """Create multi-part score"""
        score = stream.Score()
        
        # Soprano part - C major melody
        soprano = stream.Part()
        soprano.append(note.Note('E5', quarterLength=1))
        soprano.append(note.Note('D5', quarterLength=1)) 
        soprano.append(note.Note('C5', quarterLength=2))
        
        # Bass part - I-V-I progression
        bass = stream.Part()
        bass.append(note.Note('C3', quarterLength=1))
        bass.append(note.Note('G2', quarterLength=1))
        bass.append(note.Note('C3', quarterLength=2))
        
        score.insert(0, soprano)
        score.insert(0, bass)
        return score
    
    @pytest.mark.asyncio
    async def test_basic_key_detection_c_major(self, tool, c_major_scale):
        """Test basic C major detection"""
        tool.scores = {"test_score": c_major_scale}
        
        result = await tool.execute(
            score_id="test_score",
            algorithm="krumhansl"
        )
        
        assert result["status"] == "success"
        assert result["key"] == "C major"
        assert result["confidence"] > 0.8
        assert result["algorithm"] == "krumhansl"
    
    @pytest.mark.asyncio
    async def test_basic_key_detection_a_minor(self, tool, a_minor_scale):
        """Test basic A minor detection"""
        tool.scores = {"test_score": a_minor_scale}
        
        result = await tool.execute(
            score_id="test_score",
            algorithm="aarden"
        )
        
        assert result["status"] == "success"
        assert result["key"] == "a minor"
        assert result["confidence"] > 0.7
        assert result["algorithm"] == "aarden"
    
    @pytest.mark.asyncio
    async def test_all_algorithms(self, tool, c_major_scale):
        """Test all four algorithms"""
        tool.scores = {"test_score": c_major_scale}
        algorithms = ["krumhansl", "aarden", "temperley", "bellman"]
        
        for algo in algorithms:
            result = await tool.execute(
                score_id="test_score",
                algorithm=algo
            )
            
            assert result["status"] == "success"
            assert result["algorithm"] == algo
            assert "confidence" in result
            assert result["key"] in ["C major", "a minor"]  # Related keys
    
    @pytest.mark.asyncio
    async def test_consensus_algorithm(self, tool, c_major_scale):
        """Test consensus algorithm using all methods"""
        tool.scores = {"test_score": c_major_scale}
        
        result = await tool.execute(
            score_id="test_score",
            algorithm="consensus"
        )
        
        assert result["status"] == "success"
        assert result["algorithm"] == "consensus"
        assert "all_results" in result
        assert len(result["all_results"]) == 4  # All 4 algorithms
        assert result["confidence"] > 0.5  # Should have reasonable confidence
    
    @pytest.mark.asyncio
    async def test_chromatic_music_low_confidence(self, tool, chromatic_passage):
        """Test that chromatic music returns low confidence"""
        tool.scores = {"test_score": chromatic_passage}
        
        result = await tool.execute(
            score_id="test_score",
            algorithm="krumhansl"
        )
        
        assert result["status"] == "success"
        assert result["confidence"] < 0.5  # Low confidence expected
        assert "correlation_coefficient" in result
    
    @pytest.mark.asyncio
    async def test_multi_part_analysis(self, tool, multi_part_score):
        """Test analysis of multi-part score"""
        tool.scores = {"test_score": multi_part_score}
        
        result = await tool.execute(
            score_id="test_score",
            algorithm="temperley"
        )
        
        assert result["status"] == "success"
        assert result["key"] in ["C major", "G major"]  # Closely related
        assert "stream_length" in result
        assert result["stream_length"] > 0
    
    @pytest.mark.asyncio
    async def test_progress_callback(self, tool, c_major_scale):
        """Test progress reporting"""
        tool.scores = {"test_score": c_major_scale}
        progress_calls = []
        
        async def progress_callback(msg):
            progress_calls.append(msg)
        
        result = await tool.execute(
            score_id="test_score",
            algorithm="consensus",
            progress_callback=progress_callback
        )
        
        assert result["status"] == "success"
        assert len(progress_calls) > 0
        assert any("Analyzing" in call for call in progress_calls)
    
    @pytest.mark.asyncio
    async def test_missing_score_error(self, tool):
        """Test error handling for missing score"""
        result = await tool.execute(
            score_id="nonexistent",
            algorithm="krumhansl"
        )
        
        assert result["status"] == "error"
        assert "not found" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_invalid_algorithm_error(self, tool, c_major_scale):
        """Test error handling for invalid algorithm"""
        tool.scores = {"test_score": c_major_scale}
        
        result = await tool.execute(
            score_id="test_score",
            algorithm="invalid_algo"
        )
        
        assert result["status"] == "error"
        assert "algorithm" in result["error"].lower()
    
    @pytest.mark.asyncio
    async def test_empty_score(self, tool):
        """Test handling of empty score"""
        empty_score = stream.Stream()
        tool.scores = {"test_score": empty_score}
        
        result = await tool.execute(
            score_id="test_score",
            algorithm="krumhansl"
        )
        
        assert result["status"] == "success"
        # Should still return a result, even if confidence is low
        assert "key" in result
        assert "confidence" in result
    
    @pytest.mark.asyncio
    async def test_single_note_score(self, tool):
        """Test score with single note"""
        single_note = stream.Stream()
        single_note.append(note.Note('C4'))
        tool.scores = {"test_score": single_note}
        
        result = await tool.execute(
            score_id="test_score",
            algorithm="aarden"
        )
        
        assert result["status"] == "success"
        assert "key" in result
        assert result["stream_length"] == 1
    
    @pytest.mark.asyncio
    async def test_bellman_algorithm_specifics(self, tool, c_major_scale):
        """Test Bellman algorithm specific features"""
        tool.scores = {"test_score": c_major_scale}
        
        result = await tool.execute(
            score_id="test_score",
            algorithm="bellman"
        )
        
        assert result["status"] == "success"
        assert result["algorithm"] == "bellman"
        assert "key" in result
        # Bellman uses different approach, might give different confidence
        assert isinstance(result["confidence"], (int, float))
    
    @pytest.mark.asyncio
    async def test_relative_key_detection(self, tool):
        """Test detection of relative major/minor keys"""
        # Create ambiguous passage that could be C major or A minor
        ambiguous = stream.Stream()
        # Use notes common to both C major and A minor
        for pitch in ['C4', 'E4', 'G4', 'A4', 'C5', 'E5']:
            ambiguous.append(note.Note(pitch, quarterLength=0.5))
        
        tool.scores = {"test_score": ambiguous}
        
        result = await tool.execute(
            score_id="test_score",
            algorithm="consensus"
        )
        
        assert result["status"] == "success"
        # Should detect either C major or A minor
        assert result["key"] in ["C major", "a minor"]
    
    @pytest.mark.asyncio
    async def test_modulation_detection(self, tool):
        """Test score with key modulation"""
        modulating = stream.Stream()
        
        # Start in C major
        for pitch in ['C4', 'E4', 'G4', 'C5']:
            modulating.append(note.Note(pitch, quarterLength=0.5))
        
        # Modulate to G major  
        for pitch in ['G4', 'B4', 'D5', 'F#5', 'G5']:
            modulating.append(note.Note(pitch, quarterLength=0.5))
        
        tool.scores = {"test_score": modulating}
        
        result = await tool.execute(
            score_id="test_score",
            algorithm="temperley"
        )
        
        assert result["status"] == "success"
        # Might detect G major or show uncertainty
        assert "key" in result
        assert result["confidence"] < 0.9  # Should show some uncertainty
    
    @pytest.mark.asyncio
    async def test_atonal_music(self, tool):
        """Test with atonal/twelve-tone music"""
        # Create twelve-tone row
        twelve_tone = stream.Stream()
        tone_row = [0, 11, 3, 7, 10, 2, 5, 9, 1, 6, 8, 4]  # Example row
        
        for pc in tone_row:
            n = note.Note()
            n.pitch.midi = 60 + pc
            twelve_tone.append(n)
        
        tool.scores = {"test_score": twelve_tone}
        
        result = await tool.execute(
            score_id="test_score",
            algorithm="krumhansl"
        )
        
        assert result["status"] == "success"
        assert result["confidence"] < 0.4  # Very low confidence expected
    
    @pytest.mark.asyncio
    async def test_analysis_with_chords(self, tool):
        """Test analysis with chords instead of just notes"""
        chord_progression = stream.Stream()
        
        # I - IV - V - I in C major
        chord_progression.append(chord.Chord(['C4', 'E4', 'G4'], quarterLength=1))
        chord_progression.append(chord.Chord(['F4', 'A4', 'C5'], quarterLength=1))
        chord_progression.append(chord.Chord(['G4', 'B4', 'D5'], quarterLength=1))
        chord_progression.append(chord.Chord(['C4', 'E4', 'G4'], quarterLength=1))
        
        tool.scores = {"test_score": chord_progression}
        
        result = await tool.execute(
            score_id="test_score",
            algorithm="consensus"
        )
        
        assert result["status"] == "success"
        assert result["key"] == "C major"
        assert result["confidence"] > 0.8  # Strong tonal center
    
    @pytest.mark.asyncio
    async def test_minor_key_variations(self, tool):
        """Test different minor scale types"""
        # Harmonic minor (raised 7th)
        harmonic_minor = stream.Stream()
        for pitch in ['A3', 'B3', 'C4', 'D4', 'E4', 'F4', 'G#4', 'A4']:
            harmonic_minor.append(note.Note(pitch, quarterLength=0.5))
        
        tool.scores = {"test_score": harmonic_minor}
        
        result = await tool.execute(
            score_id="test_score",
            algorithm="temperley"
        )
        
        assert result["status"] == "success"
        assert "minor" in result["key"].lower()
        assert result["confidence"] > 0.6
    
    @pytest.mark.asyncio
    async def test_algorithm_comparison(self, tool, c_major_scale):
        """Compare results across algorithms"""
        tool.scores = {"test_score": c_major_scale}
        
        results = {}
        for algo in ["krumhansl", "aarden", "temperley", "bellman"]:
            result = await tool.execute(
                score_id="test_score",
                algorithm=algo
            )
            results[algo] = result
        
        # All should succeed
        assert all(r["status"] == "success" for r in results.values())
        
        # Most should agree on key
        keys = [r["key"] for r in results.values()]
        most_common_key = max(set(keys), key=keys.count)
        agreement_count = keys.count(most_common_key)
        assert agreement_count >= 3  # At least 3 out of 4 should agree
    
    @pytest.mark.asyncio
    async def test_stream_with_key_signature(self, tool):
        """Test score that already has a key signature"""
        score_with_key = stream.Stream()
        score_with_key.append(key.Key('D'))  # D major key signature
        
        # Add notes in D major
        for pitch in ['D4', 'E4', 'F#4', 'G4', 'A4', 'B4', 'C#5', 'D5']:
            score_with_key.append(note.Note(pitch, quarterLength=0.5))
        
        tool.scores = {"test_score": score_with_key}
        
        result = await tool.execute(
            score_id="test_score",
            algorithm="krumhansl"
        )
        
        assert result["status"] == "success"
        assert result["key"] == "D major"
        assert result["confidence"] > 0.9  # Very high confidence
    
    @pytest.mark.asyncio
    async def test_performance_with_large_score(self, tool):
        """Test performance with larger score"""
        large_score = stream.Stream()
        
        # Create 1000 notes in C major
        scale_degrees = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        octaves = [3, 4, 5]
        
        for i in range(1000):
            pitch = scale_degrees[i % 7] + str(octaves[i % 3])
            large_score.append(note.Note(pitch, quarterLength=0.25))
        
        tool.scores = {"test_score": large_score}
        
        import time
        start_time = time.time()
        
        result = await tool.execute(
            score_id="test_score",
            algorithm="temperley"  # Efficient algorithm
        )
        
        execution_time = time.time() - start_time
        
        assert result["status"] == "success"
        assert result["key"] == "C major"
        assert execution_time < 5.0  # Should complete within 5 seconds
        assert result["stream_length"] == 1000
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, tool):
        """Test error recovery with invalid score object"""
        tool.scores = {"test_score": "invalid_score_object"}
        
        result = await tool.execute(
            score_id="test_score",
            algorithm="krumhansl"
        )
        
        # Should handle gracefully even with invalid score
        assert result["status"] in ["success", "error"]
        if result["status"] == "error":
            assert "error" in result