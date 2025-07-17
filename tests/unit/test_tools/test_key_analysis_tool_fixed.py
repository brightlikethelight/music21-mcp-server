"""
Fixed unit tests for KeyAnalysisTool matching actual implementation
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
    
    @pytest.mark.asyncio
    async def test_basic_key_detection_c_major(self, c_major_scale):
        """Test basic C major detection"""
        score_manager = {"test_score": c_major_scale}
        tool = KeyAnalysisTool(score_manager)
        
        result = await tool.execute(score_id="test_score", algorithm="krumhansl")
        
        assert result["status"] == "success"
        assert result["key"] == "C major"
        assert result["confidence"] > 0.7
    
    @pytest.mark.asyncio
    async def test_basic_key_detection_a_minor(self, a_minor_scale):
        """Test basic A minor detection"""
        score_manager = {"test_score": a_minor_scale}
        tool = KeyAnalysisTool(score_manager)
        
        result = await tool.execute(score_id="test_score", algorithm="aarden")
        
        assert result["status"] == "success"
        assert result["key"] == "a minor"
        assert result["confidence"] > 0.6
    
    @pytest.mark.asyncio
    async def test_all_algorithms(self, c_major_scale):
        """Test all available algorithms"""
        score_manager = {"test_score": c_major_scale}
        tool = KeyAnalysisTool(score_manager)
        
        algorithms = ["krumhansl", "aarden", "temperley", "bellman"]
        for algo in algorithms:
            result = await tool.execute(score_id="test_score", algorithm=algo)
            
            assert result["status"] == "success"
            assert result["algorithm"] == algo
            assert "confidence" in result
            assert result["key"] in ["C major", "a minor"]  # Related keys
    
    @pytest.mark.asyncio
    async def test_all_algorithms_consensus(self, c_major_scale):
        """Test consensus algorithm using all methods"""
        score_manager = {"test_score": c_major_scale}
        tool = KeyAnalysisTool(score_manager)
        
        result = await tool.execute(score_id="test_score", algorithm="all")
        
        assert result["status"] == "success"
        assert result["algorithm"] == "all"
        assert "all_results" in result
        assert len(result["all_results"]) == 4  # All 4 algorithms
    
    @pytest.mark.asyncio
    async def test_empty_score(self):
        """Test handling of empty score"""
        empty_score = stream.Stream()
        score_manager = {"test_score": empty_score}
        tool = KeyAnalysisTool(score_manager)
        
        result = await tool.execute(score_id="test_score", algorithm="krumhansl")
        
        assert result["status"] == "success"
        assert "key" in result
        assert "confidence" in result
    
    @pytest.mark.asyncio
    async def test_missing_score_error(self):
        """Test error handling for missing score"""
        tool = KeyAnalysisTool({})
        
        result = await tool.execute(score_id="nonexistent", algorithm="krumhansl")
        
        assert result["status"] == "error"
        assert "not found" in result["message"].lower()
    
    @pytest.mark.asyncio
    async def test_invalid_algorithm_error(self, c_major_scale):
        """Test error handling for invalid algorithm"""
        score_manager = {"test_score": c_major_scale}
        tool = KeyAnalysisTool(score_manager)
        
        result = await tool.execute(score_id="test_score", algorithm="invalid_algo")
        
        assert result["status"] == "error"
        assert "algorithm" in result["message"].lower()
    
    @pytest.mark.asyncio
    async def test_single_note_score(self):
        """Test score with single note"""
        single_note = stream.Stream()
        single_note.append(note.Note('C4'))
        score_manager = {"test_score": single_note}
        tool = KeyAnalysisTool(score_manager)
        
        result = await tool.execute(score_id="test_score", algorithm="aarden")
        
        assert result["status"] == "success"
        assert "key" in result
    
    @pytest.mark.asyncio
    async def test_progress_callback(self, c_major_scale):
        """Test progress reporting"""
        score_manager = {"test_score": c_major_scale}
        tool = KeyAnalysisTool(score_manager)
        
        progress_calls = []
        
        def progress_callback(percent, msg):
            progress_calls.append((percent, msg))
        
        tool.set_progress_callback(progress_callback)
        
        result = await tool.execute(score_id="test_score", algorithm="all")
        
        assert result["status"] == "success"
        assert len(progress_calls) > 0
        assert any("Analyzing" in msg for _, msg in progress_calls)
    
    @pytest.mark.asyncio
    async def test_score_with_chords(self):
        """Test analysis with chords instead of just notes"""
        chord_progression = stream.Stream()
        
        # I - IV - V - I in C major
        chord_progression.append(chord.Chord(['C4', 'E4', 'G4'], quarterLength=1))
        chord_progression.append(chord.Chord(['F4', 'A4', 'C5'], quarterLength=1))
        chord_progression.append(chord.Chord(['G4', 'B4', 'D5'], quarterLength=1))
        chord_progression.append(chord.Chord(['C4', 'E4', 'G4'], quarterLength=1))
        
        score_manager = {"test_score": chord_progression}
        tool = KeyAnalysisTool(score_manager)
        
        result = await tool.execute(score_id="test_score", algorithm="krumhansl")
        
        assert result["status"] == "success"
        assert result["key"] == "C major"
        assert result["confidence"] > 0.8
    
    @pytest.mark.asyncio
    async def test_chromatic_music_low_confidence(self):
        """Test that chromatic music returns low confidence"""
        chromatic_passage = stream.Stream()
        # Chromatic scale from C to C
        for i in range(13):
            n = note.Note()
            n.pitch.midi = 60 + i  # C4 to C5 chromatic
            chromatic_passage.append(n)
        
        score_manager = {"test_score": chromatic_passage}
        tool = KeyAnalysisTool(score_manager)
        
        result = await tool.execute(score_id="test_score", algorithm="krumhansl")
        
        assert result["status"] == "success"
        assert result["confidence"] < 0.5
    
    @pytest.mark.asyncio
    async def test_multi_part_score(self):
        """Test analysis of multi-part score"""
        score = stream.Score()
        
        # Soprano part
        soprano = stream.Part()
        soprano.append(note.Note('E5', quarterLength=1))
        soprano.append(note.Note('D5', quarterLength=1))
        soprano.append(note.Note('C5', quarterLength=2))
        
        # Bass part
        bass = stream.Part()
        bass.append(note.Note('C3', quarterLength=1))
        bass.append(note.Note('G2', quarterLength=1))
        bass.append(note.Note('C3', quarterLength=2))
        
        score.insert(0, soprano)
        score.insert(0, bass)
        
        score_manager = {"test_score": score}
        tool = KeyAnalysisTool(score_manager)
        
        result = await tool.execute(score_id="test_score", algorithm="temperley")
        
        assert result["status"] == "success"
        assert result["key"] in ["C major", "G major"]
    
    @pytest.mark.asyncio
    async def test_performance_with_large_score(self):
        """Test performance with larger score"""
        large_score = stream.Stream()
        
        # Create 100 notes in C major
        scale_degrees = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        octaves = [3, 4, 5]
        
        for i in range(100):
            pitch = scale_degrees[i % 7] + str(octaves[i % 3])
            large_score.append(note.Note(pitch, quarterLength=0.25))
        
        score_manager = {"test_score": large_score}
        tool = KeyAnalysisTool(score_manager)
        
        import time
        start_time = time.time()
        
        result = await tool.execute(score_id="test_score", algorithm="temperley")
        
        execution_time = time.time() - start_time
        
        assert result["status"] == "success"
        assert result["key"] == "C major"
        assert execution_time < 5.0  # Should complete within 5 seconds
    
    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test error recovery with invalid score object"""
        score_manager = {"test_score": "invalid_score_object"}
        tool = KeyAnalysisTool(score_manager)
        
        result = await tool.execute(score_id="test_score", algorithm="krumhansl")
        
        # Should handle gracefully
        assert result["status"] in ["success", "error"]
    
    @pytest.mark.asyncio
    async def test_stream_with_key_signature(self):
        """Test score that already has a key signature"""
        score_with_key = stream.Stream()
        score_with_key.append(key.Key('D'))  # D major key signature
        
        # Add notes in D major
        for pitch in ['D4', 'E4', 'F#4', 'G4', 'A4', 'B4', 'C#5', 'D5']:
            score_with_key.append(note.Note(pitch, quarterLength=0.5))
        
        score_manager = {"test_score": score_with_key}
        tool = KeyAnalysisTool(score_manager)
        
        result = await tool.execute(score_id="test_score", algorithm="krumhansl")
        
        assert result["status"] == "success"
        assert result["key"] == "D major"
        assert result["confidence"] > 0.8