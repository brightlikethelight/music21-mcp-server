"""
Comprehensive test suite for modular tools - 100% coverage with edge cases
Tests all functionality including error handling, edge cases, and performance
"""
import asyncio
import os
import tempfile
from unittest.mock import Mock

import pytest
from music21 import chord, key, meter, note, stream, tempo

# Import tools
from music21_mcp.tools import (
    ChordAnalysisTool,
    DeleteScoreTool,
    ExportScoreTool,
    ImportScoreTool,
    KeyAnalysisTool,
    ListScoresTool,
    ScoreInfoTool,
)


@pytest.fixture
def score_manager():
    """Fresh score manager for each test"""
    return {}


@pytest.fixture
def empty_score():
    """Empty score for testing edge cases"""
    return stream.Score()


@pytest.fixture
def simple_score():
    """Simple test score with notes"""
    s = stream.Score()
    p = stream.Part()
    for pitch in ['C4', 'D4', 'E4', 'F4', 'G4']:
        p.append(note.Note(pitch, quarterLength=1))
    s.append(p)
    return s


@pytest.fixture 
def complex_score():
    """Complex score with multiple parts, tempo, time signature"""
    s = stream.Score()
    s.metadata.title = "Test Piece"
    s.metadata.composer = "Test Composer"
    
    # Soprano part
    soprano = stream.Part()
    soprano.partName = "Soprano"
    soprano.append(meter.TimeSignature('4/4'))
    soprano.append(tempo.MetronomeMark(number=120))
    soprano.append(key.KeySignature(0))  # C major
    
    for pitch in ['C5', 'E5', 'G5', 'C6']:
        soprano.append(note.Note(pitch, quarterLength=1))
    
    # Bass part
    bass = stream.Part()
    bass.partName = "Bass"
    bass.append(meter.TimeSignature('4/4'))
    
    for pitch in ['C3', 'G2', 'C3', 'C2']:
        bass.append(note.Note(pitch, quarterLength=1))
    
    s.insert(0, soprano)
    s.insert(0, bass)
    
    return s


class TestImportScoreTool:
    """Comprehensive tests for ImportScoreTool"""
    
    @pytest.mark.asyncio
    async def test_import_corpus_success(self, score_manager):
        """Test successful corpus import"""
        tool = ImportScoreTool(score_manager)
        result = await tool.execute("test_bach", "bach/bwv66.6")
        
        assert result["status"] == "success"
        assert result["score_id"] == "test_bach"
        assert result["source_type"] == "corpus"
        assert result["num_notes"] > 0
        assert result["num_measures"] > 0
        assert "test_bach" in score_manager
    
    @pytest.mark.asyncio
    async def test_import_text_notation(self, score_manager):
        """Test importing from text notation"""
        tool = ImportScoreTool(score_manager)
        result = await tool.execute("melody", "C4 E4 G4 C5")
        
        assert result["status"] == "success"
        assert result["source_type"] == "text"
        assert result["num_notes"] == 4
        assert "melody" in score_manager
    
    @pytest.mark.asyncio
    async def test_import_auto_detection(self, score_manager):
        """Test auto-detection of source types"""
        tool = ImportScoreTool(score_manager)
        
        # Should detect as corpus
        result = await tool.execute("bach", "bach/bwv7.7", source_type="auto")
        assert result["source_type"] == "corpus"
        
        # Should detect as text
        result = await tool.execute("notes", "D4 F#4 A4 D5", source_type="auto")
        assert result["source_type"] == "text"
    
    @pytest.mark.asyncio
    async def test_import_invalid_text(self, score_manager):
        """Test error handling for invalid note text"""
        tool = ImportScoreTool(score_manager)
        result = await tool.execute("bad", "C4 INVALID G4")
        
        assert result["status"] == "error"
        assert "Invalid note" in result["message"]
        assert "bad" not in score_manager
    
    @pytest.mark.asyncio
    async def test_import_duplicate_id(self, score_manager):
        """Test error for duplicate score ID"""
        tool = ImportScoreTool(score_manager)
        
        # First import
        await tool.execute("dup", "C4 D4 E4")
        
        # Try duplicate
        result = await tool.execute("dup", "F4 G4 A4")
        assert result["status"] == "error"
        assert "already exists" in result["message"]
    
    @pytest.mark.asyncio
    async def test_import_empty_inputs(self, score_manager):
        """Test validation of empty inputs"""
        tool = ImportScoreTool(score_manager)
        
        # Empty ID
        result = await tool.execute("", "C4 D4")
        assert result["status"] == "error"
        assert "cannot be empty" in result["message"]
        
        # Empty source
        result = await tool.execute("test", "")
        assert result["status"] == "error"
        assert "cannot be empty" in result["message"]
    
    @pytest.mark.asyncio
    async def test_import_invalid_corpus(self, score_manager):
        """Test handling of invalid corpus paths"""
        tool = ImportScoreTool(score_manager)
        result = await tool.execute("invalid", "fake/path")
        
        assert result["status"] == "error"
        assert "not found" in result["message"] or "Failed" in result["message"]
    
    @pytest.mark.asyncio
    async def test_import_with_progress(self, score_manager):
        """Test progress reporting"""
        tool = ImportScoreTool(score_manager)
        progress_calls = []
        
        def progress_callback(percent, message):
            progress_calls.append((percent, message))
        
        tool.set_progress_callback(progress_callback)
        await tool.execute("test", "C4 D4 E4 F4 G4")
        
        assert len(progress_calls) > 0
        assert progress_calls[-1][0] == 1.0  # 100% complete


class TestListScoresTool:
    """Comprehensive tests for ListScoresTool"""
    
    @pytest.mark.asyncio
    async def test_list_empty(self, score_manager):
        """Test listing when no scores loaded"""
        tool = ListScoresTool(score_manager)
        result = await tool.execute()
        
        assert result["status"] == "success"
        assert result["count"] == 0
        assert result["scores"] == []
    
    @pytest.mark.asyncio
    async def test_list_multiple_scores(self, score_manager, simple_score, complex_score):
        """Test listing multiple scores"""
        score_manager["simple"] = simple_score
        score_manager["complex"] = complex_score
        
        tool = ListScoresTool(score_manager)
        result = await tool.execute()
        
        assert result["status"] == "success"
        assert result["count"] == 2
        assert len(result["scores"]) == 2
        
        # Check scores are sorted by ID
        assert result["scores"][0]["id"] == "complex"
        assert result["scores"][1]["id"] == "simple"
        
        # Check metadata extraction
        complex_info = next(s for s in result["scores"] if s["id"] == "complex")
        assert complex_info["parts"] == 2
        assert "title" in complex_info
    
    @pytest.mark.asyncio
    async def test_list_with_corrupted_score(self, score_manager):
        """Test handling of corrupted score data"""
        score_manager["good"] = stream.Score()
        score_manager["bad"] = Mock(spec=stream.Score)
        score_manager["bad"].flatten.side_effect = Exception("Corrupted")
        
        tool = ListScoresTool(score_manager)
        result = await tool.execute()
        
        assert result["status"] == "success"
        assert result["count"] == 2
        
        # Bad score should have error field
        bad_info = next(s for s in result["scores"] if s["id"] == "bad")
        assert "error" in bad_info


class TestKeyAnalysisTool:
    """Comprehensive tests for KeyAnalysisTool"""
    
    @pytest.mark.asyncio
    async def test_analyze_key_simple(self, score_manager, simple_score):
        """Test basic key analysis"""
        score_manager["simple"] = simple_score
        tool = KeyAnalysisTool(score_manager)
        
        result = await tool.execute("simple")
        
        assert result["status"] == "success"
        assert "key" in result
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1
    
    @pytest.mark.asyncio
    async def test_analyze_key_all_algorithms(self, score_manager, complex_score):
        """Test with all algorithms"""
        score_manager["complex"] = complex_score
        tool = KeyAnalysisTool(score_manager)
        
        result = await tool.execute("complex", algorithm="all")
        
        assert result["status"] == "success"
        assert "algorithm_results" in result
        assert len(result["algorithm_results"]) > 0
        assert "alternatives" in result
    
    @pytest.mark.asyncio
    async def test_analyze_key_specific_algorithm(self, score_manager, simple_score):
        """Test specific algorithms"""
        score_manager["test"] = simple_score
        tool = KeyAnalysisTool(score_manager)
        
        for alg in ['krumhansl', 'aarden', 'temperley']:
            result = await tool.execute("test", algorithm=alg)
            assert result["status"] == "success"
            assert "algorithm" in result
    
    @pytest.mark.asyncio
    async def test_analyze_key_nonexistent(self, score_manager):
        """Test error for non-existent score"""
        tool = KeyAnalysisTool(score_manager)
        result = await tool.execute("nonexistent")
        
        assert result["status"] == "error"
        assert "not found" in result["message"]
    
    @pytest.mark.asyncio
    async def test_analyze_key_invalid_algorithm(self, score_manager, simple_score):
        """Test error for invalid algorithm"""
        score_manager["test"] = simple_score
        tool = KeyAnalysisTool(score_manager)
        
        result = await tool.execute("test", algorithm="invalid")
        assert result["status"] == "error"
        assert "Invalid algorithm" in result["message"]
    
    @pytest.mark.asyncio
    async def test_analyze_key_empty_score(self, score_manager, empty_score):
        """Test key analysis on empty score"""
        score_manager["empty"] = empty_score
        tool = KeyAnalysisTool(score_manager)
        
        result = await tool.execute("empty")
        # Should still return a result, even if low confidence
        assert result["status"] == "success"
        assert result["confidence"] >= 0


class TestChordAnalysisTool:
    """Comprehensive tests for ChordAnalysisTool"""
    
    @pytest.mark.asyncio
    async def test_analyze_chords_basic(self, score_manager, complex_score):
        """Test basic chord analysis"""
        score_manager["test"] = complex_score
        tool = ChordAnalysisTool(score_manager)
        
        result = await tool.execute("test")
        
        assert result["status"] == "success"
        assert "total_chords" in result
        assert "chord_progression" in result
        assert "harmonic_rhythm" in result
        assert "summary" in result
    
    @pytest.mark.asyncio
    async def test_analyze_chords_with_roman(self, score_manager, complex_score):
        """Test chord analysis with Roman numerals"""
        score_manager["test"] = complex_score
        tool = ChordAnalysisTool(score_manager)
        
        result = await tool.execute("test", include_roman_numerals=True)
        
        assert result["status"] == "success"
        assert "roman_numerals" in result
        assert "analysis_key" in result
    
    @pytest.mark.asyncio
    async def test_analyze_chords_empty_score(self, score_manager, empty_score):
        """Test chord analysis on empty score"""
        score_manager["empty"] = empty_score
        tool = ChordAnalysisTool(score_manager)
        
        result = await tool.execute("empty")
        
        assert result["status"] == "success"
        assert result["total_chords"] == 0
        assert result["chord_progression"] == []
    
    @pytest.mark.asyncio
    async def test_analyze_chords_inversions(self, score_manager):
        """Test detection of chord inversions"""
        s = stream.Score()
        p = stream.Part()
        # C major in first inversion (E-G-C)
        p.append(chord.Chord(['E3', 'G3', 'C4']))
        s.append(p)
        
        score_manager["inv"] = s
        tool = ChordAnalysisTool(score_manager)
        
        result = await tool.execute("inv", include_inversions=True)
        
        assert result["status"] == "success"
        if result["chord_progression"]:
            assert "inversion" in result["chord_progression"][0]


class TestScoreInfoTool:
    """Comprehensive tests for ScoreInfoTool"""
    
    @pytest.mark.asyncio
    async def test_get_info_comprehensive(self, score_manager, complex_score):
        """Test comprehensive score info extraction"""
        score_manager["test"] = complex_score
        tool = ScoreInfoTool(score_manager)
        
        result = await tool.execute("test")
        
        assert result["status"] == "success"
        assert result["exists"] is True
        assert "metadata" in result
        assert result["metadata"]["title"] == "Test Piece"
        assert result["metadata"]["composer"] == "Test Composer"
        assert result["num_parts"] == 2
        assert result["tempo_bpm"] == 120
        assert "time_signatures" in result
    
    @pytest.mark.asyncio
    async def test_get_info_instruments(self, score_manager, complex_score):
        """Test instrument analysis"""
        score_manager["test"] = complex_score
        tool = ScoreInfoTool(score_manager)
        
        result = await tool.execute("test", include_instruments=True)
        
        assert result["status"] == "success"
        assert "instruments" in result
        assert len(result["instruments"]) == 2
        
        # Check part names
        part_names = [inst["part_name"] for inst in result["instruments"]]
        assert "Soprano" in part_names
        assert "Bass" in part_names
    
    @pytest.mark.asyncio
    async def test_get_info_structure(self, score_manager, complex_score):
        """Test structural analysis"""
        score_manager["test"] = complex_score
        tool = ScoreInfoTool(score_manager)
        
        result = await tool.execute("test", include_structure=True)
        
        assert result["status"] == "success"
        assert "structure" in result
        assert "key_signatures" in result["structure"]
    
    @pytest.mark.asyncio
    async def test_get_info_empty_score(self, score_manager, empty_score):
        """Test info extraction from empty score"""
        score_manager["empty"] = empty_score
        tool = ScoreInfoTool(score_manager)
        
        result = await tool.execute("empty")
        
        assert result["status"] == "success"
        assert result["num_notes"] == 0
        assert result["duration_seconds"] == 0


class TestExportScoreTool:
    """Comprehensive tests for ExportScoreTool"""
    
    @pytest.mark.asyncio
    async def test_export_midi(self, score_manager, simple_score):
        """Test MIDI export"""
        score_manager["test"] = simple_score
        tool = ExportScoreTool(score_manager)
        
        result = await tool.execute("test", format="midi")
        
        assert result["status"] == "success"
        assert result["format"] == "midi"
        assert os.path.exists(result["file_path"])
        assert result["file_path"].endswith(('.mid', '.midi'))
        assert result["file_size"] > 0
        
        # Cleanup
        os.unlink(result["file_path"])
    
    @pytest.mark.asyncio
    async def test_export_musicxml(self, score_manager, complex_score):
        """Test MusicXML export"""
        score_manager["test"] = complex_score
        tool = ExportScoreTool(score_manager)
        
        result = await tool.execute("test", format="musicxml")
        
        assert result["status"] == "success"
        assert result["format"] == "musicxml"
        assert os.path.exists(result["file_path"])
        assert result["file_size"] > 0
        
        # Cleanup
        os.unlink(result["file_path"])
    
    @pytest.mark.asyncio
    async def test_export_custom_path(self, score_manager, simple_score):
        """Test export with custom output path"""
        score_manager["test"] = simple_score
        tool = ExportScoreTool(score_manager)
        
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            output_path = tmp.name
        
        result = await tool.execute("test", format="midi", output_path=output_path)
        
        assert result["status"] == "success"
        assert result["file_path"] == output_path
        assert os.path.exists(output_path)
        
        # Cleanup
        os.unlink(output_path)
    
    @pytest.mark.asyncio
    async def test_export_unsupported_format(self, score_manager, simple_score):
        """Test error for unsupported format"""
        score_manager["test"] = simple_score
        tool = ExportScoreTool(score_manager)
        
        result = await tool.execute("test", format="invalid")
        
        assert result["status"] == "error"
        assert "Unsupported format" in result["message"]
    
    @pytest.mark.asyncio
    async def test_export_nonexistent_score(self, score_manager):
        """Test export of non-existent score"""
        tool = ExportScoreTool(score_manager)
        
        result = await tool.execute("nonexistent", format="midi")
        
        assert result["status"] == "error"
        assert "not found" in result["message"]
    
    @pytest.mark.asyncio 
    async def test_export_all_formats(self, score_manager, simple_score):
        """Test exporting to all supported formats"""
        score_manager["test"] = simple_score
        tool = ExportScoreTool(score_manager)
        
        # Test main formats (skip PDF/PNG which need LilyPond)
        formats = ["midi", "musicxml", "abc", "text"]
        exported_files = []
        
        for fmt in formats:
            result = await tool.execute("test", format=fmt)
            if result["status"] == "success":
                assert os.path.exists(result["file_path"])
                exported_files.append(result["file_path"])
        
        # Cleanup
        for filepath in exported_files:
            try:
                os.unlink(filepath)
            except:
                pass


class TestDeleteScoreTool:
    """Comprehensive tests for DeleteScoreTool"""
    
    @pytest.mark.asyncio
    async def test_delete_single_score(self, score_manager, simple_score):
        """Test deleting a single score"""
        score_manager["test"] = simple_score
        tool = DeleteScoreTool(score_manager)
        
        result = await tool.execute("test")
        
        assert result["status"] == "success"
        assert result["deleted_count"] == 1
        assert "test" not in score_manager
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, score_manager):
        """Test deleting non-existent score"""
        tool = DeleteScoreTool(score_manager)
        
        result = await tool.execute("nonexistent")
        
        assert result["status"] == "error"
        assert "not found" in result["message"]
    
    @pytest.mark.asyncio
    async def test_delete_all_scores(self, score_manager, simple_score, complex_score):
        """Test deleting all scores with wildcard"""
        score_manager["s1"] = simple_score
        score_manager["s2"] = complex_score
        score_manager["s3"] = stream.Score()
        
        tool = DeleteScoreTool(score_manager)
        
        result = await tool.execute("*")
        
        assert result["status"] == "success"
        assert result["deleted_count"] == 3
        assert len(score_manager) == 0
    
    @pytest.mark.asyncio
    async def test_delete_all_empty(self, score_manager):
        """Test deleting all when no scores exist"""
        tool = DeleteScoreTool(score_manager)
        
        result = await tool.execute("*")
        
        assert result["status"] == "success"
        assert result["deleted_count"] == 0
        assert "No scores to delete" in result["message"]
    
    @pytest.mark.asyncio
    async def test_delete_empty_id(self, score_manager):
        """Test validation of empty score ID"""
        tool = DeleteScoreTool(score_manager)
        
        result = await tool.execute("")
        
        assert result["status"] == "error"
        assert "cannot be empty" in result["message"]


class TestEdgeCasesAndPerformance:
    """Edge cases and performance tests"""
    
    @pytest.mark.asyncio
    async def test_large_score_import(self, score_manager):
        """Test importing a large score"""
        # Create a large score with 1000 notes
        large_text = " ".join([f"C{i%8+1}" for i in range(1000)])
        
        tool = ImportScoreTool(score_manager)
        result = await tool.execute("large", large_text)
        
        assert result["status"] == "success"
        assert result["num_notes"] == 1000
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, score_manager):
        """Test concurrent tool operations"""
        import_tool = ImportScoreTool(score_manager)
        list_tool = ListScoresTool(score_manager)
        
        # Run multiple imports concurrently
        tasks = [
            import_tool.execute(f"score_{i}", "C4 D4 E4 F4 G4")
            for i in range(10)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(r["status"] == "success" for r in results)
        
        # List should show all scores
        list_result = await list_tool.execute()
        assert list_result["count"] == 10
    
    @pytest.mark.asyncio
    async def test_memory_cleanup(self, score_manager):
        """Test that memory is properly cleaned up"""
        import_tool = ImportScoreTool(score_manager)
        delete_tool = DeleteScoreTool(score_manager)
        
        # Import and delete many times
        for i in range(100):
            await import_tool.execute(f"temp_{i}", "C4 D4 E4")
            await delete_tool.execute(f"temp_{i}")
        
        # Should have no scores left
        assert len(score_manager) == 0
    
    @pytest.mark.asyncio
    async def test_special_characters_in_id(self, score_manager):
        """Test handling of special characters in score IDs"""
        tool = ImportScoreTool(score_manager)
        
        # Various special characters
        special_ids = [
            "score-with-dash",
            "score_with_underscore", 
            "score.with.dots",
            "score@email",
            "score#hash",
            "score with spaces"
        ]
        
        for score_id in special_ids:
            result = await tool.execute(score_id, "C4 D4")
            assert result["status"] == "success"
            assert score_id in score_manager


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery scenarios"""
    
    @pytest.mark.asyncio
    async def test_corrupted_score_handling(self, score_manager):
        """Test handling of corrupted score objects"""
        # Create a mock corrupted score
        corrupted = Mock(spec=stream.Score)
        corrupted.flatten.side_effect = Exception("Corrupted data")
        
        score_manager["corrupted"] = corrupted
        
        # Each tool should handle gracefully
        key_tool = KeyAnalysisTool(score_manager)
        chord_tool = ChordAnalysisTool(score_manager)
        info_tool = ScoreInfoTool(score_manager)
        
        # These should not crash
        key_result = await key_tool.execute("corrupted")
        assert key_result["status"] == "error"
        
        chord_result = await chord_tool.execute("corrupted") 
        assert chord_result["status"] == "error"
        
        info_result = await info_tool.execute("corrupted")
        # Info tool may partially succeed
        assert info_result["status"] in ["success", "error"]
    
    @pytest.mark.asyncio
    async def test_progress_callback_exception(self, score_manager):
        """Test that exceptions in progress callbacks don't crash operations"""
        tool = ImportScoreTool(score_manager)
        
        def bad_callback(percent, message):
            raise Exception("Callback error")
        
        tool.set_progress_callback(bad_callback)
        
        # Should still complete successfully
        result = await tool.execute("test", "C4 D4 E4")
        assert result["status"] == "success"