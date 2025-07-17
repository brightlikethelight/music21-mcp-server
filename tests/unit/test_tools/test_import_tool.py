"""
Comprehensive unit tests for ImportScoreTool
Tests all functionality including edge cases and error handling
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from music21 import stream, note, corpus, converter

from music21_mcp.tools.import_tool import ImportScoreTool


class TestImportScoreTool:
    """Test ImportScoreTool with actual implementation"""

    @pytest.fixture
    def tool(self, clean_score_storage):
        """Create tool instance with clean storage"""
        return ImportScoreTool(clean_score_storage)

    @pytest.fixture
    def sample_musicxml_file(self):
        """Create a temporary MusicXML file for testing"""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.0">
  <part-list>
    <score-part id="P1">
      <part-name>Part 1</part-name>
    </score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <note>
        <pitch>
          <step>C</step>
          <octave>4</octave>
        </pitch>
        <duration>4</duration>
        <type>whole</type>
      </note>
    </measure>
  </part>
</score-partwise>"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(xml_content)
            return f.name

    @pytest.mark.asyncio
    async def test_instantiation(self, clean_score_storage):
        """Test tool can be instantiated with score storage"""
        tool = ImportScoreTool(clean_score_storage)
        assert tool.score_manager is clean_score_storage
        assert hasattr(tool, 'execute')
        assert hasattr(tool, 'validate_inputs')

    @pytest.mark.asyncio
    async def test_import_from_corpus_success(self, tool, clean_score_storage):
        """Test importing a score from music21 corpus"""
        result = await tool.execute(
            score_id="bach_chorale",
            source="bach/bwv66.6",
            source_type="corpus"
        )
        
        assert result['status'] == 'success'
        assert result['score_id'] == 'bach_chorale'
        assert result['source_type'] == 'corpus'
        assert 'num_notes' in result
        assert 'num_measures' in result
        assert 'num_parts' in result
        assert 'pitch_range' in result
        
        # Verify score was stored
        assert 'bach_chorale' in clean_score_storage
        stored_score = clean_score_storage['bach_chorale']
        assert isinstance(stored_score, stream.Score)

    @pytest.mark.asyncio
    async def test_import_from_file_success(self, tool, clean_score_storage, sample_musicxml_file):
        """Test importing from a file"""
        try:
            result = await tool.execute(
                score_id="file_score",
                source=sample_musicxml_file,
                source_type="file"
            )
            
            assert result['status'] == 'success'
            assert result['score_id'] == 'file_score'
            assert result['source_type'] == 'file'
            assert 'file_score' in clean_score_storage
        finally:
            # Cleanup
            os.unlink(sample_musicxml_file)

    @pytest.mark.asyncio
    async def test_import_from_text_notation(self, tool, clean_score_storage):
        """Test importing from text notation"""
        result = await tool.execute(
            score_id="text_score",
            source="C4 D4 E4 F4 G4 A4 B4 C5",
            source_type="text"
        )
        
        assert result['status'] == 'success'
        assert result['score_id'] == 'text_score'
        assert result['source_type'] == 'text'
        assert result['num_notes'] == 8
        
        # Verify the notes are correct
        score = clean_score_storage['text_score']
        notes = list(score.flatten().notes)
        assert len(notes) == 8
        assert notes[0].nameWithOctave == 'C4'
        assert notes[-1].nameWithOctave == 'C5'

    @pytest.mark.asyncio
    async def test_auto_detect_corpus(self, tool):
        """Test auto-detection of corpus source"""
        result = await tool.execute(
            score_id="auto_corpus",
            source="bach/bwv66.6",
            source_type="auto"
        )
        
        assert result['status'] == 'success'
        assert result['source_type'] == 'corpus'

    @pytest.mark.asyncio
    async def test_auto_detect_file(self, tool, sample_musicxml_file):
        """Test auto-detection of file source"""
        try:
            result = await tool.execute(
                score_id="auto_file",
                source=sample_musicxml_file,
                source_type="auto"
            )
            
            assert result['status'] == 'success'
            assert result['source_type'] == 'file'
        finally:
            os.unlink(sample_musicxml_file)

    @pytest.mark.asyncio
    async def test_auto_detect_text(self, tool):
        """Test auto-detection of text notation"""
        result = await tool.execute(
            score_id="auto_text",
            source="C4 E4 G4 C5",
            source_type="auto"
        )
        
        assert result['status'] == 'success'
        assert result['source_type'] == 'text'

    @pytest.mark.asyncio
    async def test_empty_score_id_error(self, tool):
        """Test error when score_id is empty"""
        result = await tool.execute(
            score_id="",
            source="bach/bwv66.6",
            source_type="corpus"
        )
        
        assert result['status'] == 'error'
        assert 'score_id cannot be empty' in result['message']

    @pytest.mark.asyncio
    async def test_empty_source_error(self, tool):
        """Test error when source is empty"""
        result = await tool.execute(
            score_id="test",
            source="",
            source_type="corpus"
        )
        
        assert result['status'] == 'error'
        assert 'source cannot be empty' in result['message']

    @pytest.mark.asyncio
    async def test_duplicate_score_id_error(self, tool, clean_score_storage):
        """Test error when score_id already exists"""
        # First import
        await tool.execute(
            score_id="duplicate",
            source="C4 D4 E4",
            source_type="text"
        )
        
        # Try to import with same ID
        result = await tool.execute(
            score_id="duplicate",
            source="F4 G4 A4",
            source_type="text"
        )
        
        assert result['status'] == 'error'
        assert "already exists" in result['message']

    @pytest.mark.asyncio
    async def test_invalid_source_type_error(self, tool):
        """Test error with invalid source type"""
        result = await tool.execute(
            score_id="test",
            source="something",
            source_type="invalid"
        )
        
        assert result['status'] == 'error'
        assert "Invalid source_type" in result['message']

    @pytest.mark.asyncio
    async def test_file_not_found_error(self, tool):
        """Test error when file doesn't exist"""
        result = await tool.execute(
            score_id="missing",
            source="/nonexistent/file.xml",
            source_type="file"
        )
        
        assert result['status'] == 'error'
        assert "Failed to import score" in result['message']

    @pytest.mark.asyncio
    async def test_invalid_corpus_path_error(self, tool):
        """Test error with invalid corpus path"""
        result = await tool.execute(
            score_id="bad_corpus",
            source="nonexistent/piece",
            source_type="corpus"
        )
        
        assert result['status'] == 'error'
        assert "Failed to import score" in result['message']

    @pytest.mark.asyncio
    async def test_invalid_text_notation_error(self, tool):
        """Test error with invalid text notation"""
        result = await tool.execute(
            score_id="bad_text",
            source="not valid notes",
            source_type="text"
        )
        
        assert result['status'] == 'error'
        assert "Failed to import score" in result['message']

    @pytest.mark.asyncio
    async def test_progress_reporting(self, tool):
        """Test that progress is reported during import"""
        progress_calls = []
        
        def progress_callback(percent, message):
            progress_calls.append((percent, message))
        
        tool.set_progress_callback(progress_callback)
        
        await tool.execute(
            score_id="progress_test",
            source="C4 D4 E4 F4",
            source_type="text"
        )
        
        # Should have progress calls
        assert len(progress_calls) > 0
        assert any(0.0 <= p[0] <= 1.0 for p in progress_calls)
        assert any("Import complete" in p[1] for p in progress_calls)

    @pytest.mark.asyncio
    async def test_metadata_extraction(self, tool, clean_score_storage):
        """Test that metadata is correctly extracted"""
        result = await tool.execute(
            score_id="metadata_test",
            source="C3 E3 G3 C4 E4 G4 C5 E5",
            source_type="text"
        )
        
        assert result['status'] == 'success'
        assert result['num_notes'] == 8
        assert result['pitch_range'] > 0  # C3 to E5 is 16 semitones
        assert result['num_parts'] == 1

    @pytest.mark.asyncio
    async def test_complex_text_notation(self, tool, clean_score_storage):
        """Test importing complex text notation with accidentals"""
        result = await tool.execute(
            score_id="complex_text",
            source="C#4 Eb4 F#4 Bb4",
            source_type="text"
        )
        
        assert result['status'] == 'success'
        assert result['num_notes'] == 4
        
        score = clean_score_storage['complex_text']
        notes = list(score.flatten().notes)
        assert notes[0].nameWithOctave == 'C#4'
        assert notes[1].nameWithOctave == 'E-4'

    @pytest.mark.asyncio
    async def test_edge_case_single_note(self, tool, clean_score_storage):
        """Test importing a single note"""
        result = await tool.execute(
            score_id="single_note",
            source="C4",
            source_type="text"
        )
        
        assert result['status'] == 'success'
        assert result['num_notes'] == 1
        assert result['pitch_range'] == 0

    @pytest.mark.asyncio
    async def test_supported_file_extensions(self, tool):
        """Test that tool recognizes supported file extensions"""
        extensions = ImportScoreTool.SUPPORTED_FILE_EXTENSIONS
        assert '.xml' in extensions
        assert '.musicxml' in extensions
        assert '.mid' in extensions
        assert '.midi' in extensions
        assert '.abc' in extensions
        assert '.krn' in extensions
        assert '.mei' in extensions

    def test_is_note_like_method(self, tool):
        """Test the _is_note_like helper method"""
        assert tool._is_note_like("C4") is True
        assert tool._is_note_like("C#4") is True
        assert tool._is_note_like("Bb3") is True
        assert tool._is_note_like("G10") is True
        assert tool._is_note_like("Hello") is False
        assert tool._is_note_like("123") is False
        assert tool._is_note_like("C") is False

    @pytest.mark.asyncio
    async def test_unknown_source_type_after_detection(self, tool):
        """Test handling of source that can't be detected"""
        with patch.object(tool, '_detect_source_type', return_value='unknown'):
            result = await tool.execute(
                score_id="unknown",
                source="something",
                source_type="auto"
            )
            
            assert result['status'] == 'error'
            assert "Unknown source type" in result['message']