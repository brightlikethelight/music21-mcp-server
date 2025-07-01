"""
Pytest test suite for Music21 MCP Server
Tests all core functionality with proper fixtures and assertions
"""

import os
import tempfile

import pytest
from music21 import stream

# Import server functions
from music21_mcp.server import (analyze_chords, analyze_key, delete_score,
                                export_score, get_score_info, import_score,
                                list_scores, score_manager)


@pytest.fixture(autouse=True)
async def clear_scores():
    """Clear score manager before each test"""
    score_manager.clear()
    yield
    score_manager.clear()


@pytest.fixture
async def bach_score():
    """Load a Bach chorale for testing"""
    result = await import_score("bach_test", "bach/bwv66.6")
    assert result["status"] == "success"
    return "bach_test"


@pytest.fixture
async def simple_score():
    """Create a simple test score"""
    result = await import_score("simple", "C4 D4 E4 F4 G4 A4 B4 C5")
    assert result["status"] == "success"
    return "simple"


class TestImportScore:
    """Test score import functionality"""

    @pytest.mark.asyncio
    async def test_import_corpus(self):
        """Test importing from music21 corpus"""
        result = await import_score("test1", "bach/bwv66.6")
        assert result["status"] == "success"
        assert result["score_id"] == "test1"
        assert result["num_notes"] > 0
        assert result["num_measures"] > 0
        assert result["source_type"] == "corpus"

    @pytest.mark.asyncio
    async def test_import_text(self):
        """Test importing from text notation"""
        result = await import_score("test2", "C4 E4 G4 C5")
        assert result["status"] == "success"
        assert result["num_notes"] == 4
        assert result["source_type"] == "text"

    @pytest.mark.asyncio
    async def test_import_invalid_corpus(self):
        """Test error handling for invalid corpus path"""
        result = await import_score("test3", "invalid/path")
        assert result["status"] == "error"
        assert "not found" in result["message"]

    @pytest.mark.asyncio
    async def test_import_invalid_text(self):
        """Test error handling for invalid note text"""
        result = await import_score("test4", "C4 X9 G4")
        assert result["status"] == "error"
        assert "Invalid note" in result["message"]

    @pytest.mark.asyncio
    async def test_auto_detect(self):
        """Test auto-detection of source type"""
        # Test corpus detection
        result = await import_score("test5", "mozart/k155", source_type="auto")
        assert result["status"] == "success"
        assert result["source_type"] == "corpus"

        # Test text detection
        result = await import_score("test6", "D4 F#4 A4", source_type="auto")
        assert result["status"] == "success"
        assert result["source_type"] == "text"


class TestListScores:
    """Test score listing functionality"""

    @pytest.mark.asyncio
    async def test_list_empty(self):
        """Test listing when no scores loaded"""
        result = await list_scores()
        assert result["status"] == "success"
        assert result["count"] == 0
        assert result["scores"] == []

    @pytest.mark.asyncio
    async def test_list_multiple(self):
        """Test listing multiple scores"""
        await import_score("score1", "C4 D4 E4")
        await import_score("score2", "bach/bwv66.6")

        result = await list_scores()
        assert result["status"] == "success"
        assert result["count"] == 2
        assert len(result["scores"]) == 2

        # Check score IDs are present
        score_ids = [s["id"] for s in result["scores"]]
        assert "score1" in score_ids
        assert "score2" in score_ids


class TestAnalyzeKey:
    """Test key analysis functionality"""

    @pytest.mark.asyncio
    async def test_analyze_key_bach(self, bach_score):
        """Test key analysis on Bach chorale"""
        result = await analyze_key(bach_score)
        assert result["status"] == "success"
        assert "key" in result
        assert "confidence" in result
        assert result["confidence"] > 0.5  # Bach chorales should have clear keys

    @pytest.mark.asyncio
    async def test_analyze_key_simple(self, simple_score):
        """Test key analysis on simple scale"""
        result = await analyze_key(simple_score)
        assert result["status"] == "success"
        assert result["key"] in ["C major", "A minor"]  # C major scale

    @pytest.mark.asyncio
    async def test_analyze_key_nonexistent(self):
        """Test error handling for non-existent score"""
        result = await analyze_key("does_not_exist")
        assert result["status"] == "error"
        assert "not found" in result["message"]


class TestAnalyzeChords:
    """Test chord analysis functionality"""

    @pytest.mark.asyncio
    async def test_analyze_chords_bach(self, bach_score):
        """Test chord analysis on Bach chorale"""
        result = await analyze_chords(bach_score)
        assert result["status"] == "success"
        assert result["total_chords"] > 0
        assert len(result["chord_progression"]) > 0
        assert "roman_numerals" in result

    @pytest.mark.asyncio
    async def test_analyze_chords_simple(self, simple_score):
        """Test chord analysis on simple melody"""
        result = await analyze_chords(simple_score)
        assert result["status"] == "success"
        # Simple melody might not have chords
        assert "total_chords" in result

    @pytest.mark.asyncio
    async def test_analyze_chords_empty(self):
        """Test chord analysis on empty score"""
        score_manager["empty"] = stream.Score()
        result = await analyze_chords("empty")
        assert result["status"] == "success"
        assert result["total_chords"] == 0


class TestGetScoreInfo:
    """Test score info retrieval"""

    @pytest.mark.asyncio
    async def test_get_info_bach(self, bach_score):
        """Test getting info from Bach chorale"""
        result = await get_score_info(bach_score)
        assert result["status"] == "success"
        assert result["exists"] is True
        assert result["num_parts"] > 0
        assert result["num_measures"] > 0
        assert result["duration_seconds"] > 0

    @pytest.mark.asyncio
    async def test_get_info_metadata(self, bach_score):
        """Test metadata retrieval"""
        result = await get_score_info(bach_score)
        assert "metadata" in result
        metadata = result["metadata"]
        assert "movementName" in metadata or "title" in metadata


class TestExportScore:
    """Test score export functionality"""

    @pytest.mark.asyncio
    async def test_export_midi(self, simple_score):
        """Test MIDI export"""
        result = await export_score(simple_score, format="midi")
        assert result["status"] == "success"
        assert result["format"] == "midi"
        assert os.path.exists(result["file_path"])
        assert result["file_path"].endswith(".mid")

        # Cleanup
        os.unlink(result["file_path"])

    @pytest.mark.asyncio
    async def test_export_musicxml(self, simple_score):
        """Test MusicXML export"""
        result = await export_score(simple_score, format="musicxml")
        assert result["status"] == "success"
        assert result["file_path"].endswith(".musicxml")
        assert os.path.exists(result["file_path"])

        # Cleanup
        os.unlink(result["file_path"])

    @pytest.mark.asyncio
    async def test_export_custom_path(self, simple_score):
        """Test export with custom output path"""
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            output_path = tmp.name

        result = await export_score(
            simple_score, format="midi", output_path=output_path
        )
        assert result["status"] == "success"
        assert result["file_path"] == output_path
        assert os.path.exists(output_path)

        # Cleanup
        os.unlink(output_path)

    @pytest.mark.asyncio
    async def test_export_unsupported_format(self, simple_score):
        """Test error handling for unsupported format"""
        result = await export_score(simple_score, format="invalid")
        assert result["status"] == "error"
        assert "Unsupported format" in result["message"]


class TestDeleteScore:
    """Test score deletion functionality"""

    @pytest.mark.asyncio
    async def test_delete_existing(self, simple_score):
        """Test deleting an existing score"""
        # Verify score exists
        assert simple_score in score_manager

        result = await delete_score(simple_score)
        assert result["status"] == "success"
        assert simple_score not in score_manager

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self):
        """Test deleting non-existent score"""
        result = await delete_score("does_not_exist")
        assert result["status"] == "error"
        assert "not found" in result["message"]

    @pytest.mark.asyncio
    async def test_delete_all(self):
        """Test deleting all scores"""
        # Import multiple scores
        await import_score("s1", "C4 D4")
        await import_score("s2", "E4 F4")
        await import_score("s3", "G4 A4")

        assert len(score_manager) == 3

        # Delete all
        result = await delete_score("*")
        assert result["status"] == "success"
        assert result["deleted_count"] == 3
        assert len(score_manager) == 0


class TestIntegration:
    """Integration tests combining multiple operations"""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete workflow from import to export"""
        # Import
        import_result = await import_score("workflow", "bach/bwv66.6")
        assert import_result["status"] == "success"

        # Analyze
        key_result = await analyze_key("workflow")
        assert key_result["status"] == "success"

        chord_result = await analyze_chords("workflow")
        assert chord_result["status"] == "success"

        # Get info
        info_result = await get_score_info("workflow")
        assert info_result["status"] == "success"

        # Export
        export_result = await export_score("workflow", format="musicxml")
        assert export_result["status"] == "success"
        assert os.path.exists(export_result["file_path"])

        # Cleanup
        os.unlink(export_result["file_path"])
        await delete_score("workflow")

    @pytest.mark.asyncio
    async def test_multiple_scores_analysis(self):
        """Test analyzing multiple scores"""
        scores = {
            "bach1": "bach/bwv66.6",
            "bach2": "bach/bwv7.7",
            "mozart": "mozart/k155/movement1",
        }

        # Import all scores
        for score_id, source in scores.items():
            result = await import_score(score_id, source)
            assert result["status"] == "success"

        # Analyze all scores
        results = {}
        for score_id in scores:
            key_result = await analyze_key(score_id)
            assert key_result["status"] == "success"
            results[score_id] = key_result["key"]

        # Verify we got different analyses
        assert len(set(results.values())) >= 2  # At least 2 different keys
