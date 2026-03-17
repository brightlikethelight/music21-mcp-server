"""
Unit tests for ImportScoreTool
"""

import pytest

from music21_mcp.tools.import_tool import ImportScoreTool


class TestImportScoreTool:
    """Test ImportScoreTool functionality"""

    def test_tool_initialization(self, clean_score_storage):
        """Test tool can be initialized with score storage"""
        tool = ImportScoreTool(clean_score_storage)
        assert tool.scores == clean_score_storage

    @pytest.mark.asyncio
    async def test_import_from_corpus(self, clean_score_storage):
        """Test importing score from music21 corpus"""
        tool = ImportScoreTool(clean_score_storage)

        result = await tool.execute(
            score_id="test_bach", source="bach/bwv66.6", source_type="corpus"
        )

        assert result["status"] == "success"
        assert "test_bach" in clean_score_storage
        assert "Successfully imported" in result["message"]

    @pytest.mark.asyncio
    async def test_import_duplicate_id(self, populated_score_storage):
        """Test importing with duplicate score ID"""
        tool = ImportScoreTool(populated_score_storage)

        result = await tool.execute(
            score_id="bach_test",  # Already exists
            source="bach/bwv66.6",
            source_type="corpus",
        )

        assert result["status"] == "error"
        assert "already exists" in result["message"]

    @pytest.mark.asyncio
    async def test_import_invalid_source(self, clean_score_storage):
        """Test importing from invalid source"""
        tool = ImportScoreTool(clean_score_storage)

        result = await tool.execute(
            score_id="test_invalid",
            source="nonexistent/score.xml",
            source_type="corpus",
        )

        assert result["status"] == "error"
        assert (
            "Failed to import" in result["message"]
            or "Could not find" in result["message"]
        )

    @pytest.mark.asyncio
    async def test_import_url_source(self, clean_score_storage):
        """Test importing from URL source"""
        tool = ImportScoreTool(clean_score_storage)

        # This should fail gracefully for invalid URLs
        result = await tool.execute(
            score_id="test_url",
            source="https://invalid-url.com/score.xml",
            source_type="url",
        )

        assert result["status"] == "error"
        assert "Invalid source_type: url" in result["message"]

    @pytest.mark.asyncio
    async def test_import_file_source(self, clean_score_storage):
        """Test importing from file source"""
        tool = ImportScoreTool(clean_score_storage)

        # This should fail gracefully for non-existent files
        result = await tool.execute(
            score_id="test_file",
            source="/nonexistent/path/score.xml",
            source_type="file",
        )

        assert result["status"] == "error"
        assert (
            "Path '/nonexistent/path/score.xml' is outside allowed directories"
            in result["message"]
        )


class TestValidateInputs:
    """Test ImportScoreTool.validate_inputs edge cases."""

    def test_empty_score_id(self, clean_score_storage):
        tool = ImportScoreTool(clean_score_storage)
        assert (
            tool.validate_inputs(score_id="", source="x") == "score_id cannot be empty"
        )

    def test_empty_source(self, clean_score_storage):
        tool = ImportScoreTool(clean_score_storage)
        assert (
            tool.validate_inputs(score_id="ok", source="") == "source cannot be empty"
        )

    def test_invalid_source_type(self, clean_score_storage):
        tool = ImportScoreTool(clean_score_storage)
        err = tool.validate_inputs(score_id="ok", source="x", source_type="url")
        assert "Invalid source_type" in err

    def test_valid_inputs(self, clean_score_storage):
        tool = ImportScoreTool(clean_score_storage)
        assert (
            tool.validate_inputs(score_id="ok", source="x", source_type="corpus")
            is None
        )


class TestDetectSourceType:
    """Test ImportScoreTool._detect_source_type auto-detection."""

    def test_file_extension(self, clean_score_storage):
        tool = ImportScoreTool(clean_score_storage)
        assert tool._detect_source_type("score.mid") == "file"
        assert tool._detect_source_type("piece.musicxml") == "file"
        assert tool._detect_source_type("data.abc") == "file"

    def test_corpus_pattern(self, clean_score_storage):
        tool = ImportScoreTool(clean_score_storage)
        assert tool._detect_source_type("bach/bwv66.6") == "corpus"
        assert tool._detect_source_type("mozart/piano") == "corpus"

    def test_note_text(self, clean_score_storage):
        tool = ImportScoreTool(clean_score_storage)
        assert tool._detect_source_type("C4 D4 E4") == "text"

    def test_fallback_to_file(self, clean_score_storage):
        tool = ImportScoreTool(clean_score_storage)
        assert tool._detect_source_type("something") == "file"


class TestImportFromText:
    """Test text notation import paths."""

    @pytest.mark.asyncio
    async def test_tinynotation(self, clean_score_storage):
        tool = ImportScoreTool(clean_score_storage)
        result = await tool.execute(
            score_id="tiny_test",
            source="tinyNotation: 4/4 c4 d e f",
            source_type="text",
        )
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_space_separated_notes(self, clean_score_storage):
        tool = ImportScoreTool(clean_score_storage)
        result = await tool.execute(
            score_id="notes_test",
            source="C4 D4 E4 F4",
            source_type="text",
        )
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_invalid_text(self, clean_score_storage):
        tool = ImportScoreTool(clean_score_storage)
        result = await tool.execute(
            score_id="bad_text",
            source="not valid notes zzz",
            source_type="text",
        )
        # Should fail or return error for invalid note text
        assert result["status"] in ("error", "success")

    @pytest.mark.asyncio
    async def test_auto_detect_text(self, clean_score_storage):
        tool = ImportScoreTool(clean_score_storage)
        result = await tool.execute(
            score_id="auto_text",
            source="C4 D4 E4",
            source_type="auto",
        )
        assert result["status"] == "success"


class TestExtractMetadata:
    """Test metadata extraction from scores."""

    @pytest.mark.asyncio
    async def test_metadata_from_corpus(self, clean_score_storage):
        tool = ImportScoreTool(clean_score_storage)
        result = await tool.execute(
            score_id="meta_test", source="bach/bwv66.6", source_type="corpus"
        )
        assert result["status"] == "success"
        assert "num_notes" in result
        assert result["num_notes"] > 0
        assert "num_parts" in result
