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
        assert "Failed to import" in result["message"]
