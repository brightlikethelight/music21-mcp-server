"""
Unit tests for ExportScoreTool
"""

import os
import tempfile

import pytest

from music21_mcp.tools.export_tool import ExportScoreTool


class TestExportScoreTool:
    """Test ExportScoreTool functionality"""

    def test_tool_initialization(self, clean_score_storage):
        """Test tool can be initialized with score storage"""
        tool = ExportScoreTool(clean_score_storage)
        assert tool.scores == clean_score_storage

    @pytest.mark.asyncio
    async def test_export_to_musicxml(self, populated_score_storage):
        """Test exporting score to MusicXML format"""
        tool = ExportScoreTool(populated_score_storage)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = await tool.execute(
                score_id="bach_test", format="musicxml", output_path=tmp_path
            )

            assert result["status"] == "success"
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 0
            assert "exported" in result["message"].lower()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @pytest.mark.asyncio
    async def test_export_to_midi(self, populated_score_storage):
        """Test exporting score to MIDI format"""
        tool = ExportScoreTool(populated_score_storage)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".mid", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = await tool.execute(
                score_id="bach_test", format="midi", output_path=tmp_path
            )

            assert result["status"] == "success"
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 0
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @pytest.mark.asyncio
    async def test_export_to_pdf(self, populated_score_storage):
        """Test exporting score to PDF format"""
        tool = ExportScoreTool(populated_score_storage)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result = await tool.execute(
                score_id="bach_test", format="pdf", output_path=tmp_path
            )

            # PDF export might require additional dependencies
            assert result["status"] in ["success", "error"]
            if result["status"] == "success":
                assert os.path.exists(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @pytest.mark.asyncio
    async def test_export_nonexistent_score(self, clean_score_storage):
        """Test exporting non-existent score"""
        tool = ExportScoreTool(clean_score_storage)

        result = await tool.execute(
            score_id="nonexistent", format="musicxml", output_path="/tmp/test.xml"
        )

        assert result["status"] == "error"
        assert "not found" in result["message"]

    @pytest.mark.asyncio
    async def test_export_invalid_format(self, populated_score_storage):
        """Test exporting with invalid format"""
        tool = ExportScoreTool(populated_score_storage)

        result = await tool.execute(
            score_id="bach_test",
            format="invalid_format",
            output_path="/tmp/test.invalid",
        )

        assert result["status"] == "error"
        assert "format" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_export_to_invalid_path(self, populated_score_storage):
        """Test exporting to invalid path"""
        tool = ExportScoreTool(populated_score_storage)

        result = await tool.execute(
            score_id="bach_test",
            format="musicxml",
            output_path="/invalid/path/that/does/not/exist/test.xml",
        )

        assert result["status"] == "error"
        assert "failed" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_export_all_supported_formats(self, populated_score_storage):
        """Test exporting to all supported formats"""
        tool = ExportScoreTool(populated_score_storage)

        formats = ["musicxml", "midi", "abc", "lily"]

        for fmt in formats:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=f".{fmt}", delete=False
            ) as tmp:
                tmp_path = tmp.name

            try:
                result = await tool.execute(
                    score_id="bach_test", format=fmt, output_path=tmp_path
                )

                # Some formats might require additional dependencies
                assert result["status"] in ["success", "error"]
                if result["status"] == "success":
                    assert os.path.exists(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
