"""Tests for mcp_tool decorator error handling branches in mcp_adapter.py"""

import pytest

from music21_mcp.adapters.mcp_adapter import MCPAdapter
from music21_mcp.exceptions import (
    AnalysisError,
    ExportError,
    GenerationError,
    Music21MCPError,
    ScoreImportError,
    ScoreNotFoundError,
    ValidationError,
)


@pytest.fixture
def adapter():
    return MCPAdapter()


class TestMCPToolErrorHandling:
    @pytest.mark.asyncio
    async def test_score_not_found_error(self, adapter):
        result = await adapter.score_info("nonexistent_score_xyz")
        assert result["status"] == "error"
        error_text = result.get("error", "") or result.get("message", "")
        assert "nonexistent_score_xyz" in error_text

    @pytest.mark.asyncio
    async def test_validation_error_on_imitate_style(self, adapter):
        """imitate_style raises ValidationError when neither score_id nor composer given"""
        result = await adapter.imitate_style()
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_key_analysis_score_not_found(self, adapter):
        result = await adapter.key_analysis("missing_score_abc")
        assert result["status"] == "error"
        error_text = result.get("error", "") or result.get("message", "")
        assert "missing_score_abc" in error_text

    @pytest.mark.asyncio
    async def test_chord_analysis_score_not_found(self, adapter):
        result = await adapter.chord_analysis("no_such_score")
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_harmony_analysis_score_not_found(self, adapter):
        result = await adapter.harmony_analysis("fake_id")
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_export_score_not_found(self, adapter):
        result = await adapter.export_score("ghost_score")
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_delete_score_not_found(self, adapter):
        result = await adapter.delete_score("phantom_score")
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_pattern_recognition_score_not_found(self, adapter):
        result = await adapter.pattern_recognition("vanished_score")
        assert result["status"] == "error"
