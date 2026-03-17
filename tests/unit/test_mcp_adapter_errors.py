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


class TestMCPToolDecoratorExceptionBranches:
    """Test @mcp_tool decorator catches each exception type correctly via monkeypatch."""

    @pytest.mark.asyncio
    async def test_score_import_error(self, adapter, monkeypatch):
        async def raise_err(*a, **kw):
            raise ScoreImportError("src", "corpus", "bad data")

        monkeypatch.setattr(adapter.core_service, "import_score", raise_err)
        result = await adapter.import_score("x", "y", "corpus")
        assert result["status"] == "error"
        assert "bad data" in result["error"]

    @pytest.mark.asyncio
    async def test_export_error(self, adapter, monkeypatch):
        async def raise_err(*a, **kw):
            raise ExportError("s", "midi", "write failed")

        monkeypatch.setattr(adapter.core_service, "export_score", raise_err)
        result = await adapter.export_score("s", "midi")
        assert result["status"] == "error"
        assert "write failed" in result["error"]

    @pytest.mark.asyncio
    async def test_analysis_error(self, adapter, monkeypatch):
        async def raise_err(*a, **kw):
            raise AnalysisError("key", "s", "analysis broke")

        monkeypatch.setattr(adapter.core_service, "analyze_key", raise_err)
        result = await adapter.key_analysis("s")
        assert result["status"] == "error"
        assert "analysis broke" in result["error"]

    @pytest.mark.asyncio
    async def test_generation_error(self, adapter, monkeypatch):
        async def raise_err(*a, **kw):
            raise GenerationError("harmonization", "gen failed")

        monkeypatch.setattr(adapter.core_service, "harmonize_melody", raise_err)
        result = await adapter.harmonize_melody("s")
        assert result["status"] == "error"
        assert "gen failed" in result["error"]

    @pytest.mark.asyncio
    async def test_key_error(self, adapter, monkeypatch):
        async def raise_err(*a, **kw):
            raise KeyError("missing_field")

        monkeypatch.setattr(adapter.core_service, "analyze_chords", raise_err)
        result = await adapter.chord_analysis("s")
        assert result["status"] == "error"
        assert "missing required data" in result["error"]

    @pytest.mark.asyncio
    async def test_type_error(self, adapter, monkeypatch):
        async def raise_err(*a, **kw):
            raise TypeError("bad type")

        monkeypatch.setattr(adapter.core_service, "analyze_chords", raise_err)
        result = await adapter.chord_analysis("s")
        assert result["status"] == "error"
        assert "invalid input type" in result["error"]

    @pytest.mark.asyncio
    async def test_os_error(self, adapter, monkeypatch):
        async def raise_err(*a, **kw):
            raise OSError("disk full")

        monkeypatch.setattr(adapter.core_service, "analyze_chords", raise_err)
        result = await adapter.chord_analysis("s")
        assert result["status"] == "error"
        assert "file error" in result["error"]

    @pytest.mark.asyncio
    async def test_connection_error(self, adapter, monkeypatch):
        async def raise_err(*a, **kw):
            raise ConnectionError("refused")

        monkeypatch.setattr(adapter.core_service, "analyze_chords", raise_err)
        result = await adapter.chord_analysis("s")
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_timeout_error(self, adapter, monkeypatch):
        async def raise_err(*a, **kw):
            raise TimeoutError("timed out")

        monkeypatch.setattr(adapter.core_service, "analyze_chords", raise_err)
        result = await adapter.chord_analysis("s")
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_generic_exception(self, adapter, monkeypatch):
        async def raise_err(*a, **kw):
            raise RuntimeError("something unexpected")

        monkeypatch.setattr(adapter.core_service, "analyze_chords", raise_err)
        result = await adapter.chord_analysis("s")
        assert result["status"] == "error"
        assert "something unexpected" in result["error"]
