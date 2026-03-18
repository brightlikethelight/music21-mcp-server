"""Tests for server_minimal.py tool functions and resources.

Tests cover:
1. Direct calls to @mcp.tool() wrapper functions via .fn accessor
2. MCP resource handlers via .fn accessor
3. Module-level objects and registration verification
"""

import pytest

import music21_mcp.server_minimal as sm
from music21_mcp.server_minimal import mcp_adapter

# ---------------------------------------------------------------------------
# Part 1: Direct wrapper function tests
# FastMCP @mcp.tool() wraps functions into FunctionTool objects.
# The original async function is accessible via the .fn attribute.
# ---------------------------------------------------------------------------


class TestHealthCheckWrapper:
    @pytest.mark.asyncio
    async def test_health_check_returns_healthy(self):
        result = await sm.health_check.fn()
        assert result["status"] == "healthy"
        assert result["server"] == "Music21 MCP Server - Minimal"
        assert "tools_available" in result
        assert "adapter_version" in result
        assert "core_service_healthy" in result

    @pytest.mark.asyncio
    async def test_health_check_tool_count(self):
        result = await sm.health_check.fn()
        assert result["tools_available"] == 13


class TestScoreManagementWrappers:
    @pytest.mark.asyncio
    async def test_list_scores_wrapper(self):
        result = await sm.list_scores.fn()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_import_score_wrapper(self):
        result = await sm.import_score.fn("wrapper_test", "bach/bwv66.6", "corpus")
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_score_info_wrapper_missing(self):
        result = await sm.score_info.fn("nonexistent_wrapper")
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_export_score_wrapper_missing(self):
        result = await sm.export_score.fn("nonexistent_wrapper")
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_delete_score_wrapper_missing(self):
        result = await sm.delete_score.fn("nonexistent_wrapper")
        assert result["status"] == "error"


class TestAnalysisWrappers:
    @pytest.mark.asyncio
    async def test_key_analysis_wrapper_missing(self):
        result = await sm.key_analysis.fn("nonexistent_wrapper")
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_chord_analysis_wrapper_missing(self):
        result = await sm.chord_analysis.fn("nonexistent_wrapper")
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_harmony_analysis_wrapper_missing(self):
        result = await sm.harmony_analysis.fn("nonexistent_wrapper")
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_voice_leading_analysis_wrapper_missing(self):
        result = await sm.voice_leading_analysis.fn("nonexistent_wrapper")
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_pattern_recognition_wrapper_missing(self):
        result = await sm.pattern_recognition.fn("nonexistent_wrapper")
        assert result["status"] == "error"


class TestGenerationWrappers:
    @pytest.mark.asyncio
    async def test_harmonize_melody_wrapper_missing(self):
        result = await sm.harmonize_melody.fn("nonexistent_wrapper")
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_generate_counterpoint_wrapper_missing(self):
        result = await sm.generate_counterpoint.fn("nonexistent_wrapper")
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_imitate_style_wrapper_no_args(self):
        result = await sm.imitate_style.fn()
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_imitate_style_wrapper_missing_score(self):
        result = await sm.imitate_style.fn(score_id="nonexistent_wrapper")
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# Part 2: MCP resource handler tests
# FastMCP wraps @mcp.resource() into FunctionResource / FunctionResourceTemplate.
# The original async function is accessible via the .fn attribute.
# ---------------------------------------------------------------------------


class TestMCPResources:
    @pytest.mark.asyncio
    async def test_list_scores_resource(self):
        result = await sm.list_scores_resource.fn()
        assert isinstance(result, dict)
        assert "contents" in result
        assert isinstance(result["contents"], list)

    @pytest.mark.asyncio
    async def test_get_score_resource_missing(self):
        result = await sm.get_score_resource.fn("nonexistent_resource")
        assert isinstance(result, dict)
        assert "contents" in result
        assert len(result["contents"]) == 1
        assert result["contents"][0]["uri"] == "music21://scores/nonexistent_resource"


# ---------------------------------------------------------------------------
# Part 3: Module-level object and registration tests
# ---------------------------------------------------------------------------


class TestModuleLevelObjects:
    """Test that server_minimal module-level objects are properly initialized."""

    def test_mcp_adapter_exists(self):
        assert mcp_adapter is not None

    def test_supported_tools_list(self):
        tools = mcp_adapter.get_supported_tools()
        assert len(tools) == 13
        assert "import_score" in tools
        assert "health_check" not in tools  # health_check is server-level only

    def test_protocol_compatibility(self):
        compat = mcp_adapter.check_protocol_compatibility()
        assert "supported_version" in compat
        assert "current_version" in compat

    def test_mcp_server_instance(self):
        assert sm.mcp is not None
        assert sm.mcp.name == "Music21 MCP Server - Minimal"

    def test_has_mcp_flag(self):
        assert sm.HAS_MCP is True

    def test_main_function_exists(self):
        assert callable(sm.main)


# ---------------------------------------------------------------------------
# Part 4: main() branch coverage tests
# ---------------------------------------------------------------------------


class TestMainFunction:
    """Test main() branches: no MCP, keyboard interrupt, exception, success."""

    def test_main_no_mcp(self, monkeypatch):
        """When HAS_MCP is False, main() logs error and returns early."""
        monkeypatch.setattr(sm, "HAS_MCP", False)
        errors = []
        monkeypatch.setattr(sm.logger, "error", lambda msg: errors.append(msg))
        sm.main()
        assert any("MCP package not available" in e for e in errors)

    def test_main_keyboard_interrupt(self, monkeypatch):
        """KeyboardInterrupt during mcp.run() triggers graceful shutdown log."""
        monkeypatch.setattr(sm, "HAS_MCP", True)
        monkeypatch.setattr(
            sm.mcp, "run", lambda: (_ for _ in ()).throw(KeyboardInterrupt)
        )
        infos = []
        monkeypatch.setattr(sm.logger, "info", lambda msg: infos.append(msg))
        sm.main()
        assert any("stopped by user" in i for i in infos)

    def test_main_exception(self, monkeypatch):
        """RuntimeError during mcp.run() is logged and re-raised."""
        monkeypatch.setattr(sm, "HAS_MCP", True)

        def _raise():
            raise RuntimeError("test boom")

        monkeypatch.setattr(sm.mcp, "run", _raise)
        errors = []
        monkeypatch.setattr(sm.logger, "error", lambda msg: errors.append(msg))
        monkeypatch.setattr(sm.logger, "info", lambda msg: None)
        with pytest.raises(RuntimeError, match="test boom"):
            sm.main()
        assert any("Server error" in e for e in errors)

    def test_main_success(self, monkeypatch):
        """When mcp.run() completes without error, main() exits cleanly."""
        monkeypatch.setattr(sm, "HAS_MCP", True)
        monkeypatch.setattr(sm.mcp, "run", lambda: None)
        monkeypatch.setattr(sm.logger, "info", lambda msg: None)
        sm.main()  # should not raise
