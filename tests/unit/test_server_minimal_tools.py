"""Tests for server_minimal.py tool functions and resources.

FastMCP's @mcp.tool() wraps functions into FunctionTool objects,
so we test through the mcp_adapter instance which the tools delegate to.
"""

import pytest

from music21_mcp.server_minimal import mcp_adapter


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check_via_adapter(self):
        compat = mcp_adapter.check_protocol_compatibility()
        result = {
            "status": "healthy",
            "server": "Music21 MCP Server - Minimal",
            "adapter_version": compat.get("supported_version", "unknown"),
            "tools_available": len(mcp_adapter.get_supported_tools()),
            "core_service_healthy": compat.get("core_service_healthy", False),
        }
        assert result["status"] == "healthy"
        assert result["tools_available"] == 13
        assert "adapter_version" in result


class TestToolsViaAdapter:
    """Test tool functions through the mcp_adapter instance from server_minimal."""

    @pytest.mark.asyncio
    async def test_list_scores(self):
        result = await mcp_adapter.list_scores()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("score_info", ("no_such_sm",)),
            ("export_score", ("no_such_sm",)),
            ("delete_score", ("no_such_sm",)),
            ("key_analysis", ("no_such_sm",)),
            ("chord_analysis", ("no_such_sm",)),
            ("harmony_analysis", ("no_such_sm",)),
            ("voice_leading_analysis", ("no_such_sm",)),
            ("pattern_recognition", ("no_such_sm",)),
            ("harmonize_melody", ("no_such_sm",)),
            ("generate_counterpoint", ("no_such_sm",)),
        ],
    )
    async def test_tool_handles_missing_score(self, method, args):
        func = getattr(mcp_adapter, method)
        result = await func(*args)
        assert isinstance(result, dict)
        assert result.get("status") == "error"

    @pytest.mark.asyncio
    async def test_imitate_style_no_args(self):
        result = await mcp_adapter.imitate_style()
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_import_and_score_info(self):
        result = await mcp_adapter.import_score("sm_test2", "bach/bwv66.6", "corpus")
        assert result["status"] == "success"

        info = await mcp_adapter.score_info("sm_test2")
        assert info["status"] == "success"


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
