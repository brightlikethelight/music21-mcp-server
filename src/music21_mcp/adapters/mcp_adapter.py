#!/usr/bin/env python3
"""
MCP Protocol Adapter
Isolates MCP protocol concerns from core music analysis service

This adapter handles:
- MCP protocol specifics (which change frequently)
- Tool registration and routing
- Response format translation
- Error handling and logging

The core music analysis service remains completely isolated from MCP.
When MCP breaks (every 3-6 months), only this adapter needs updating.
"""

import logging
from typing import Any

from ..services import MusicAnalysisService

logger = logging.getLogger(__name__)


class MCPAdapter:
    """
    Adapter between MCP protocol and core music analysis service

    Handles protocol volatility by isolating all MCP-specific code.
    When FastMCP introduces breaking changes, only this class needs updates.
    """

    def __init__(self):
        """Initialize MCP adapter with core service"""
        self.core_service = MusicAnalysisService()
        self.mcp_version = "fastmcp-2.9.0"  # Track which version we support

        logger.info(f"MCP adapter initialized for {self.mcp_version}")

    # === MCP Tool Interface ===
    # These methods match MCP tool signatures but delegate to core service

    async def import_score(
        self, score_id: str, source: str, source_type: str = "corpus"
    ) -> dict[str, Any]:
        """MCP tool: Import a score from various sources"""
        try:
            result = await self.core_service.import_score(score_id, source, source_type)
            return self._format_mcp_response(result, "import_score")
        except Exception as e:
            return self._format_mcp_error(f"Import failed: {str(e)}", "import_score")

    async def list_scores(self) -> dict[str, Any]:
        """MCP tool: List all available scores"""
        try:
            result = await self.core_service.list_scores()
            return self._format_mcp_response(result, "list_scores")
        except Exception as e:
            return self._format_mcp_error(f"List failed: {str(e)}", "list_scores")

    async def score_info(self, score_id: str) -> dict[str, Any]:
        """MCP tool: Get detailed information about a score"""
        try:
            result = await self.core_service.get_score_info(score_id)
            return self._format_mcp_response(result, "score_info")
        except Exception as e:
            return self._format_mcp_error(f"Info failed: {str(e)}", "score_info")

    async def export_score(
        self, score_id: str, format: str = "musicxml"
    ) -> dict[str, Any]:
        """MCP tool: Export a score to various formats"""
        try:
            result = await self.core_service.export_score(score_id, format)
            return self._format_mcp_response(result, "export_score")
        except Exception as e:
            return self._format_mcp_error(f"Export failed: {str(e)}", "export_score")

    async def delete_score(self, score_id: str) -> dict[str, Any]:
        """MCP tool: Delete a score from storage"""
        try:
            result = await self.core_service.delete_score(score_id)
            return self._format_mcp_response(result, "delete_score")
        except Exception as e:
            return self._format_mcp_error(f"Delete failed: {str(e)}", "delete_score")

    async def key_analysis(self, score_id: str) -> dict[str, Any]:
        """MCP tool: Analyze the key signature of a score"""
        try:
            result = await self.core_service.analyze_key(score_id)
            return self._format_mcp_response(result, "key_analysis")
        except Exception as e:
            return self._format_mcp_error(
                f"Key analysis failed: {str(e)}", "key_analysis"
            )

    async def chord_analysis(self, score_id: str) -> dict[str, Any]:
        """MCP tool: Analyze chord progressions in a score"""
        try:
            result = await self.core_service.analyze_chords(score_id)
            return self._format_mcp_response(result, "chord_analysis")
        except Exception as e:
            return self._format_mcp_error(
                f"Chord analysis failed: {str(e)}", "chord_analysis"
            )

    async def harmony_analysis(
        self, score_id: str, analysis_type: str = "roman"
    ) -> dict[str, Any]:
        """MCP tool: Perform harmony analysis"""
        try:
            result = await self.core_service.analyze_harmony(score_id, analysis_type)
            return self._format_mcp_response(result, "harmony_analysis")
        except Exception as e:
            return self._format_mcp_error(
                f"Harmony analysis failed: {str(e)}", "harmony_analysis"
            )

    async def voice_leading_analysis(self, score_id: str) -> dict[str, Any]:
        """MCP tool: Analyze voice leading quality"""
        try:
            result = await self.core_service.analyze_voice_leading(score_id)
            return self._format_mcp_response(result, "voice_leading_analysis")
        except Exception as e:
            return self._format_mcp_error(
                f"Voice leading analysis failed: {str(e)}", "voice_leading_analysis"
            )

    async def pattern_recognition(
        self, score_id: str, pattern_type: str = "melodic"
    ) -> dict[str, Any]:
        """MCP tool: Recognize musical patterns"""
        try:
            result = await self.core_service.recognize_patterns(score_id, pattern_type)
            return self._format_mcp_response(result, "pattern_recognition")
        except Exception as e:
            return self._format_mcp_error(
                f"Pattern recognition failed: {str(e)}", "pattern_recognition"
            )

    async def harmonize_melody(
        self, score_id: str, style: str = "classical", voice_parts: int = 4
    ) -> dict[str, Any]:
        """MCP tool: Generate harmonization for a melody"""
        try:
            result = await self.core_service.harmonize_melody(score_id, style)
            return self._format_mcp_response(result, "harmonize_melody")
        except Exception as e:
            return self._format_mcp_error(
                f"Harmonization failed: {str(e)}", "harmonize_melody"
            )

    async def generate_counterpoint(
        self, score_id: str, species: int = 1, voice_position: str = "above"
    ) -> dict[str, Any]:
        """MCP tool: Generate counterpoint melody"""
        try:
            # Convert species number to string for the tool
            species_map = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}
            species_str = species_map.get(species, "first")

            # Call the tool directly since service method signature mismatch
            result = await self.core_service.counterpoint_tool.execute(
                score_id=score_id, species=species_str, voice_position=voice_position
            )
            return self._format_mcp_response(result, "generate_counterpoint")
        except Exception as e:
            return self._format_mcp_error(
                f"Counterpoint generation failed: {str(e)}", "generate_counterpoint"
            )

    async def imitate_style(
        self,
        score_id: str | None = None,
        composer: str | None = None,
        generation_length: int = 16,
        complexity: str = "medium",
    ) -> dict[str, Any]:
        """MCP tool: Generate music imitating a specific style"""
        try:
            if score_id:
                # Use score as style source
                result = await self.core_service.imitate_style(
                    score_id, composer or "custom"
                )
            elif composer:
                # Use predefined composer style
                result = await self.core_service.style_tool.execute(
                    composer=composer,
                    generation_length=generation_length,
                    complexity=complexity,
                )
            else:
                return self._format_mcp_error(
                    "Must provide either score_id or composer", "imitate_style"
                )

            return self._format_mcp_response(result, "imitate_style")
        except Exception as e:
            return self._format_mcp_error(
                f"Style imitation failed: {str(e)}", "imitate_style"
            )

    # === MCP Response Formatting ===
    # Handle MCP-specific response format requirements

    def _format_mcp_response(
        self, core_result: dict[str, Any], tool_name: str
    ) -> dict[str, Any]:
        """Format core service response for MCP protocol"""
        # MCP expects specific response format
        # This is where protocol changes usually break things
        if isinstance(core_result, dict) and "status" in core_result:
            # Core service already returns properly formatted response
            return core_result
        # Wrap raw results in MCP format
        return {
            "status": "success",
            "tool": tool_name,
            "data": core_result,
            "message": f"{tool_name} completed successfully",
        }

    def _format_mcp_error(self, error_message: str, tool_name: str) -> dict[str, Any]:
        """Format error response for MCP protocol"""
        logger.error(f"MCP tool {tool_name} error: {error_message}")
        return {
            "status": "error",
            "tool": tool_name,
            "error": error_message,
            "message": f"{tool_name} failed",
        }

    # === Protocol Evolution Support ===

    def get_supported_tools(self) -> list[str]:
        """Get list of all supported MCP tools"""
        return [
            "import_score",
            "list_scores",
            "score_info",
            "export_score",
            "delete_score",
            "key_analysis",
            "chord_analysis",
            "harmony_analysis",
            "voice_leading_analysis",
            "pattern_recognition",
            "harmonize_melody",
            "generate_counterpoint",
            "imitate_style",
        ]

    def check_protocol_compatibility(self) -> dict[str, Any]:
        """Check if current FastMCP version is compatible"""
        try:
            import fastmcp

            current_version = getattr(fastmcp, "__version__", "unknown")

            return {
                "supported_version": self.mcp_version,
                "current_version": current_version,
                "compatible": current_version.startswith("2.9"),
                "core_service_healthy": self.core_service.get_score_count() >= 0,
            }
        except ImportError:
            return {
                "supported_version": self.mcp_version,
                "current_version": "not_installed",
                "compatible": False,
                "error": "FastMCP not available",
            }

    def create_server(self) -> object:
        """Create MCP server instance (for testing)"""
        try:
            from fastmcp import FastMCP

            server: object = FastMCP()
            self._register_tools(server)
            return server
        except ImportError as err:
            raise ImportError("FastMCP not available") from err

    def _register_tools(self, server=None):
        """Register MCP tools with server (for testing)"""
        # This is a stub for testing purposes
        # In a real implementation, this would register all the tools
        # with the FastMCP server instance
        pass
