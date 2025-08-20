#!/usr/bin/env python3
"""
Music21 MCP Server - Minimal Implementation

A simple, industry-standard MCP server that provides music21 functionality.
Following best practices from successful MCP servers like GitHub's official implementation.

Features:
- Zero configuration required
- Works immediately with Claude Desktop
- Simple, focused, does one thing well
- High test coverage
- MCP protocol compliant

Usage:
    python -m music21_mcp.server_minimal
"""

import logging

# Try to import FastMCP 2.x
try:
    from fastmcp import FastMCP

    HAS_MCP = True
except ImportError:
    HAS_MCP = False

    class _FastMCP:  # Renamed to avoid redefinition
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "FastMCP package not installed. Please install with: pip install fastmcp"
            )

    FastMCP = _FastMCP  # type: ignore  # Fallback class for when fastmcp not installed


# Import protocol adapter (isolates MCP concerns from core value)
try:
    # Try relative imports first (when run as module)
    from .adapters import MCPAdapter
except ImportError:
    # Fallback to absolute imports (when run directly)
    from music21_mcp.adapters import MCPAdapter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP adapter (contains all core music analysis functionality)
# This isolates protocol concerns from valuable music analysis features
mcp_adapter = MCPAdapter()

# Create FastMCP server
mcp: FastMCP = FastMCP("Music21 MCP Server - Minimal")

logger.info("Music21 MCP Server initialized with adapter pattern")

# Register tools with FastMCP using adapter pattern
# All functionality is in the adapter, MCP just handles protocol routing


@mcp.tool()
async def import_score(score_id: str, source: str, source_type: str = "corpus"):
    """Import a score from various sources"""
    return await mcp_adapter.import_score(score_id, source, source_type)


@mcp.tool()
async def list_scores():
    """List all available scores"""
    return await mcp_adapter.list_scores()


@mcp.tool()
async def score_info(score_id: str):
    """Get detailed information about a score"""
    return await mcp_adapter.score_info(score_id)


@mcp.tool()
async def export_score(score_id: str, format: str = "musicxml"):
    """Export a score to various formats"""
    return await mcp_adapter.export_score(score_id, format)


@mcp.tool()
async def delete_score(score_id: str):
    """Delete a score from storage"""
    return await mcp_adapter.delete_score(score_id)


@mcp.tool()
async def key_analysis(score_id: str):
    """Analyze the key signature of a score"""
    return await mcp_adapter.key_analysis(score_id)


@mcp.tool()
async def chord_analysis(score_id: str):
    """Analyze chord progressions in a score"""
    return await mcp_adapter.chord_analysis(score_id)


@mcp.tool()
async def harmony_analysis(score_id: str, analysis_type: str = "roman"):
    """Perform harmony analysis (roman numeral or functional)"""
    return await mcp_adapter.harmony_analysis(score_id, analysis_type)


@mcp.tool()
async def voice_leading_analysis(score_id: str):
    """Analyze voice leading patterns in a score"""
    return await mcp_adapter.voice_leading_analysis(score_id)


@mcp.tool()
async def pattern_recognition(score_id: str, pattern_type: str = "melodic"):
    """Recognize patterns in music"""
    return await mcp_adapter.pattern_recognition(score_id, pattern_type)


@mcp.tool()
async def harmonize_melody(
    score_id: str, style: str = "classical", voice_parts: int = 4
):
    """Generate harmonization for a melody in various styles (classical, jazz, pop, modal)"""
    return await mcp_adapter.harmonize_melody(score_id, style, voice_parts)


@mcp.tool()
async def generate_counterpoint(
    score_id: str, species: int = 1, voice_position: str = "above"
):
    """Generate species counterpoint (1=note-against-note, 2=2:1, 3=3:1, 4=syncopated, 5=florid)"""
    return await mcp_adapter.generate_counterpoint(score_id, species, voice_position)


@mcp.tool()
async def imitate_style(
    score_id: str = None,
    composer: str = None,
    generation_length: int = 16,
    complexity: str = "medium",
):
    """Generate music imitating a specific composer style (bach, mozart, chopin, debussy) or analyze score for style"""
    return await mcp_adapter.imitate_style(
        score_id, composer, generation_length, complexity
    )


# Additional tools available through adapter but simplified for MCP
@mcp.tool()
async def health_check():
    """Check server and adapter health"""
    compatibility = mcp_adapter.check_protocol_compatibility()
    return {
        "status": "healthy",
        "server": "Music21 MCP Server - Minimal",
        "adapter_version": compatibility.get("supported_version", "unknown"),
        "tools_available": len(mcp_adapter.get_supported_tools()),
        "core_service_healthy": compatibility.get("core_service_healthy", False),
    }


# MCP Resources - for score browsing (using adapter)
@mcp.resource("music21://scores")
async def list_scores_resource():
    """List all scores as an MCP resource"""
    scores_result = await mcp_adapter.list_scores()
    score_ids = scores_result.get("data", {}).get("scores", [])

    return {
        "contents": [
            {
                "uri": f"music21://scores/{score_id}",
                "name": score_id,
                "mimeType": "application/json",
            }
            for score_id in score_ids
        ]
    }


@mcp.resource("music21://scores/{score_id}")
async def get_score_resource(score_id: str):
    """Get score information as an MCP resource"""
    result = await mcp_adapter.score_info(score_id)

    return {
        "contents": [
            {
                "uri": f"music21://scores/{score_id}",
                "name": score_id,
                "mimeType": "application/json",
                "text": str(result),
            }
        ]
    }


def main():
    """Main entry point"""
    if not HAS_MCP:
        logger.error("MCP package not available. Please install with: pip install mcp")
        return

    logger.info("🎵 Music21 MCP Server - Minimal Implementation")
    logger.info("📊 16 music analysis and generation tools available")
    logger.info("🚀 Starting server...")

    try:
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    main()
