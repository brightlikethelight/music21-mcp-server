"""
Protocol Adapters
Isolate protocol-specific code from core music analysis service

This package contains adapters for different protocols:
- MCP (Model Context Protocol) - for AI assistant integration
- HTTP (REST API) - for web service integration
- CLI - for command-line usage
- Python - for direct library usage

The core music analysis service remains protocol-independent.
"""

# Lazy imports to avoid pulling in unnecessary dependencies at startup.
# The MCP server only needs MCPAdapter; eagerly importing all adapters
# caused startup crashes when optional dependencies were missing.
from .mcp_adapter import MCPAdapter

__all__ = [
    "MCPAdapter",
    "HTTPAdapter",
    "create_http_server",
    "CLIAdapter",
    "PythonAdapter",
    "Music21Analysis",
    "create_music_analyzer",
    "create_sync_analyzer",
]


def __getattr__(name):
    """Lazy-load adapters that aren't needed for MCP startup."""
    if name == "CLIAdapter":
        from .cli_adapter import CLIAdapter

        return CLIAdapter
    if name in ("HTTPAdapter", "create_http_server"):
        from .http_adapter import HTTPAdapter, create_http_server

        return HTTPAdapter if name == "HTTPAdapter" else create_http_server
    if name in (
        "PythonAdapter",
        "Music21Analysis",
        "create_music_analyzer",
        "create_sync_analyzer",
    ):
        from .python_adapter import (
            Music21Analysis,
            PythonAdapter,
            create_music_analyzer,
            create_sync_analyzer,
        )

        _map = {
            "PythonAdapter": PythonAdapter,
            "Music21Analysis": Music21Analysis,
            "create_music_analyzer": create_music_analyzer,
            "create_sync_analyzer": create_sync_analyzer,
        }
        return _map[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
