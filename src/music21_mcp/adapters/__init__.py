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

from .mcp_adapter import MCPAdapter
from .http_adapter import HTTPAdapter, create_http_server
from .cli_adapter import CLIAdapter
from .python_adapter import PythonAdapter, Music21Analysis, create_music_analyzer, create_sync_analyzer

__all__ = [
    "MCPAdapter",
    "HTTPAdapter", "create_http_server",
    "CLIAdapter", 
    "PythonAdapter", "Music21Analysis", "create_music_analyzer", "create_sync_analyzer"
]