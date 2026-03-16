# Architecture Overview

Music21 MCP Server is a multi-interface music analysis server built on the music21 library.

## Directory Structure

```
music21-mcp-server/
├── src/music21_mcp/
│   ├── __init__.py              # Package entry, exports main classes
│   ├── services.py              # Core analysis service (protocol-independent)
│   ├── server_minimal.py        # MCP server entry point
│   ├── launcher.py              # Unified launcher for all interfaces
│   │
│   ├── adapters/                # Protocol adapters
│   │   ├── mcp_adapter.py       # MCP protocol adapter
│   │   ├── http_adapter.py      # HTTP/REST API adapter
│   │   ├── cli_adapter.py       # Command-line interface adapter
│   │   └── python_adapter.py    # Direct Python API adapter
│   │
│   ├── tools/                   # Music analysis tools (13 total)
│   │   ├── base_tool.py         # Base class for all tools
│   │   ├── import_tool.py       # Score import
│   │   ├── export_tool.py       # Score export
│   │   ├── list_tool.py         # List scores
│   │   ├── delete_tool.py       # Delete scores
│   │   ├── score_info_tool.py   # Score information
│   │   ├── key_analysis_tool.py # Key signature analysis
│   │   ├── chord_analysis_tool.py      # Chord progression analysis
│   │   ├── harmony_analysis_tool.py    # Roman numeral analysis
│   │   ├── voice_leading_tool.py       # Voice leading analysis
│   │   ├── pattern_recognition_tool.py # Pattern detection
│   │   ├── harmonization_tool.py       # Automatic harmonization
│   │   ├── counterpoint_tool.py        # Counterpoint generation
│   │   └── style_imitation_tool.py     # Style imitation
│   │
│   ├── exceptions.py            # Custom exception hierarchy
│   ├── resource_manager.py      # Memory and score management
│   ├── health_checks.py         # Health check system
│   ├── observability.py         # Logging and metrics
│   ├── rate_limiter.py          # Rate limiting
│   ├── performance_cache.py     # High-performance caching
│   ├── performance_optimizations.py  # Optimized analysis tools
│   ├── async_executor.py        # Async/thread execution
│   └── parallel_processor.py    # Parallel batch processing
│
├── tests/
│   ├── unit/                    # Unit tests for tools
│   ├── integration/             # Integration tests
│   ├── core/                    # Core service tests
│   └── conftest.py              # Pytest fixtures
│
├── docs/                        # Documentation
├── examples/                    # Usage examples
├── scripts/                     # Utility scripts
└── .github/workflows/           # CI/CD pipelines
```

## Architecture Principles

### 1. Protocol-Independent Core

The `MusicAnalysisService` class in `services.py` contains all music analysis logic,
completely isolated from any protocol (MCP, HTTP, CLI). This means:

- Core functionality survives protocol changes
- Easy to test without protocol overhead
- Multiple interfaces share the same logic

### 2. Adapter Pattern

Each interface (MCP, HTTP, CLI, Python) has a dedicated adapter that:

- Translates protocol-specific requests to core service calls
- Handles protocol-specific error formatting
- Manages protocol-specific concerns (authentication, serialization)

### 3. Tool-Based Design

Each music analysis capability is encapsulated in a tool class:

- Tools are protocol-agnostic
- Tools share a common base class
- Tools can be composed and extended

## Data Flow

```
Client Request
      │
      ▼
┌─────────────┐
│   Adapter   │  (MCP/HTTP/CLI/Python)
└─────────────┘
      │
      ▼
┌─────────────────────┐
│ MusicAnalysisService│  (Core service)
└─────────────────────┘
      │
      ▼
┌─────────────┐
│    Tool     │  (Key analysis, chord analysis, etc.)
└─────────────┘
      │
      ▼
┌─────────────┐
│   music21   │  (Music analysis library)
└─────────────┘
      │
      ▼
    Result
```

## Entry Points

| Interface | Entry Point | Command |
|-----------|-------------|---------|
| MCP | `server_minimal.py` | `python -m music21_mcp.server_minimal` |
| HTTP | `http_adapter.py` | `python -m music21_mcp.launcher http` |
| CLI | `cli_adapter.py` | `python -m music21_mcp.launcher cli` |
| Python | `python_adapter.py` | `from music21_mcp import create_sync_analyzer` |
| All | `launcher.py` | `python -m music21_mcp.launcher` |

## Configuration

Configuration uses environment variables with `MUSIC21_` prefix:

- `MUSIC21_MCP_HOST` / `MUSIC21_MCP_PORT` — server binding
- `MUSIC21_MCP_TIMEOUT` / `MUSIC21_TOOL_TIMEOUT` — operation timeouts
- `MUSIC21_CORS_ORIGINS` — HTTP CORS policy
- See `.env.example` for the full list with defaults

## Resource Management

The `ResourceManager` class handles:

- Score storage with TTL expiration
- Memory limits and automatic cleanup
- Background cleanup threads
- Health monitoring
