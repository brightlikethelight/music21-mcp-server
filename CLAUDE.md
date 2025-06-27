# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a music21 MCP (Model Context Protocol) server that provides music analysis and composition capabilities through the music21 library. The project is currently in the planning phase with an architecture defined but no implementation yet.

## Project Setup

Since the project is not yet implemented, you'll need to initialize it:

1. Create a `pyproject.toml` with music21 and MCP dependencies
2. Set up the directory structure as outlined in `project_architecture_overview.md`
3. Initialize the main MCP server in `src/music21_mcp/server.py`

## Architecture

The project follows a modular, phased implementation approach:

### Phase 1: Core Music Analysis
- **Score I/O** (`core/score_parser.py`): Handles reading/writing MusicXML, MIDI, ABC notation, and other formats
- **Music Theory** (`core/theory_analyzer.py`): Key detection, scale analysis, interval calculations
- **Rhythm Analysis** (`core/rhythm_analyzer.py`): Tempo detection, meter analysis, rhythmic pattern recognition

### Phase 2: Advanced Analysis
- **Harmony** (`core/harmonic_analyzer.py`): Chord progression analysis, Roman numeral analysis, voice leading rules
- **Melody** (`core/melodic_analyzer.py`): Melodic contour analysis, phrase detection, motif identification
- **Voice Leading** (`core/voice_analyzer.py`): Part-writing analysis, voice independence metrics

### Phase 3: Creative Tools
- **Composition** (`core/composer.py`): Melody generation, harmonization, counterpoint generation
- **Orchestration** (`core/orchestrator.py`): Instrumentation suggestions, range checking, idiomatic writing

### Supporting Modules
- **Analysis Module** (`analysis/`): Pattern recognition, style classification, statistical analysis
- **Visualization Module** (`visualization/`): Score rendering, analysis plots, interactive displays
- **Utilities** (`utils/`): Music validators, format converters, cache management

## Development Commands

Once the project is set up with a proper `pyproject.toml`, typical commands would be:

```bash
# Install dependencies
pip install -e .

# Run tests (once pytest is configured)
pytest tests/

# Run the MCP server
python -m music21_mcp.server
```

## Key Implementation Notes

1. All music analysis functions should leverage the music21 library's capabilities
2. The MCP server should expose tools for each major analysis type
3. Cache frequently accessed scores to improve performance
4. Support multiple input/output formats through the format converters
5. Implement proper error handling for invalid musical input
6. Use music21's Stream objects as the primary data structure

## Testing Strategy

Tests should be organized by:
- **Unit tests** (`tests/unit/`): Test individual analyzers and utilities
- **Integration tests** (`tests/integration/`): Test complete workflows
- **Performance tests** (`tests/performance/`): Benchmark analysis speed
- **Test data** (`tests/test_data/`): Include sample MIDI, MusicXML, and ABC files

## MCP Server Implementation

The main server (`src/music21_mcp/server.py`) should:
1. Register tools for each analysis capability
2. Handle score input/output through various formats
3. Provide clear, musical terminology in responses
4. Support batch operations for analyzing multiple scores
5. Implement proper resource management for music21 objects