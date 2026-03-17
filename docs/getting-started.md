# Getting Started

This guide helps you get up and running with Music21 MCP Server.

## Prerequisites

- Python 3.10 or higher
- pip or uv package manager

## Installation

### From PyPI (Recommended)

```bash
pip install music21-mcp-server
```

### From Source

```bash
# Clone the repository
git clone https://github.com/brightlikethelight/music21-mcp-server.git
cd music21-mcp-server

# Install with uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Or with pip
pip install -e .
```

## Configure music21

Run the music21 configuration wizard (one-time setup):

```bash
python -m music21.configure
```

This sets up the music21 corpus and environment.

## Quick Start

### 1. Python Library (Simplest)

```python
from music21_mcp import create_sync_analyzer

# Create analyzer
analyzer = create_sync_analyzer()

# Import a Bach chorale from the corpus
analyzer.import_score("chorale", "bach/bwv66.6", "corpus")

# Analyze the key
result = analyzer.analyze_key("chorale")
print(f"Key: {result}")

# Get harmony analysis
harmony = analyzer.analyze_harmony("chorale", "roman")
print(f"Harmony: {harmony}")
```

### 2. Command Line Interface

```bash
# Show available commands
python -m music21_mcp.launcher cli --help

# Import a score
python -m music21_mcp.launcher cli import chorale bach/bwv66.6 corpus

# Analyze the key
python -m music21_mcp.launcher cli key-analysis chorale

# List all loaded scores
python -m music21_mcp.launcher cli list
```

### 3. HTTP API Server

```bash
# Start the HTTP server
python -m music21_mcp.launcher http

# Server runs at http://localhost:8000
# API docs at http://localhost:8000/docs
```

Use curl or any HTTP client:

```bash
# Import a score
curl -X POST "http://localhost:8000/scores/import" \
  -H "Content-Type: application/json" \
  -d '{"score_id": "chorale", "source": "bach/bwv66.6", "source_type": "corpus"}'

# Analyze the key
curl -X POST "http://localhost:8000/analysis/key" \
  -H "Content-Type: application/json" \
  -d '{"score_id": "chorale"}'
```

### 4. MCP Server (for Claude Desktop)

```bash
# Start the MCP server
python -m music21_mcp.server_minimal
```

Add to Claude Desktop config (`~/.config/claude-desktop/config.json`):

```json
{
  "mcpServers": {
    "music21-analysis": {
      "command": "python",
      "args": ["-m", "music21_mcp.server_minimal"],
      "env": {
        "PYTHONPATH": "/path/to/music21-mcp-server/src"
      }
    }
  }
}
```

## Available Tools

The server provides 13 music analysis tools:

| Tool | Description |
|------|-------------|
| `import_score` | Import from corpus, file, or URL |
| `list_scores` | List all loaded scores |
| `score_info` | Get score metadata |
| `export_score` | Export to MIDI, MusicXML, etc. |
| `delete_score` | Remove a score from memory |
| `key_analysis` | Analyze key signature |
| `chord_analysis` | Analyze chord progressions |
| `harmony_analysis` | Roman numeral analysis |
| `voice_leading_analysis` | Voice leading quality analysis |
| `pattern_recognition` | Find melodic/rhythmic patterns |
| `harmonize_melody` | Generate harmonization |
| `generate_counterpoint` | Create counterpoint |
| `imitate_style` | Style imitation |

## Next Steps

- See [Architecture](architecture.md) for system design
- Check [examples/](../examples/) for more code samples
- Read API docs at http://localhost:8000/docs when running HTTP server
