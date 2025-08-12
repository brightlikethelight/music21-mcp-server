# Music21 Analysis - Multi-Interface Music Server

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/music21-mcp-server.svg)](https://badge.fury.io/py/music21-mcp-server)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![MCP](https://img.shields.io/badge/MCP-Model%20Context%20Protocol-green)](https://modelcontextprotocol.io)

**Professional music analysis with 4 different interfaces** - MCP server, HTTP API, CLI tools, and Python library. Built on the powerful music21 library with protocol-independent architecture for maximum reliability.

## 🎯 Why Multiple Interfaces?

Based on 2025 research showing **MCP has 40-50% production success rate**, this project provides **multiple pathways** to the same powerful music21 analysis functionality:

- 📡 **MCP Server** - For Claude Desktop integration (when it works)
- 🌐 **HTTP API** - For web applications (reliable backup) 
- 💻 **CLI Tools** - For automation (always works)
- 🐍 **Python Library** - For direct programming access

## 🎵 Core Music Analysis Features

### Analysis Tools (13 Available)
- **Import & Export**: MusicXML, MIDI, ABC, Lilypond, music21 corpus
- **Key Analysis**: Multiple algorithms (Krumhansl, Aarden, Bellman-Budge)
- **Harmony Analysis**: Roman numerals, chord progressions, cadence detection
- **Voice Leading**: Parallel motion detection, voice crossing analysis
- **Pattern Recognition**: Melodic, rhythmic, and harmonic patterns

### Advanced Capabilities  
- **Harmonization**: Bach chorale and jazz style harmonization
- **Counterpoint**: Species counterpoint generation (1-5)
- **Style Imitation**: Learn and generate music in composer styles
- **Score Manipulation**: Transposition, time stretching, orchestration

## 🚀 Quick Start

### Installation

#### Quick Install from PyPI (Recommended)
```bash
# Install the latest stable version
pip install music21-mcp-server

# Configure music21 corpus (first time only)
python -m music21.configure
```

#### Development Install
```bash
# Clone repository
git clone https://github.com/brightlikethelight/music21-mcp-server.git
cd music21-mcp-server

# Install with UV (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Or with pip
pip install -r requirements.txt

# Configure music21 corpus
python -m music21.configure
```

### Usage - Pick Your Interface

#### 🎯 Show All Available Interfaces
```bash
python -m music21_mcp.launcher
```

#### 📡 MCP Server (for Claude Desktop)

##### Automated Setup (Easiest)
```bash
# Run the setup script
python setup_claude_desktop.py

# This automatically configures Claude Desktop for you
```

##### Manual Setup
```bash
# Start MCP server
python -m music21_mcp.launcher mcp

# Configure Claude Desktop with:
# ~/.config/claude-desktop/config.json (macOS/Linux)
# %APPDATA%/Claude/claude_desktop_config.json (Windows)
{
  "mcpServers": {
    "music21-mcp-server": {
      "command": "python",
      "args": ["-m", "music21_mcp"],
      "env": {
        "MUSIC21_MCP_LOG_LEVEL": "INFO",
        "MUSIC21_MCP_MAX_MEMORY_MB": "512",
        "MUSIC21_MCP_MAX_SCORES": "100"
      }
    }
  }
}
```

#### 🌐 HTTP API Server (for web apps)
```bash
# Start HTTP API server
python -m music21_mcp.launcher http
# Opens: http://localhost:8000
# API docs: http://localhost:8000/docs

# Customize host/port with environment variables
export MUSIC21_MCP_HOST=0.0.0.0
export MUSIC21_MCP_PORT=8080
python -m music21_mcp.launcher http

# Example usage:
curl -X POST "http://localhost:8000/scores/import" \
  -H "Content-Type: application/json" \
  -d '{"score_id": "chorale", "source": "bach/bwv66.6", "source_type": "corpus"}'

curl -X POST "http://localhost:8000/analysis/key" \
  -H "Content-Type: application/json" \
  -d '{"score_id": "chorale"}'
```

#### 💻 CLI Tools (for automation)
```bash
# Show CLI status
python -m music21_mcp.launcher cli

# Import a score
music21-import --id my_score --source "bach/bwv66.6" --type corpus

# Analyze key
music21-analyze-key --score my_score

# Export to MusicXML
music21-export --score my_score --format musicxml --output my_score.xml
```

#### 🐍 Python Library (for programming)
```python
from music21_mcp import create_music_analyzer

# Async usage
analyzer = await create_music_analyzer()
result = await analyzer.import_score("my_score", "bach/bwv66.6", "corpus")
key_result = await analyzer.analyze_key("my_score")

# Sync usage
from music21_mcp import create_sync_analyzer
analyzer = create_sync_analyzer()
result = analyzer.import_score("my_score", "bach/bwv66.6", "corpus")
```

## 📚 Documentation

- **[Quick Start Tutorial](examples/notebooks/quickstart_tutorial.ipynb)** - 5-minute Jupyter notebook tutorial
- **[Claude Desktop Setup](CLAUDE_DESKTOP_SETUP.md)** - Automated setup for Claude Desktop
- **[API Reference](docs/api_reference.md)** - Complete API documentation
- **[Examples](examples/)** - Working code examples
- **[Performance Guide](docs/performance.md)** - Optimization tips

## 🎯 Example: Analyze Bach Chorale

```python
# Using the Python library
from music21_mcp import create_sync_analyzer

analyzer = create_sync_analyzer()

# Import Bach chorale
analyzer.import_score("bach", "bach/bwv66.6", "corpus")

# Analyze key
key = analyzer.analyze_key("bach")
print(f"Key: {key['key']} (confidence: {key['confidence']}%)")

# Analyze harmony
harmony = analyzer.analyze_harmony("bach")
for chord in harmony['roman_numerals'][:10]:
    print(f"  {chord}")

# Check voice leading
voice_leading = analyzer.analyze_voice_leading("bach")
print(f"Voice leading score: {voice_leading['overall_score']}/100")
```

## 🏗️ Architecture

The project uses a **protocol-independent core** that survives breaking changes:

```
┌─────────────────┐
│  Music21 Core   │  ← Pure music analysis logic
└────────┬────────┘
         │
    ┌────┴────┐
    │ Service │     ← Protocol-independent service layer
    └────┬────┘
         │
    ┌────┴────────────────────────┐
    │        Adapters              │
    ├──────────┬─────────┬────────┤
    │   MCP    │  HTTP   │  CLI   │  ← Multiple interfaces
    └──────────┴─────────┴────────┘
```

## 🔧 Configuration

Environment variables for customization:

```bash
# Logging
export MUSIC21_MCP_LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR

# Performance
export MUSIC21_MCP_MAX_MEMORY_MB=512  # Memory limit
export MUSIC21_MCP_MAX_SCORES=100     # Max loaded scores

# Network (HTTP interface)
export MUSIC21_MCP_HOST=0.0.0.0       # Server host
export MUSIC21_MCP_PORT=8000          # Server port
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=music21_mcp --cov-report=html

# Run specific test file
pytest tests/test_chord_analysis.py

# Run quickstart tutorial validation
python test_quickstart_tutorial.py
```

## 📈 Performance

The server includes advanced performance optimizations:

- **Caching**: Roman numeral analysis cached (14.7s → <1s)
- **Parallel Processing**: Multi-threaded chord analysis
- **Async Architecture**: Non-blocking event loop (44.9% faster)
- **Resource Management**: Automatic memory cleanup with TTL cache
- **Structured Logging**: JSON logs with correlation IDs

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🌟 Credits

Built on the amazing [music21](http://web.mit.edu/music21/) library by MIT.

## 🔗 Links

- **PyPI**: [https://pypi.org/project/music21-mcp-server/](https://pypi.org/project/music21-mcp-server/)
- **GitHub**: [https://github.com/brightlikethelight/music21-mcp-server](https://github.com/brightlikethelight/music21-mcp-server)
- **MCP Registry**: [Coming Soon]
- **Documentation**: [Coming Soon]

---

**Made with ♪ by the Music21 MCP Team**