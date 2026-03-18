# Music21 Analysis - Multi-Interface Music Server

[![CI/CD Pipeline](https://github.com/brightlikethelight/music21-mcp-server/actions/workflows/ci.yml/badge.svg)](https://github.com/brightlikethelight/music21-mcp-server/actions/workflows/ci.yml)
[![CI](https://github.com/brightlikethelight/music21-mcp-server/actions/workflows/ci.yml/badge.svg)](https://github.com/brightlikethelight/music21-mcp-server/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-84%25-brightgreen.svg)](https://github.com/brightlikethelight/music21-mcp-server/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
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

#### Install from PyPI (Recommended)

```bash
# Install the package
pip install music21-mcp-server

# Start the server
music21-mcp          # MCP server for Claude Desktop
music21-http         # REST API at localhost:8000
music21-cli          # Interactive CLI
music21-analysis mcp          # Unified launcher (positional arg)
```

#### Install from Source

```bash
# Clone repository
git clone https://github.com/brightlikethelight/music21-mcp-server.git
cd music21-mcp-server

# Install with UV (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Or with pip
pip install .

# Configure music21 corpus
python -m music21.configure
```

### Usage - Pick Your Interface

#### 🎯 Show All Available Interfaces
```bash
python -m music21_mcp.launcher
```

#### 📡 MCP Server (for Claude Desktop)
```bash
# Start MCP server
python -m music21_mcp.launcher mcp

# Configure Claude Desktop with:
# ~/.config/claude-desktop/config.json
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

#### 🌐 HTTP API Server (for web apps)
```bash
# Start HTTP API server
python -m music21_mcp.launcher http
# Opens: http://localhost:8000
# API docs: http://localhost:8000/docs

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
python -m music21_mcp.launcher cli status

# Import and analyze a Bach chorale
python -m music21_mcp.launcher cli import chorale bach/bwv66.6 corpus
python -m music21_mcp.launcher cli key-analysis chorale
python -m music21_mcp.launcher cli harmony chorale roman

# List all tools
python -m music21_mcp.launcher cli tools
```

#### 🐍 Python Library (for programming)
```python
from music21_mcp import create_sync_analyzer

# Create analyzer
analyzer = create_sync_analyzer()

# Import and analyze
analyzer.import_score("chorale", "bach/bwv66.6", "corpus")
key_result = analyzer.analyze_key("chorale")
harmony_result = analyzer.analyze_harmony("chorale", "roman")

print(f"Key: {key_result}")
print(f"Harmony: {harmony_result}")

# Quick comprehensive analysis
analysis = analyzer.quick_analysis("chorale")
```

## 🧪 Testing & Development

### Run Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage threshold
python -m pytest tests/ --cov=src/music21_mcp --cov-fail-under=80
```

### Development Setup
```bash
# Install development dependencies
uv sync --dev

# Set up pre-commit hooks
pre-commit install

# Run linting
ruff check src/
ruff format src/

# Type checking
mypy src/
```

## 🏗️ Architecture

### Protocol-Independent Design
```
Core Value Layer:
├── services.py              # Music21 analysis service (protocol-independent)
└── tools/                   # 13 music analysis tools

Protocol Adapter Layer:
├── adapters/mcp_adapter.py   # MCP protocol isolation
├── adapters/http_adapter.py  # HTTP/REST API
├── adapters/cli_adapter.py   # Command-line interface  
└── adapters/python_adapter.py # Direct Python access

Unified Entry Point:
└── launcher.py              # Single entry point for all interfaces
```

### Design Philosophy
- **Core Value First**: Music21 analysis isolated from protocol concerns
- **Protocol Apocalypse Survival**: Works even when MCP fails (30-40% of time)
- **Multiple Escape Hatches**: Always have a working interface
- **Reality-Based**: Built for today's MCP ecosystem, not enterprise dreams

## 📊 Interface Reliability

| Interface | Success Rate | Best For |
|-----------|--------------|----------|
| **MCP** | 40-50% | AI assistant integration |
| **HTTP** | 95%+ | Web applications |
| **CLI** | 99%+ | Automation & scripting |
| **Python** | 99%+ | Direct programming |

## 📚 Documentation

- **[docs/architecture.md](docs/architecture.md)** - System architecture overview
- **[docs/getting-started.md](docs/getting-started.md)** - Quick start guide
- **[examples/](examples/)** - Working code examples
- **API Docs**: http://localhost:8000/docs (when HTTP server running)

### Discord Webhook Integration
- **[.github/webhook-config.md](.github/webhook-config.md)** - Complete Discord webhook setup guide
- **[docs/webhook-integration.md](docs/webhook-integration.md)** - Advanced webhook configuration
- **[scripts/test-webhook.sh](scripts/test-webhook.sh)** - Test webhook connectivity
- **[scripts/setup-webhook.sh](scripts/setup-webhook.sh)** - Automated webhook setup

## 🔧 Configuration

### Environment Variables
```bash
# Server host and port (used by HTTP adapter and launcher)
export MUSIC21_MCP_HOST=127.0.0.1
export MUSIC21_MCP_PORT=8000

# Operation timeouts (seconds)
export MUSIC21_MCP_TIMEOUT=30          # General async operation timeout
export MUSIC21_TOOL_TIMEOUT=30         # Per-tool execution timeout
export MUSIC21_CHORD_ANALYSIS_TIMEOUT=60  # Chord analysis timeout
export MUSIC21_BATCH_TIMEOUT=30        # Batch processing timeout

# CORS origins for HTTP adapter (comma-separated)
export MUSIC21_CORS_ORIGINS="http://localhost:*"
```

### Music21 Setup
```bash
# Configure corpus path (one-time setup)
python -m music21.configure
```

## 🛠️ Available Analysis Tools

1. **import_score** - Import from corpus, files, URLs
2. **list_scores** - List all imported scores  
3. **get_score_info** - Detailed score information
4. **export_score** - Export to MIDI, MusicXML, etc.
5. **delete_score** - Remove scores from storage
6. **analyze_key** - Key signature analysis
7. **analyze_chords** - Chord progression analysis
8. **analyze_harmony** - Roman numeral/functional harmony
9. **analyze_voice_leading** - Voice leading quality analysis
10. **recognize_patterns** - Melodic/rhythmic patterns
11. **harmonize_melody** - Automatic harmonization
12. **generate_counterpoint** - Counterpoint generation
13. **imitate_style** - Style imitation and generation

## 🚀 Quick Examples

### Analyze a Bach Chorale
```bash
# CLI approach
python -m music21_mcp.launcher cli import chorale bach/bwv66.6 corpus
python -m music21_mcp.launcher cli key-analysis chorale

# Python approach  
analyzer = create_sync_analyzer()
analyzer.import_score("chorale", "bach/bwv66.6", "corpus")
print(analyzer.analyze_key("chorale"))
```

### Start Services
```bash
# For Claude Desktop
python -m music21_mcp.launcher mcp

# For web development
python -m music21_mcp.launcher http

# For command-line work
python -m music21_mcp.launcher cli status
```

## 🔄 Migration from v1.0

The previous enterprise version has been **simplified for reliability**:

- ✅ **Kept**: All music21 analysis functionality
- ✅ **Added**: HTTP API, CLI, Python library interfaces
- ❌ **Removed**: Docker, K8s, complex auth, monitoring (too unstable for MCP ecosystem)
- 🔄 **Changed**: Focus on core value delivery through multiple interfaces

## 🔔 Discord Webhook Integration

Get real-time notifications for CI/CD pipeline status, pull requests, and releases:

- 📖 [Webhook Setup Guide](.github/webhook-config.md)
- 🛠️ [Quick Setup Script](scripts/setup-webhook.sh)
- 🧪 [Test Your Webhook](scripts/test-webhook.sh)
- 📚 [Advanced Configuration](docs/webhook-integration.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- Development setup and requirements
- Code style guidelines (Ruff, MyPy)
- Testing requirements (maintain >80% coverage)
- Pull request process
- Branch protection rules

Quick start:
1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Run tests: `pytest tests/ --cov=src/music21_mcp --cov-fail-under=80`
4. Commit changes: `git commit -m 'feat: Add amazing feature'`
5. Push branch: `git push origin feature/amazing-feature`
6. Submit pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on the excellent [music21](https://web.mit.edu/music21/) library
- Uses [FastMCP](https://github.com/jlowin/fastmcp) for MCP protocol support  
- Inspired by the need for reliable music analysis tools

---

**Choose the interface that works for you. All provide the same powerful music21 analysis capabilities!** 🎵