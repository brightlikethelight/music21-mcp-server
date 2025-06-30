# Music21 MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Music21](https://img.shields.io/badge/music21-latest-green.svg)](https://github.com/cuthbertLab/music21)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A stable and production-ready Model Context Protocol (MCP) server that provides powerful music analysis capabilities through the music21 library. Analyze scores, detect keys, identify chords, and export to various formats - all through a simple API.

## ✨ Features

- 🎵 **Import Music** from multiple sources (files, corpus, text notation)
- 🔍 **Analyze Key** with confidence scoring using advanced algorithms
- 🎸 **Detect Chords** and progressions with Roman numeral analysis
- 📊 **Extract Metadata** including tempo, time signatures, and instrumentation
- 💾 **Export Scores** to MIDI, MusicXML, ABC notation, and more
- 🚀 **100% Reliable** with comprehensive error handling
- 🧪 **Well Tested** with full test coverage

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/music21-mcp-server.git
cd music21-mcp-server

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

### Basic Usage

```python
# Run as a script
python -m music21_mcp.server

# Or use in your code
from music21_mcp.server import import_score, analyze_key, analyze_chords

# Import a Bach chorale
await import_score("bach", "bach/bwv66.6")

# Analyze its key
result = await analyze_key("bach")
print(f"Key: {result['key']} (confidence: {result['confidence']:.2%})")
```

### Claude Desktop Integration

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "music21": {
      "command": "python",
      "args": ["-m", "music21_mcp.server"],
      "cwd": "/path/to/music21-mcp-server"
    }
  }
}
```

## 📖 Documentation

- [Getting Started Guide](docs/getting-started.md) - Installation and first steps
- [API Reference](docs/simplified-api.md) - Complete tool documentation
- [Examples](examples/) - Code examples and use cases
- [Architecture](docs/architecture.md) - Technical design and structure

## 🛠️ Available Tools

| Tool | Description | Example |
|------|-------------|---------|
| **`import_score`** | Import music from various sources | `import_score("id", "bach/bwv66.6")` |
| **`list_scores`** | List all loaded scores | `list_scores()` |
| **`analyze_key`** | Detect the musical key | `analyze_key("id")` |
| **`analyze_chords`** | Extract chord progressions | `analyze_chords("id")` |
| **`get_score_info`** | Get comprehensive metadata | `get_score_info("id")` |
| **`export_score`** | Export to various formats | `export_score("id", "midi")` |
| **`delete_score`** | Remove scores from memory | `delete_score("id")` |

## 📚 Examples

### Analyzing a Classical Piece

```python
# Import a Mozart sonata
await import_score("mozart", "mozart/k331/movement1")

# Get detailed information
info = await get_score_info("mozart")
print(f"Composer: {info['metadata']['composer']}")
print(f"Duration: {info['duration_seconds']:.1f} seconds")

# Analyze harmony
chords = await analyze_chords("mozart")
print(f"Chord progression: {chords['chord_progression'][:5]}")
```

### Working with Custom Music

```python
# Create music from text notation
await import_score("melody", "C4 E4 G4 C5 G4 E4 C4")

# Analyze and export
key = await analyze_key("melody")
await export_score("melody", format="musicxml", output_path="my_melody.xml")
```

### Batch Processing

```python
# Process multiple Bach chorales
chorales = ["bach/bwv66.6", "bach/bwv7.7", "bach/bwv10.7"]

for i, chorale in enumerate(chorales):
    await import_score(f"chorale_{i}", chorale)
    result = await analyze_key(f"chorale_{i}")
    print(f"{chorale}: {result['key']}")
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python run_tests.py

# Run specific test file
pytest tests/test_server_pytest.py -v

# Run with coverage
pytest --cov=src/music21_mcp --cov-report=html
```

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run code formatting
black src/ tests/

# Run linting
flake8 src/ tests/

# Run type checking
mypy src/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Music21](https://web.mit.edu/music21/) for the incredible music analysis library
- [MCP](https://modelcontextprotocol.io/) for the Model Context Protocol specification
- [FastMCP](https://github.com/jlowin/fastmcp) for the MCP server framework

## 📞 Support

- 📚 [Documentation](docs/)
- 🐛 [Issue Tracker](https://github.com/yourusername/music21-mcp-server/issues)
- 💬 [Discussions](https://github.com/yourusername/music21-mcp-server/discussions)

---

<p align="center">Made with ❤️ for music and AI</p>