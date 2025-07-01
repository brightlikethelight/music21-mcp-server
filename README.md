# Music21 MCP Server

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![MCP](https://img.shields.io/badge/MCP-Model%20Context%20Protocol-green)](https://modelcontextprotocol.io)

A production-ready Model Context Protocol (MCP) server that provides comprehensive music analysis and generation capabilities through the music21 library.

## 🎵 Features

### Core Analysis Tools
- **Import & Export**: Support for MusicXML, MIDI, ABC, Lilypond, and music21 corpus
- **Key Analysis**: Multiple algorithms (Krumhansl, Aarden, Bellman-Budge)
- **Harmony Analysis**: Chord progressions, Roman numerals, cadence detection
- **Voice Leading**: Parallel detection, voice crossing analysis, smoothness metrics
- **Pattern Recognition**: Melodic, rhythmic, and harmonic pattern identification

### Advanced Capabilities
- **Harmonization**: Bach chorale and jazz style harmonization
- **Counterpoint Generation**: Species counterpoint (1-5) following strict rules
- **Style Imitation**: Learn and generate music in the style of input pieces
- **Score Manipulation**: Transposition, time stretching, orchestration

### Production Features
- **Circuit Breakers**: Automatic failure recovery within 60 seconds
- **Rate Limiting**: Token bucket algorithm for resource protection
- **Memory Management**: Automatic garbage collection and score caching
- **Health Monitoring**: Real-time metrics and health checks
- **Graceful Shutdown**: Clean resource cleanup on termination

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/music21-mcp-server.git
cd music21-mcp-server

# Install with pip
pip install -e .

# Or with Poetry
poetry install
```

### Basic Usage

```bash
# Start the server
python -m music21_mcp.server

# Or use the CLI
music21-mcp serve
```

### Docker

```bash
# Build image
docker build -t music21-mcp-server .

# Run container
docker run -p 8000:8000 music21-mcp-server
```

## 📖 API Reference

### Import Score
Import a musical score from various sources.

```json
{
  "tool": "import_score",
  "arguments": {
    "score_id": "bach_invention_1",
    "source": "bach/inventions/invent1",
    "source_type": "corpus"
  }
}
```

### Analyze Key
Determine the key of a musical piece.

```json
{
  "tool": "analyze_key",
  "arguments": {
    "score_id": "bach_invention_1",
    "algorithm": "krumhansl"
  }
}
```

### Generate Counterpoint
Create counterpoint following classical rules.

```json
{
  "tool": "generate_counterpoint",
  "arguments": {
    "cantus_firmus_id": "cantus",
    "output_score_id": "counterpoint_result",
    "species": 1,
    "mode": "major"
  }
}
```

[View Full API Documentation →](docs/api/)

## 🏗️ Architecture

The server follows a modular architecture with clear separation of concerns:

```
src/music21_mcp/
├── server.py          # Main FastMCP server with resilience features
├── tools/             # Individual tool implementations
│   ├── import_tool.py
│   ├── key_analysis_tool.py
│   └── ...
├── core/              # Core analyzers and algorithms
│   ├── harmonic_analyzer.py
│   ├── melodic_analyzer.py
│   └── ...
└── resilience.py      # Production resilience patterns
```

### Resilience Patterns

- **Circuit Breaker**: Prevents cascading failures with configurable thresholds
- **Rate Limiter**: Token bucket algorithm with burst support
- **Resource Pool**: Connection pooling with health checks
- **Memory Guard**: Automatic cleanup at configurable thresholds

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/music21_mcp

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests
pytest -m "not slow"    # Skip slow tests
```

### Test Coverage
- Unit tests for all tools and analyzers
- Integration tests for tool interactions
- Performance tests for stress scenarios
- Property-based tests for algorithmic correctness

## 🚢 Deployment

### Environment Variables

```bash
MUSIC21_MCP_HOST=0.0.0.0
MUSIC21_MCP_PORT=8000
MUSIC21_MCP_MAX_SCORES=100
MUSIC21_MCP_MEMORY_LIMIT_MB=2048
MUSIC21_MCP_LOG_LEVEL=INFO
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: music21-mcp-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: music21-mcp
  template:
    spec:
      containers:
      - name: server
        image: music21-mcp-server:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
```

## 📊 Monitoring

### Health Check Endpoint
```bash
GET /health
```

### Metrics Endpoint
```bash
GET /metrics  # Prometheus format
```

### Example Metrics
- `music21_requests_total`: Total requests processed
- `music21_request_duration_seconds`: Request latency histogram
- `music21_scores_in_memory`: Number of scores currently cached
- `music21_circuit_breaker_state`: Circuit breaker states

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
poetry install --with dev

# Run code formatters
black src tests
ruff check src tests --fix

# Run type checking
mypy src
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [music21](http://web.mit.edu/music21/) - The amazing music analysis library
- [Model Context Protocol](https://modelcontextprotocol.io) - The MCP specification
- [FastMCP](https://github.com/anthropics/fastmcp) - Fast MCP implementation

## 📬 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/music21-mcp-server/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/music21-mcp-server/discussions)
- **Email**: support@example.com

---

Built with ❤️ for the music and AI community