# MCP Registry Installation Guide - Music21 Analysis Server

## Overview

The Music21 MCP Server provides professional music analysis capabilities through the Model Context Protocol. This guide covers installation for various MCP-compatible applications.

## Prerequisites

- **Python**: 3.10 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 512MB RAM, recommended 2GB for large scores
- **Storage**: 200MB for base installation, additional space for music files

## Quick Installation

### Method 1: Direct Installation (Recommended)

```bash
# Install directly from GitHub
pip install git+https://github.com/brightlikethelight/music21-mcp-server.git

# Configure music21 corpus (required for full functionality)
python -m music21.configure
```

### Method 2: Clone and Install

```bash
# Clone the repository
git clone https://github.com/brightlikethelight/music21-mcp-server.git
cd music21-mcp-server

# Install with UV (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Or with pip
pip install -r requirements.txt

# Configure music21
python -m music21.configure
```

## MCP Application Configurations

### Claude Desktop

Add the following configuration to your Claude Desktop config file:

**Location**: `~/.config/claude-desktop/config.json` (Linux/macOS) or `%APPDATA%\Claude\config.json` (Windows)

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

**Note**: Replace `/path/to/music21-mcp-server/src` with the actual path to your installation.

### VS Code with MCP Extension

1. Install the MCP extension for VS Code
2. Add to your VS Code settings.json:

```json
{
  "mcp.servers": {
    "music21-analysis": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "music21_mcp.server_minimal"],
      "cwd": "/path/to/music21-mcp-server"
    }
  }
}
```

### Cursor IDE

Add to your Cursor configuration:

```json
{
  "mcp": {
    "servers": {
      "music21-analysis": {
        "command": "python",
        "args": ["-m", "music21_mcp.server_minimal"],
        "env": {
          "PYTHONPATH": "/path/to/music21-mcp-server/src"
        }
      }
    }
  }
}
```

### Zed Editor

Add to your Zed settings:

```json
{
  "experimental": {
    "mcp": {
      "servers": {
        "music21-analysis": {
          "command": "python",
          "args": ["-m", "music21_mcp.server_minimal"]
        }
      }
    }
  }
}
```

### Continue.dev

Add to your Continue configuration:

```json
{
  "mcpServers": {
    "music21-analysis": {
      "command": "python",
      "args": ["-m", "music21_mcp.server_minimal"]
    }
  }
}
```

## Platform-Specific Instructions

### macOS

```bash
# Install Python using Homebrew (if not already installed)
brew install python@3.11

# Install the server
pip3 install git+https://github.com/brightlikethelight/music21-mcp-server.git

# Configure music21
python3 -m music21.configure

# Test installation
python3 -m music21_mcp.server_minimal --help
```

### Linux (Ubuntu/Debian)

```bash
# Install Python and dependencies
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Create virtual environment (recommended)
python3 -m venv music21-mcp-env
source music21-mcp-env/bin/activate

# Install the server
pip install git+https://github.com/brightlikethelight/music21-mcp-server.git

# Configure music21
python -m music21.configure
```

### Windows

```powershell
# Install Python from python.org or use winget
winget install Python.Python.3.11

# Create virtual environment (recommended)
python -m venv music21-mcp-env
music21-mcp-env\Scripts\activate

# Install the server
pip install git+https://github.com/brightlikethelight/music21-mcp-server.git

# Configure music21
python -m music21.configure
```

## Docker Installation

For containerized deployment:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install the server
RUN pip install git+https://github.com/brightlikethelight/music21-mcp-server.git

# Configure music21
RUN python -m music21.configure

# Expose MCP stdio interface
CMD ["python", "-m", "music21_mcp.server_minimal"]
```

Build and run:

```bash
docker build -t music21-mcp-server .
docker run -i music21-mcp-server
```

## Configuration Options

### Environment Variables

- `MUSIC21_CORPUS_PATH`: Custom path for music21 corpus files
- `MCP_LOG_LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `MCP_MAX_SCORES`: Maximum number of scores to keep in memory (default: 10)
- `MCP_CACHE_SIZE`: Cache size for analysis results (default: 100MB)

### Advanced Configuration

Create a `.env` file in your project directory:

```env
MUSIC21_CORPUS_PATH=/path/to/custom/corpus
MCP_LOG_LEVEL=INFO
MCP_MAX_SCORES=20
MCP_CACHE_SIZE=200MB
MCP_TIMEOUT=60
```

## Verification

Test your installation:

```bash
# Test basic functionality
python -m music21_mcp.launcher test

# Test MCP server
python -m music21_mcp.server_minimal &
echo '{"jsonrpc": "2.0", "method": "tools/list", "id": 1}' | python -m music21_mcp.server_minimal

# Test with sample score
python -c "
from music21_mcp.tools import ImportScoreTool
tool = ImportScoreTool()
result = tool.execute(score_id='test', source='bach/bwv66.6', source_type='corpus')
print('Success!' if result['success'] else 'Failed!')
"
```

## Troubleshooting

### Common Issues

#### "Module not found" errors
```bash
# Ensure proper Python path
export PYTHONPATH="/path/to/music21-mcp-server/src:$PYTHONPATH"

# Or reinstall with pip -e for development
pip install -e .
```

#### Music21 corpus issues
```bash
# Reconfigure music21
python -m music21.configure

# Or set manual corpus path
export MUSIC21_CORPUS_PATH="/path/to/music21/corpus"
```

#### Permission errors on macOS/Linux
```bash
# Use user installation
pip install --user git+https://github.com/brightlikethelight/music21-mcp-server.git
```

#### Memory issues with large scores
```bash
# Increase memory limits
export MCP_MAX_SCORES=5
export MCP_CACHE_SIZE=50MB
```

### Logging and Diagnostics

Enable detailed logging:

```bash
export MCP_LOG_LEVEL=DEBUG
python -m music21_mcp.server_minimal
```

Check server health:

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "health_check",
    "arguments": {}
  },
  "id": 1
}
```

### Performance Optimization

For better performance:

```bash
# Install optional dependencies
pip install music21-mcp-server[visualization,audio]

# Use parallel processing
export MCP_PARALLEL_PROCESSING=true

# Optimize for memory usage
export MCP_MEMORY_OPTIMIZATION=true
```

## Multiple Interface Support

This server also provides HTTP API and CLI interfaces as fallbacks:

```bash
# HTTP API server (port 8000)
python -m music21_mcp.launcher http

# CLI interface
python -m music21_mcp.launcher cli

# Show all available interfaces
python -m music21_mcp.launcher
```

## Updating

To update to the latest version:

```bash
# Update from GitHub
pip install --upgrade git+https://github.com/brightlikethelight/music21-mcp-server.git

# Or if cloned locally
cd music21-mcp-server
git pull
pip install --upgrade -e .
```

## Uninstalling

```bash
# Remove the package
pip uninstall music21-mcp-server

# Clean up configuration (optional)
rm -rf ~/.music21rc
```

## Support

- **Issues**: https://github.com/brightlikethelight/music21-mcp-server/issues
- **Discussions**: https://github.com/brightlikethelight/music21-mcp-server/discussions
- **Email**: brightliu@college.harvard.edu

## Next Steps

After installation:

1. Read the [MCP Tools Documentation](MCP_TOOLS.md) for detailed tool usage
2. Check out [Example Usage Patterns](MCP_EXAMPLES.md) for common workflows
3. Explore the [API Documentation](API.md) for advanced usage
4. Join the [GitHub Discussions](https://github.com/brightlikethelight/music21-mcp-server/discussions) for community support