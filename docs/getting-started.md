# Getting Started with Music21 MCP Server

Welcome to the Music21 MCP Server! This guide will help you get up and running quickly.

## Prerequisites

Before you begin, ensure you have:

- Python 3.8 or higher installed
- Basic familiarity with Python async/await
- (Optional) MCP client for MCP integration

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/music21-mcp-server.git
cd music21-mcp-server
```

### 2. Set Up Virtual Environment

It's recommended to use a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install the Package

```bash
# Install in development mode
pip install -e .

# Or install just the requirements
pip install -r requirements.txt
```

## First Steps

### Running the Server

You can run the server in several ways:

```bash
# As a module
python -m music21_mcp.server

# Or directly
python src/music21_mcp/server.py
```

### Basic Usage Example

Create a file `example.py`:

```python
import asyncio
from music21_mcp.server import (
    import_score, analyze_key, analyze_chords, 
    export_score, score_manager
)

async def main():
    # Import a Bach chorale from the corpus
    result = await import_score("bach", "bach/bwv66.6")
    print(f"Imported: {result}")
    
    # Analyze the key
    key_result = await analyze_key("bach")
    print(f"Key: {key_result['key']} (confidence: {key_result['confidence']:.2%})")
    
    # Analyze chords
    chord_result = await analyze_chords("bach")
    print(f"Found {chord_result['total_chords']} chords")
    
    # Export to MIDI
    export_result = await export_score("bach", format="midi")
    print(f"Exported to: {export_result['file_path']}")

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:

```bash
python example.py
```

## Understanding the Tools

### 1. import_score

Import music from various sources:

```python
# From corpus
await import_score("bach", "bach/bwv66.6")

# From text notation
await import_score("melody", "C4 D4 E4 F4 G4 A4 B4 C5")

# From file
await import_score("mysong", "/path/to/file.mid")
```

### 2. analyze_key

Detect the musical key with confidence scoring:

```python
result = await analyze_key("score_id")
# Returns: {'status': 'success', 'key': 'G major', 'confidence': 0.89}
```

### 3. analyze_chords

Extract chord progressions:

```python
result = await analyze_chords("score_id")
# Returns chord progression, Roman numerals, and more
```

### 4. export_score

Export to various formats:

```python
# Export formats: midi, musicxml, abc, lilypond
await export_score("score_id", format="midi", output_path="output.mid")
```

## MCP client Integration

### 1. Configure MCP client

Add to your MCP client config file:

**macOS**: `~/Library/Application Support/Claude/mcp_config.json`
**Windows**: `%APPDATA%\Claude\mcp_config.json`

```json
{
  "mcpServers": {
    "music21": {
      "command": "python",
      "args": ["-m", "music21_mcp.server"],
      "cwd": "/absolute/path/to/music21-mcp-server"
    }
  }
}
```

### 2. Restart MCP client

After saving the configuration, restart MCP client.

### 3. Use in Claude

You can now use music analysis tools in Claude:

```
Please analyze the key of Bach's BWV 66.6
```

## Common Use Cases

### Analyzing Multiple Pieces

```python
pieces = ["bach/bwv66.6", "mozart/k331/movement1", "beethoven/opus18no1/movement1"]

for i, piece in enumerate(pieces):
    await import_score(f"piece_{i}", piece)
    key = await analyze_key(f"piece_{i}")
    print(f"{piece}: {key['key']}")
```

### Creating Simple Melodies

```python
# Major scale
await import_score("major_scale", "C4 D4 E4 F4 G4 A4 B4 C5")

# Arpeggio
await import_score("arpeggio", "C4 E4 G4 C5 G4 E4 C4")
```

### Batch Export

```python
formats = ["midi", "musicxml", "abc"]
for fmt in formats:
    result = await export_score("score_id", format=fmt)
    print(f"Exported {fmt}: {result['file_path']}")
```

## Troubleshooting

### Common Issues

**ImportError: No module named 'music21'**
- Solution: `pip install music21`

**Score not found error**
- Make sure you've imported the score first
- Check the score_id matches exactly

**Export fails**
- Ensure you have write permissions
- Check disk space
- Some formats (like PDF) require external tools

### Getting Help

- Check the [API Reference](simplified-api.md)
- Browse [Examples](../examples/)
- Open an [Issue](https://github.com/yourusername/music21-mcp-server/issues)

## Next Steps

- Explore the [full API documentation](simplified-api.md)
- Try the [example scripts](../examples/)
- Learn about the [architecture](architecture.md)
- Contribute to the project!

Happy music analysis! 🎵